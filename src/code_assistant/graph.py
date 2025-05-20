"""Main entrypoint for the code assistant graph.

This module defines the core structure and functionality of the code assistant graph.
It includes the main graph definition, state management, and key functions for
processing documentation, generating code, testing code, and incorporating feedback
for improved solutions.
"""

from typing import Dict, List, Optional, cast

from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph, START

from code_assistant.configuration import Configuration
from code_assistant.state import CodeSolution, GraphState, InputState
from code_assistant.utils import (
    check_code_execution,
    check_imports,
    format_docs, 
    get_message_text,
    load_chat_model,
    extract_required_modules,
    documentation_exists,
    ingest_github_repo,
    get_github_tools,
    build_agent_tools,
)


class SearchQuery(BaseModel):
    """Search the indexed documents for a query."""

    query: str


class IngestGithubRepo(BaseModel):
    """Ingest a GitHub repository for code/documentation search."""
    repo_url: str = Field(..., description="The URL of the GitHub repository to ingest.")


async def process_documentation(
    state: GraphState, *, config: RunnableConfig
) -> Dict:
    """Process documentation from the user input and state.
    
    This function extracts or formats documentation either from the user's messages
    or from documents already in the state.
    
    Args:
        state (GraphState): The current state containing messages and/or documents.
        config (RunnableConfig): Configuration for the processing.
        
    Returns:
        Dict: A dictionary with updated documents.
    """
    messages = state.messages
    
    # If we already have documentation, use it
    if state.documentation:
        return {"documents": state.documentation}

    # If a MongoDB document ID/reference is present in the state, fetch from MongoDB
    doc_id = getattr(state, 'mongodb_doc_id', None)
    if doc_id:
        from code_assistant.utils import get_document_from_mongodb
        mongo_doc = get_document_from_mongodb(doc_id)
        if mongo_doc:
            return {"documents": [mongo_doc]}

    # Extract documentation from the latest message
    user_input = get_message_text(messages[-1])
    # For now, we'll just create a document from the user input
    from langchain_core.documents import Document
    doc = Document(page_content=user_input)
    return {"documents": [doc]}


async def generate_code(
    state: GraphState, *, config: RunnableConfig
) -> Dict:
    """Generate code based on the user's question and documentation.
    
    This function uses an LLM to generate code that addresses the user's requirements
    while utilizing the provided documentation.
    
    Args:
        state (GraphState): The current state with user messages and documentation.
        config (RunnableConfig): Configuration for code generation.
        
    Returns:
        Dict: A dictionary with the generated code solution and updated messages.
    """
    print("---GENERATING CODE SOLUTION---")
    
    # Get configuration
    configuration = Configuration.from_runnable_config(config)
    
    # Get state components
    messages = state.messages
    iterations = state.iterations
    error = state.error
    documents = state.documents
    
    # Format documentation
    formatted_docs = format_docs(documents)
    
    # Detect if the user wants to brainstorm or have a natural language session before starting a task
    user_text = get_message_text(messages[-1]).strip().lower()
    brainstorm_keywords = ["brainstorm", "chat", "discuss", "idea", "think", "explore"]
    task_keywords = ["ingest", "generate", "create", "build", "code", "github.com"]
    if any(word in user_text for word in brainstorm_keywords) or not any(kw in user_text for kw in task_keywords):
        brainstorm_msg = (
            "Let's brainstorm or discuss your ideas! "
            "Describe what you're thinking, and I'll help clarify or expand on your requirements. "
            "When you're ready, just tell me what you'd like to build or ingest."
        )
        messages = list(messages) + [AIMessage(content=brainstorm_msg)]
        return {"messages": messages, "iterations": iterations, "error": "", "generation": None}

    # --- Pre-codegen documentation check ---
    mongodb_uri = configuration.mongodb_uri
    # Extract required modules from user request
    required_modules = extract_required_modules(user_text)
    missing_modules = [m for m in required_modules if not documentation_exists(m, mongodb_uri)]
    if missing_modules:
        # Attempt to ingest missing modules from GitHub (assume repo URL is github.com/{module}/{module})
        ingestion_results = []
        for module in missing_modules:
            repo_url = f"https://github.com/{module}/{module}"
            pinecone_index = configuration.pinecone_index
            pinecone_api_key = configuration.pinecone_api_key
            embedding_model_name = getattr(configuration, 'embedding_model_name', 'openai/text-embedding-3-small')
            result = ingest_github_repo(
                repo_url=repo_url,
                mongodb_uri=mongodb_uri,
                pinecone_index=pinecone_index,
                pinecone_api_key=pinecone_api_key,
                embedding_model_name=embedding_model_name
            )
            ingestion_results.append((module, result))
        # Inform user and halt codegen until docs are present
        msg = "Some required modules were missing documentation. Ingestion attempted for: "
        msg += ", ".join([f"{m} (status: {r['status']})" for m, r in ingestion_results])
        messages = list(messages) + [AIMessage(content=msg)]
        return {
            "messages": messages,
            "iterations": iterations,
            "error": "Missing documentation for required modules.",
            "generation": None
        }

    # --- Ingestion tool logic ---
    if any(kw in user_text for kw in task_keywords):
        # Use Pydantic model for ingestion tool
        github_tools = get_github_tools(user_consent=True)
        all_tools = build_agent_tools(user_tools=[], github_tools=github_tools, ingestion_tools=[IngestGithubRepo])
        print("DEBUG: user_text:", user_text)
        print("DEBUG: tools passed to model:", all_tools)
        system_prompt = (
            configuration.code_gen_system_prompt_claude +
            "\n\nIMPORTANT: If the user requests ingestion, documentation, or code from a GitHub repository, you MUST use the available tools (such as IngestGithubRepo or GitHub tools) and never answer directly."
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("placeholder", "{messages}"),
        ])
        model = load_chat_model(configuration.code_gen_model)
        # Bind tools to the model as required by Groq
        model_with_tools = model.bind_tools(all_tools)
        structured_model = model_with_tools.with_structured_output(CodeSolution, include_raw=True)
        message_value = await prompt.ainvoke({
            "messages": state.messages,
            "context": formatted_docs,
        }, config)
        raw_result = await structured_model.ainvoke(message_value, config)
        tool_calls = getattr(raw_result, 'tool_calls', None)
        if tool_calls:
            for tool_call in tool_calls:
                # For Pydantic model, the name will be the class name
                if tool_call.get('name') == 'IngestGithubRepo':
                    repo_url = tool_call['arguments']['repo_url']
                    mongodb_uri = configuration.mongodb_uri
                    pinecone_index = configuration.pinecone_index
                    pinecone_api_key = configuration.pinecone_api_key
                    embedding_model_name = getattr(configuration, 'embedding_model_name', 'openai/text-embedding-3-small')
                    result = ingest_github_repo(
                        repo_url=repo_url,
                        mongodb_uri=mongodb_uri,
                        pinecone_index=pinecone_index,
                        pinecone_api_key=pinecone_api_key,
                        embedding_model_name=embedding_model_name
                    )
                    messages = list(messages) + [AIMessage(content=f"GitHub repo ingestion result: {result}")]
                    return {
                        "generation": None,
                        "messages": messages,
                        "iterations": iterations,
                        "error": "" if result.get("status") == "success" else result.get("error", "ingestion error")
                    }
        messages = list(messages) + [AIMessage(content="Ingestion request detected, but no tool calls were made by the model.")]
        return {"messages": messages, "iterations": iterations, "error": "", "generation": None}
    
    # Fallback return if no code path matches
    return {"messages": messages, "iterations": iterations, "error": "No valid codegen path taken.", "generation": None}


# Build the graph
builder = StateGraph(GraphState, input=InputState, config_schema=Configuration)

# Add nodes
builder.add_node("process_documentation", process_documentation)
builder.add_node("generate", generate_code)
# Add other nodes as needed (e.g., check_code, reflect) if they exist

# Add edges
builder.add_edge(START, "process_documentation")
builder.add_edge("process_documentation", "generate")
# Add other edges as needed

# Compile the graph
graph = builder.compile()
graph.name = "CodeAssistant"