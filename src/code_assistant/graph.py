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
from pydantic import BaseModel
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
)


class SearchQuery(BaseModel):
    """Search the indexed documents for a query."""

    query: str


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

    # --- Always ingest langchain-sandbox and langgraph-codeact ---
    mongodb_uri = configuration.mongodb_uri
    pinecone_index = configuration.pinecone_index
    pinecone_api_key = configuration.pinecone_api_key
    embedding_model_name = getattr(configuration, 'embedding_model_name', 'openai/text-embedding-3-small')
    sandbox_result = ingest_github_repo(
        repo_url="https://github.com/langchain-ai/langchain-sandbox",
        mongodb_uri=mongodb_uri,
        pinecone_index=pinecone_index,
        pinecone_api_key=pinecone_api_key,
        embedding_model_name=embedding_model_name
    )
    codeact_result = ingest_github_repo(
        repo_url="https://github.com/langchain-ai/langgraph-codeact",
        mongodb_uri=mongodb_uri,
        pinecone_index=pinecone_index,
        pinecone_api_key=pinecone_api_key,
        embedding_model_name=embedding_model_name
    )
    if sandbox_result.get("status") != "success" or codeact_result.get("status") != "success":
        messages = list(messages) + [AIMessage(content=f"Failed to ingest required sandbox/codeact modules: sandbox={sandbox_result}, codeact={codeact_result}")]
        return {"messages": messages, "iterations": iterations, "error": "Sandbox/codeact ingestion failed.", "generation": None}

    # --- Pre-codegen documentation check ---
    required_modules = extract_required_modules(user_text)
        return {"messages": messages, "iterations": iterations, "error": "Missing documentation for required modules.", "generation": None}

    # --- Ingestion tool logic ---
    # Only run if a task keyword is present
    if any(kw in user_text for kw in task_keywords):
        # Define ingestion tool schema
        ingestion_tool = {
            "type": "function",
            "function": {
                "name": "ingest_github_repo",
                "description": "Ingest a GitHub repository and index its documentation for semantic search.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "repo_url": {
                            "type": "string",
                            "description": "The URL of the GitHub repository to ingest."
                        }
                    },
                    "required": ["repo_url"]
                }
            }
        }
        # Model/tool selection logic (Groq, etc.)
        model_name = configuration.code_gen_model
        groq_tool_models = [
            "meta-llama/llama-4-scout-17b-16e-instruct",
            "meta-llama/llama-4-maverick-17b-128e-instruct",
            "qwen-qwq-32b",
            "deepseek-r1-distill-qwen-32b",
            "deepseek-r1-distill-llama-70b",
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "gemma2-9b-it"
        ]
        use_tools = False
        if (model_name.startswith("groq/") or model_name.startswith("groq-")) and any(model_name.endswith(m) for m in groq_tool_models):
            use_tools = True
        # Prepare prompt and model
        prompt = ChatPromptTemplate.from_messages([
            ("system", configuration.code_gen_system_prompt_claude),
            ("placeholder", "{messages}"),
        ])
        model = load_chat_model(configuration.code_gen_model)
        structured_model = model.with_structured_output(CodeSolution, include_raw=True)
        message_value = await prompt.ainvoke({
            "messages": state.messages,
            "context": formatted_docs,
        }, config)
        # Call LLM with tool support
        if use_tools:
            raw_result = await structured_model.ainvoke(message_value, config, tools=[ingestion_tool])
            tool_calls = getattr(raw_result, 'tool_calls', None)
            if tool_calls:
                for tool_call in tool_calls:
                    if tool_call.get('name') == 'ingest_github_repo':
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
        # If not using tools, just return a fallback message
        messages = list(messages) + [AIMessage(content="Ingestion request detected, but tool use is not enabled for this model.")]
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