"""Main entrypoint for the code assistant graph."""

from typing import Dict, List, Optional, Sequence, cast
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field
from langgraph.graph import END, StateGraph, START

from code_assistant.configuration import Configuration
from code_assistant.state import CodeSolution, GraphState, InputState
from code_assistant.tools import (
    extract_tool_calls,
    format_tool_call,
    extract_required_modules,
    documentation_exists,
    get_github_tools,
    build_agent_tools,
    check_and_ingest_missing_modules
)
from code_assistant.utils import (
    check_code_execution,
    check_imports,
    format_docs, 
    get_message_text,
    load_chat_model,
    ingest_github_repo
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
    """Process user input for documentation or ingestion requests."""
    messages = state.messages
    user_input = get_message_text(messages[-1]) if messages else ""
    
    if not user_input:
        return {"documents": [], "messages": messages, "error": "No user input"}
    
    # For now, create a document from user input
    from langchain_core.documents import Document
    doc = Document(page_content=user_input)
    return {"documents": [doc]}


async def generate_code(
    state: GraphState, *, config: RunnableConfig
) -> Dict:
    """Generate code based on user input, with built-in knowledge checking and ingestion."""
    messages = list(state.messages)  # Convert to list for modification
    iterations = state.iterations
    user_text = get_message_text(messages[-1])
    formatted_docs = format_docs(state.get_documents())
    configuration = cast(Configuration, config.get("configurable", {}))

    # Check for ingestion requests
    if "ingest" in user_text.lower() and "github" in user_text.lower():
        github_tools = get_github_tools(user_consent=True)
        all_tools = build_agent_tools(
            user_tools=[], 
            github_tools=github_tools, 
            ingestion_tools=[IngestGithubRepo]
        )
        model = load_chat_model(configuration.code_gen_model)
        model_with_tools = model.bind_tools(all_tools)
        
        try:
            result = model_with_tools.invoke(messages)
            tool_calls = extract_tool_calls(result)
            
            for tool_call in tool_calls:
                formatted_call = format_tool_call(tool_call)
                if formatted_call['name'] == 'IngestGithubRepo':
                    args = formatted_call['arguments']
                    result = await ingest_github_repo(
                        repo_url=args['repo_url'],
                        mongodb_uri=configuration.mongodb_uri,
                        pinecone_index=configuration.pinecone_index,
                        pinecone_api_key=configuration.pinecone_api_key,
                        embedding_model_name=configuration.embedding_model_name
                    )
                    # Handle the result which may be a boolean or a dictionary
                    status_message = "successful" if result == True or (isinstance(result, dict) and result.get('status') == 'success') else "failed"
                    messages.append(AIMessage(content=f"GitHub repo ingestion complete: {status_message}"))
                    return {
                        "messages": messages,
                        "iterations": iterations,
                        "error": "",
                        "generation": None
                    }
        except Exception as e:
            messages.append(AIMessage(content=f"Error during ingestion: {str(e)}"))
            return {
                "messages": messages,
                "iterations": iterations,
                "error": str(e),
                "generation": None
            }
      # For code generation requests, check required knowledge and modules
    required_modules = extract_required_modules(user_text)
    if required_modules:
        # Log the detected modules for debugging
        print(f"Detected modules: {required_modules}")
        
        # Validate documentation exists for all required modules
        missing_modules = [m for m in required_modules if not documentation_exists(m, configuration.mongodb_uri)]
        if missing_modules:
            # Log the missing modules
            print(f"Missing modules that need ingestion: {missing_modules}")
            
            # Attempt to ingest missing modules automatically
            ingestion_results = await check_and_ingest_missing_modules(
                required_modules=missing_modules,
                mongodb_uri=configuration.mongodb_uri,
                pinecone_config={
                    'index': configuration.pinecone_index,
                    'api_key': configuration.pinecone_api_key,
                    'embedding_model': configuration.embedding_model_name
                }
            )
            
            # Check results and proceed or halt
            failed_ingestions = {m: r for m, r in ingestion_results.items() if r.get('status') != 'success'}
            if failed_ingestions:
                msg = f"Unable to proceed with code generation. Failed to ingest documentation for: {', '.join(failed_ingestions.keys())}"
                messages.append(AIMessage(content=msg))
                return {
                    "messages": messages,
                    "iterations": iterations,
                    "error": "Missing required module documentation",
                    "generation": None
                }
            else:
                msg = f"Successfully ingested documentation for: {', '.join(ingestion_results.keys())}"
                messages.append(AIMessage(content=msg))
    
    # Proceed with code generation with validated knowledge
    github_tools = get_github_tools(user_consent=True)
    all_tools = build_agent_tools(
        user_tools=[], 
        github_tools=github_tools,
        ingestion_tools=[IngestGithubRepo]
    )

    model = load_chat_model(configuration.code_gen_model)
    model_with_tools = model.bind_tools(all_tools)

    try:
        system_prompt = (
            configuration.code_gen_system_prompt_claude +
            "\n\nIMPORTANT: If you need documentation or code from a GitHub repository, "
            "use the available tools and never answer without proper documentation."
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("placeholder", "{messages}"),
        ])

        message_value = prompt.format_messages(messages=messages, context=formatted_docs)
        result = model_with_tools.invoke(message_value)
        
        # Extract any tool calls
        tool_calls = extract_tool_calls(result)
        for tool_call in tool_calls:
            formatted_call = format_tool_call(tool_call)
            if formatted_call['name'] == 'IngestGithubRepo':
                # Handle additional ingestion requests
                args = formatted_call['arguments']
                ingestion_result = await ingest_github_repo(
                    repo_url=args['repo_url'],
                    mongodb_uri=configuration.mongodb_uri,
                    pinecone_index=configuration.pinecone_index,
                    pinecone_api_key=configuration.pinecone_api_key,
                    embedding_model_name=configuration.embedding_model_name
                )
                # Handle the result which may be a boolean or a dictionary
                status_message = "successful" if ingestion_result == True or (isinstance(ingestion_result, dict) and ingestion_result.get('status') == 'success') else "failed"
                messages.append(AIMessage(content=f"Additional repo ingestion result: {status_message}"))
        
        # Return the final result
        return {
            "messages": messages,
            "iterations": iterations + 1,
            "error": "",
            "generation": result
        }
        
    except Exception as e:
        messages.append(AIMessage(content=f"Error during code generation: {str(e)}"))
        return {
            "messages": messages,
            "iterations": iterations,
            "error": str(e),
            "generation": None
        }


# Build the graph
builder = StateGraph(GraphState, input=InputState, config_schema=Configuration)

# Add nodes
builder.add_node("process_documentation", process_documentation)
builder.add_node("generate", generate_code)

# Add edges
builder.add_edge(START, "process_documentation")
builder.add_edge("process_documentation", "generate")

# Compile the graph
graph = builder.compile()
graph.name = "CodeAssistant"