"""LangGraph implementation of the code assistant.

This module contains the state definitions and graph configuration
for the code assistant using LangGraph.
"""

import re
import json
from typing import Dict, List, Optional, Any, cast, Tuple, Sequence, Type, Callable
import traceback

from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field
from langgraph.graph import END, StateGraph, START
from langchain_core.tools import Tool  # Use simple Tool instead of BaseTool

from code_assistant.configuration import Configuration
from code_assistant.state import CodeSolution, GraphState, InputState
from code_assistant.tools import (
    extract_required_modules,
    documentation_exists,
    check_and_ingest_missing_modules,
    extract_tool_calls,
    format_tool_call,
    get_github_tools,
    build_agent_tools
)
from code_assistant.utils import (
    get_message_text,
    format_docs,
    load_chat_model,
    check_code_execution,
    ingest_github_repo
)

from langchain.schema import Document


class SearchQuery(BaseModel):
    """Search the indexed documents for a query."""
    query: str = Field(..., description="The query to search for in the indexed documents.")


# Schema for GitHub repo ingestion
class GithubRepoSchema(BaseModel):
    """Schema for GitHub repository ingestion."""
    repo_url: str = Field(..., description="The URL of the GitHub repository to ingest.")


# Define simplified helper functions for the tool
def _github_repo_ingest_run(repo_url: str) -> str:
    """Run GitHub repository ingestion synchronously (not implemented)."""
    raise NotImplementedError("This tool only supports async execution")

async def _github_repo_ingest_async_run(repo_url: str) -> str:
    """Run GitHub repository ingestion asynchronously."""
    if not repo_url:
        raise ValueError("repo_url is required")
    return f"Started ingestion of GitHub repository: {repo_url}"

# Create the tool using the function-based approach
IngestGithubRepo = Tool(
    name="IngestGithubRepo",
    description="Ingest a GitHub repository for code/documentation search",
    func=_github_repo_ingest_run,
    coroutine=_github_repo_ingest_async_run,
    args_schema=GithubRepoSchema
)


async def process_documentation(
    state: GraphState, *, config: RunnableConfig
) -> Dict:
    """Process user input for documentation or ingestion requests."""
    messages = state.messages
    user_input = get_message_text(messages[-1]) if messages else ""
    
    if not user_input:
        return {"documents": [], "messages": messages, "error": "No user input"}
    
    # For now, create a document from user input
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
    configuration = Configuration.from_runnable_config(config)
    missing_modules = []  # Initialize with empty list to avoid "possibly unbound" error

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
            # Stream the response by using the streaming parameter
            result = await model_with_tools.ainvoke(messages, config=config, streaming=True)
            tool_calls = extract_tool_calls(result)
            
            for tool_call in tool_calls:
                formatted_call = format_tool_call(tool_call)
                if formatted_call['name'] == 'IngestGithubRepo':
                    # Get repo_url from arguments, handling both possible formats
                    if isinstance(formatted_call.get('arguments'), dict):
                        repo_url = formatted_call['arguments'].get('repo_url')
                    else:
                        # For backward compatibility or different format
                        repo_url = formatted_call.get('repo_url')
                    
                    if not repo_url:
                        raise ValueError("Missing repo_url in tool call arguments")
                        
                    result = await ingest_github_repo(
                        repo_url=repo_url,
                        mongodb_uri=configuration.mongodb_uri,
                        pinecone_index=configuration.pinecone_index,
                        pinecone_api_key=configuration.pinecone_api_key,
                        embedding_model_name=configuration.embedding_model_name
                    )
                    # Handle the result which may be a boolean or a dictionary
                    status_message = "successful" if result == True or (isinstance(result, dict) and result.get('status') == 'success') else "failed"
                    
                    # Add more detailed error information
                    detailed_message = ""
                    if status_message == "failed":
                        # Check for common configuration issues
                        if not configuration.mongodb_uri:
                            detailed_message += "MongoDB URI is not configured. "
                        if not configuration.pinecone_index:
                            detailed_message += "Pinecone index is not configured. "
                        if not configuration.pinecone_api_key:
                            detailed_message += "Pinecone API key is not configured. "
                        
                        # Add detailed_message only if there's content
                        if detailed_message:
                            status_message += f" (Reason: {detailed_message.strip()})"
                        
                        # Log the error details
                        print(f"GitHub ingestion failed: {detailed_message}")
                    
                    messages.append(AIMessage(content=f"GitHub repo ingestion complete: {status_message}"))
                    return {
                        "messages": messages,
                        "iterations": iterations,
                        "error": "",
                        "generation": None
                    }
        except Exception as e:
            messages.append(AIMessage(content=f"Error during ingestion: {str(e)}\n{traceback.format_exc()}"))
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
    
    # Save this information in memory for future reference
    state.add_to_memory(
        key=f"modules_check_{iterations}",
        value={
            "required_modules": required_modules,
            "missing_modules": missing_modules,
            "timestamp": __import__('datetime').datetime.now().isoformat()
        },
        metadata={"type": "module_check"}
    )
    
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
        
        # Use astream_events for more granular streaming with streaming parameter
        event_generator = model_with_tools.astream_events(
            message_value,
            config=config,
            streaming=True,
            version="v1"
        )
        
        # Process events
        result = None
        async for event in event_generator:
            if event["event"] == "on_chat_model_stream":
                # Process streaming tokens if needed
                pass
            elif event["event"] == "on_chat_model_end":
                # Safe access to the messages - different models might return different structures
                if "data" in event and isinstance(event["data"], dict):
                    if "messages" in event["data"] and len(event["data"]["messages"]) > 0:
                        result = event["data"]["messages"][0]
                    elif "message" in event["data"]:
                        result = event["data"]["message"]
        
        # If we didn't get a result from streaming, fall back to normal invocation
        if result is None:
            result = await model_with_tools.ainvoke(message_value, config=config)
        
        # Verify the generated code if it exists
        # This is a basic verification - expanded quality gates would be implemented here
        if result and hasattr(result, "content"):
            code_content = str(result.content)  # Convert content to string to ensure it's processable
            
            # Extract code blocks
            code_blocks = re.findall(r"```(?:python)?\s*(.*?)```", code_content, re.DOTALL)
            
            if code_blocks:
                # Check imports and code execution
                execution_result = await check_code_execution(code_blocks[0])
                
                # Store verification results in memory
                state.add_to_memory(
                    key=f"code_verification_{iterations}",
                    value={
                        "execution_result": execution_result,
                        "timestamp": __import__('datetime').datetime.now().isoformat()
                    },
                    metadata={"type": "code_verification"}
                )
                
                # If verification fails, append message about the issues
                if not execution_result.get("success", False):
                    error_msg = "Code verification failed:\n"
                    error_msg += f"- {execution_result.get('description', 'Unknown error')}\n"
                    
                    messages.append(AIMessage(content=error_msg))
                    # We don't return here to allow the code to be returned with the errors
        
        # Extract tool calls and handle them
        tool_calls = extract_tool_calls(result)
        for tool_call in tool_calls:
            formatted_call = format_tool_call(tool_call)
            if formatted_call['name'] == 'IngestGithubRepo':
                # Handle additional ingestion requests
                # Get repo_url from arguments, handling both possible formats
                if isinstance(formatted_call.get('arguments'), dict):
                    repo_url = formatted_call['arguments'].get('repo_url')
                else:
                    # For backward compatibility or different format
                    repo_url = formatted_call.get('repo_url')
                
                if not repo_url:
                    raise ValueError("Missing repo_url in tool call arguments")
                
                ingestion_result = await ingest_github_repo(
                    repo_url=repo_url,
                    mongodb_uri=configuration.mongodb_uri,
                    pinecone_index=configuration.pinecone_index,
                    pinecone_api_key=configuration.pinecone_api_key,
                    embedding_model_name=configuration.embedding_model_name
                )
                # Handle the result which may be a boolean or a dictionary
                status_message = "successful" if ingestion_result == True or (isinstance(ingestion_result, dict) and ingestion_result.get('status') == 'success') else "failed"
                messages.append(AIMessage(content=f"Additional repo ingestion result: {status_message}"))
        
        # Manage conversation history to prevent context window issues
        state.trim_messages(max_count=15)  # Keep only the last 15 messages
        state.update_conversation_summary()  # Update the conversation summary
        
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