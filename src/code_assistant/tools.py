"""Tools and tool-related functions for the code assistant.

This module contains functions for creating, formatting, and using tools
with large language models, including GitHub repository tools, documentation tools,
and other agent-based tool implementations.

Functions:
    extract_tool_calls: Extract tool calls from an AI message.
    format_tool_call: Format a tool call for execution.
    get_github_tools: Get GitHub repository tools based on user consent.
    build_agent_tools: Build a complete set of agent tools.
    extract_required_modules: Extract required modules from text.
    documentation_exists: Check if documentation exists for a module.
    check_and_ingest_missing_modules: Check and ingest missing module documentation.
"""

import os
import re
import json
from typing import Dict, List, Optional, Any, Union, Type, Sequence
import traceback
from pydantic import BaseModel
from langchain_core.tools import BaseTool

# Define ToolException locally to avoid import issues
class ToolException(Exception):
    """Exception raised by tool execution."""
    def __init__(self, message: str, recoverable: bool = False):
        super().__init__(message)
        self.recoverable = recoverable

from langchain_core.messages import BaseMessage


def extract_required_modules(text: str) -> list[str]:
    """Extract required modules from text.
    
    This function identifies import statements in code or text describing
    a coding task and extracts the module names for documentation retrieval.
    
    Args:
        text (str): The text or code to analyze.
        
    Returns:
        list[str]: A list of unique module names.
    """
    # First, normalize indentation by removing leading spaces from each line
    normalized_text = "\n".join(line.lstrip() for line in text.split("\n"))
    
    # Patterns to match various import formats
    import_patterns = [
        r'(?:^|\n)\s*import\s+([a-zA-Z0-9_.,\s]+)(?:$|\n)',                      # import module
        r'(?:^|\n)\s*from\s+([a-zA-Z0-9_.]+)\s+import\s+[a-zA-Z0-9_.,\s*]+(?:$|\n)',  # from module import ...
        r'(?:^|\n)\s*(?:using|require|library|include)\s+[\'"]?([a-zA-Z0-9_.]+)[\'"]?(?:$|\n)',  # other languages
        r'(?i)(?:^|\n)\s*(?:pip install|conda install|npm install|yarn add|gem install|install package)\s+[\'"]?([a-zA-Z0-9_.@/-]+)[\'"]?(?:$|\n)',  # installation commands
    ]
    
    # Extract all potential modules
    modules = []
    for pattern in import_patterns:
        matches = re.findall(pattern, normalized_text)
        for match in matches:
            if ',' in match:  # Handle multiple imports (import numpy, pandas)
                modules.extend([m.strip() for m in match.split(',')])
            else:
                modules.append(match.strip())
    
    # Clean up module names
    cleaned_modules = []
    for module in modules:
        # Handle "import pandas as pd" format
        if " as " in module:
            module = module.split(" as ")[0].strip()
            
        # Extract base module (e.g., 'pandas' from 'pandas.DataFrame')
        base_module = module.split('.')[0].strip()
        
        # Add the base module if it's not empty
        if base_module:
            cleaned_modules.append(base_module)
    
    # Return unique modules
    return list(set(cleaned_modules))


def documentation_exists(module: str, mongodb_uri: str) -> bool:
    """Check if documentation exists for a module.
    
    Args:
        module (str): The module name to check.
        mongodb_uri (str): MongoDB connection URI.
        
    Returns:
        bool: True if documentation exists, False otherwise.
    """
    try:
        from pymongo import MongoClient
        
        # Connect to MongoDB
        client = MongoClient(mongodb_uri)
        db = client.get_default_database()
        
        # Check if the module exists in the documentation collection
        count = db.documents.count_documents({"metadata.module": module})
        return count > 0
    except Exception as e:
        print(f"Error checking documentation existence: {e}")
        return False


async def check_and_ingest_missing_modules(required_modules: list[str], mongodb_uri: str, pinecone_config: dict) -> dict:
    """Check and ingest missing module documentation.
    
    This function checks if documentation exists for required modules and
    attempts to ingest documentation for missing modules.
    
    Args:
        required_modules (list[str]): List of required module names.
        mongodb_uri (str): MongoDB connection URI.
        pinecone_config (dict): Configuration for Pinecone.
            
    Returns:
        dict: Dictionary mapping module names to ingestion results.
    """
    from code_assistant.utils import ingest_github_repo
    
    results = {}
    
    # Map of known modules to their documentation repos
    module_repo_map = {
        'pandas': 'pandas-dev/pandas',
        'numpy': 'numpy/numpy',
        'matplotlib': 'matplotlib/matplotlib',
        'sklearn': 'scikit-learn/scikit-learn',
        'tensorflow': 'tensorflow/tensorflow',
        'torch': 'pytorch/pytorch',
        'transformers': 'huggingface/transformers',
        'langchain': 'langchain-ai/langchain',
    }
    
    for module in required_modules:
        try:
            # Skip if documentation already exists
            if documentation_exists(module, mongodb_uri):
                results[module] = {'status': 'exists', 'message': 'Documentation already exists'}
                continue
                
            # Check if we know the repo for this module
            repo = module_repo_map.get(module)
            if not repo:
                results[module] = {'status': 'error', 'message': 'Unknown module repository'}
                continue
                
            # Attempt to ingest documentation            
            success = await ingest_github_repo(
                repo_url=f"https://github.com/{repo}",
                mongodb_uri=mongodb_uri,
                pinecone_index=pinecone_config['index'],
                pinecone_api_key=pinecone_config['api_key'],
                embedding_model_name=pinecone_config['embedding_model']
            )
            
            if success:
                results[module] = {'status': 'success', 'message': f'Successfully ingested documentation from {repo}'}
            else:
                results[module] = {'status': 'error', 'message': f'Failed to ingest documentation from {repo}'}
                
        except Exception as e:
            results[module] = {'status': 'error', 'message': str(e)}
            
    return results


def extract_tool_calls(response: Any) -> List[Dict[str, Any]]:
    """Extract tool calls from the model response.
    
    Args:
        response (Any): The model response.
        
    Returns:
        List[Dict[str, Any]]: List of tool calls.
    """
    if hasattr(response, "tool_calls") and response.tool_calls:
        return response.tool_calls
    elif hasattr(response, "additional_kwargs") and "tool_calls" in response.additional_kwargs:
        return response.additional_kwargs["tool_calls"]
    return []


def format_tool_call(tool_call: Dict[str, Any]) -> Dict[str, Any]:
    """Format tool call for execution.
    
    Args:
        tool_call (Dict[str, Any]): The tool call to format.
        
    Returns:
        Dict[str, Any]: Formatted tool call with name and arguments.
    """
    try:
        if "function" in tool_call:
            # Handle OpenAI-style tool calls
            function = tool_call["function"]
            name = function.get("name", "")
            
            # Handle arguments with proper error handling
            arguments = {}
            if "arguments" in function:
                try:
                    arg_str = function["arguments"]
                    if arg_str and isinstance(arg_str, str):
                        arguments = json.loads(arg_str)
                    elif isinstance(arg_str, dict):
                        # Already a dictionary
                        arguments = arg_str
                except json.JSONDecodeError:
                    print(f"Warning: Could not decode tool call arguments: {function['arguments']}")
                    arguments = {}
                    
            return {
                "name": name,
                "arguments": arguments,
            }
        elif "name" in tool_call and "args" in tool_call:
            # Handle LangChain-style tool calls
            return {
                "name": tool_call["name"],
                "arguments": tool_call["args"],
            }
        elif "name" in tool_call and "arguments" in tool_call:
            # Handle tools that already have the right format
            return {
                "name": tool_call["name"],
                "arguments": tool_call["arguments"],
            }
            
        # Return a standardized structure for other formats
        return {
            "name": tool_call.get("name", ""),
            "arguments": tool_call.get("args", tool_call.get("arguments", {}))
        }
    except Exception as e:
        print(f"Error formatting tool call: {str(e)}")
        print(f"Raw tool call: {tool_call}")
        # Return a minimal valid structure to avoid downstream errors
        return {
            "name": "",
            "arguments": {}
        }


def get_github_tools(user_consent: bool = False):
    """Get GitHub repository tools based on user consent.
    
    This function creates and returns GitHub repository tools if the user
    has provided consent to use them.
    
    Args:
        user_consent (bool): Whether the user has consented to use GitHub tools.
        
    Returns:
        list: A list of GitHub tools if user_consent is True, otherwise an empty list.
    """
    if not user_consent:
        return []
    
    try:
        # Use direct import with proper error handling
        import importlib
        github_repo_module = importlib.import_module('langchain_community.tools.github.repository')
        github_tool_module = importlib.import_module('langchain_community.tools.github.tool')
        
        GitHubRepositoryAPI = github_repo_module.GitHubRepositoryAPI
        GitHubAction = github_tool_module.GitHubAction
        
        # Get GitHub token from environment variable
        github_token = os.getenv("GITHUB_TOKEN")
        if not github_token:
            print("Warning: GITHUB_TOKEN not found in environment variables. GitHub tools will not be available.")
            return []
            
        # Create GitHub repository API
        repo_api = GitHubRepositoryAPI(
            github_api_token=github_token,
            repo_owner="owner",  # Will be overridden by run-time parameters
            repo_name="repo",    # Will be overridden by run-time parameters
        )
        
        # Create GitHub tools
        return [
            GitHubAction(
                name="search_repo_content",
                description="Search through a GitHub repository for files matching the query",
                api_wrapper=repo_api,
                mode="search_code",
            ),
            GitHubAction(
                name="get_file_contents",
                description="Get the contents of a file from a GitHub repository",
                api_wrapper=repo_api,
                mode="get_file_contents",
            ),
        ]
        
    except ImportError:
        print("Warning: langchain_community GitHub tools not available. Install with: pip install langchain-community")
        return []
    except Exception as e:
        print(f"Error creating GitHub tools: {e}")
        return []


def build_agent_tools(user_tools: list, github_tools: list | None = None, ingestion_tools: list | None = None) -> list:
    """Build a complete set of agent tools.
    
    This function combines user-provided tools, GitHub tools, and ingestion tools
    into a unified list for use with an agent.
    
    Args:
        user_tools (list): List of user-provided tools.
        github_tools (list, optional): List of GitHub tools. Defaults to None.
        ingestion_tools (list, optional): List of ingestion tools. Defaults to None.
        
    Returns:
        list: A list of all agent tools.
    """
    all_tools = list(user_tools)  # Make a copy of user tools
    
    # Add GitHub tools if available
    if github_tools:
        all_tools.extend(github_tools)
        
    # Handle ingestion tools
    if ingestion_tools:
        # Initialize each tool class if they're class types
        for tool_cls in ingestion_tools:
            if isinstance(tool_cls, type) and issubclass(tool_cls, BaseTool):
                try:
                    tool_instance = tool_cls()
                    all_tools.append(tool_instance)
                except Exception as e:
                    print(f"Error initializing tool {tool_cls.__name__}: {e}")
            else:
                # If it's already an instance, just add it
                all_tools.append(tool_cls)
    
    return all_tools


async def handle_tool_error(tool_name: str, error: Exception, retry_handler=None) -> Dict[str, Any]:
    """Handle errors from tool execution in a standardized way.
    
    Args:
        tool_name (str): The name of the tool that encountered an error
        error (Exception): The exception that occurred
        retry_handler (callable, optional): Function to handle retries
        
    Returns:
        Dict[str, Any]: Standardized error response
    """
    error_trace = traceback.format_exc()
    error_message = str(error)
    
    # Log the error for debugging
    print(f"Error in tool {tool_name}: {error_message}")
    print(f"Traceback: {error_trace}")
    
    # If we have a retry handler, try to use it
    if retry_handler and callable(retry_handler):
        try:
            return await retry_handler(tool_name, error)
        except Exception as retry_error:
            # If retry also fails, continue with normal error handling
            print(f"Retry handling failed: {str(retry_error)}")
    
    # Construct a standardized error response
    error_response = {
        "status": "error",
        "tool": tool_name,
        "error_type": type(error).__name__,
        "message": error_message,
        "recoverable": isinstance(error, ToolException) and getattr(error, "recoverable", False)
    }
    
    return error_response


async def safe_tool_call(tool_func, tool_name: str, **kwargs) -> Dict[str, Any]:
    """Safely execute a tool call with standardized error handling.
    
    Args:
        tool_func (callable): The tool function to call
        tool_name (str): The name of the tool
        **kwargs: Arguments to pass to the tool function
        
    Returns:
        Dict[str, Any]: Tool result or error response
    """
    try:
        result = await tool_func(**kwargs)
        return {
            "status": "success",
            "tool": tool_name,
            "result": result
        }
    except Exception as e:
        return await handle_tool_error(tool_name, e)


class ToolCallResult:
    """Class to represent the result of a tool call with improved error handling."""
    
    def __init__(self, success: bool, tool_name: str, result: Any = None, error: Optional[Exception] = None):
        """Initialize the tool call result.
        
        Args:
            success (bool): Whether the tool call was successful
            tool_name (str): The name of the tool
            result (Any, optional): The result if successful
            error (Optional[Exception], optional): The error if not successful
        """
        self.success = success
        self.tool_name = tool_name
        self.result = result
        self.error = error
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to a dictionary representation.
        
        Returns:
            Dict[str, Any]: Dictionary representation of the result
        """
        if self.success:
            return {
                "status": "success",
                "tool": self.tool_name,
                "result": self.result
            }
        else:
            return {
                "status": "error",
                "tool": self.tool_name,
                "error_type": type(self.error).__name__ if self.error else "UnknownError",
                "message": str(self.error) if self.error else "Unknown error",
                "recoverable": isinstance(self.error, ToolException) and getattr(self.error, "recoverable", False)
            }
            

class RetryableToolException(ToolException):
    """A tool exception that can be retried.
    
    This exception indicates that the tool execution failed but the operation
    can be retried, potentially with different parameters or after some time.
    """
    
    def __init__(self, message: str, retry_after_seconds: Optional[int] = None):
        """Initialize the retryable tool exception.
        
        Args:
            message (str): The error message
            retry_after_seconds (Optional[int], optional): Suggested wait time before retrying
        """
        super().__init__(message)
        self.recoverable = True
        self.retry_after_seconds = retry_after_seconds
