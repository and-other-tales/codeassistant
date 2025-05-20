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
from typing import Dict, List, Optional, Any, Union

from langchain_core.messages import BaseMessage
from langchain_core.tools import BaseTool


def extract_required_modules(text: str) -> list[str]:
    """Extract required modules from text.
    
    This function identifies import statements in code or text describing
    a coding task and extracts the module names for documentation retrieval.
    
    Args:
        text (str): The text or code to analyze.
        
    Returns:
        list[str]: A list of unique module names.
    """
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
        matches = re.findall(pattern, text)
        for match in matches:
            if ',' in match:  # Handle multiple imports (import numpy, pandas)
                modules.extend([m.strip() for m in match.split(',')])
            else:
                modules.append(match.strip())
    
    # Clean up module names
    cleaned_modules = []
    for module in modules:
        # Extract base module (e.g., 'pandas' from 'pandas.DataFrame')
        base_module = module.split('.')[0]
        
        # Filter out common built-ins and empty strings
        if base_module and base_module not in ['__future__', 'os', 'sys', 'typing', 're', 'json', 'math', 'time']:
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


def extract_tool_calls(model_output: BaseMessage) -> List[Dict]:
    """Extract tool calls from a message.
    
    Args:
        model_output (BaseMessage): The message containing tool calls.
        
    Returns:
        List[Dict]: A list of tool calls.
    """
    return getattr(model_output, "tool_calls", [])


def format_tool_call(tool_call: Dict) -> Dict:
    """Format a tool call for execution.
    
    Args:
        tool_call (Dict): The raw tool call.
        
    Returns:
        Dict: Formatted tool call.
    """
    return {
        "name": tool_call.get("name", ""),
        "arguments": json.loads(tool_call.get("args", "{}")),
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
