#!/usr/bin/env python3
"""Test script for GitHub repository ingestion.

This script tests the GitHub repository ingestion functionality
with the modified IngestGithubRepo tool.
"""

import asyncio
import json
from typing import Dict, List, Any

from code_assistant.tools import format_tool_call

async def test_tool_call_formatting():
    """Test the tool call formatting function."""
    # Test OpenAI-style tool calls
    openai_tool_call = {
        "id": "call_abc123",
        "type": "function",
        "function": {
            "name": "IngestGithubRepo",
            "arguments": '{"repo_url": "https://github.com/langchain-ai/langchain"}'
        }
    }
    
    # Test LangChain-style tool calls
    langchain_tool_call = {
        "name": "IngestGithubRepo",
        "args": {
            "repo_url": "https://github.com/langchain-ai/langchain"
        }
    }
    
    # Test broken tool calls
    broken_tool_call = {
        "function": {
            "name": "IngestGithubRepo",
            "arguments": '{"repo_url": '  # Incomplete JSON
        }
    }
    
    # Format and print results
    print("OpenAI-style tool call formatted:")
    formatted_openai = format_tool_call(openai_tool_call)
    print(json.dumps(formatted_openai, indent=2))
    
    print("\nLangChain-style tool call formatted:")
    formatted_langchain = format_tool_call(langchain_tool_call)
    print(json.dumps(formatted_langchain, indent=2))
    
    print("\nBroken tool call formatted:")
    formatted_broken = format_tool_call(broken_tool_call)
    print(json.dumps(formatted_broken, indent=2))

async def main():
    """Run the tests."""
    print("Testing GitHub repository ingestion tools")
    print("========================================")
    
    await test_tool_call_formatting()
    
    print("\nTests completed.")

if __name__ == "__main__":
    asyncio.run(main()) 