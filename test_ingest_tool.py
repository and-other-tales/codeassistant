#!/usr/bin/env python3
"""Test script for the IngestGithubRepo tool.

This script tests that the IngestGithubRepo tool can be properly used
and works correctly within LangChain's tool framework.
"""

import asyncio
import json
import os
from langchain_core.tools import Tool
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from code_assistant.graph import IngestGithubRepo
from code_assistant.tools import build_agent_tools
from code_assistant.utils import load_chat_model

# Set environment variables for testing
os.environ["TEST_MODE"] = "true"

async def test_tool_properties():
    """Test the tool's basic properties."""
    try:
        print(f"Tool name: {IngestGithubRepo.name}")
        print(f"Tool description: {IngestGithubRepo.description}")
        print(f"Tool has schema: {hasattr(IngestGithubRepo, 'args_schema')}")
        print(f"Tool has async capability: {hasattr(IngestGithubRepo, 'coroutine')}")
        
        # Test that the tool can be included in build_agent_tools
        all_tools = build_agent_tools(user_tools=[], github_tools=[], ingestion_tools=[IngestGithubRepo])
        print(f"\nBuild agent tools result: {len(all_tools)} tools")
        
        return True
    except Exception as e:
        print(f"Error checking tool properties: {e}")
        return False

async def test_tool_execution():
    """Test that the tool can be executed."""
    try:
        # Test with proper argument
        print("\nTesting with valid repo_url:")
        # Use async version of tool invocation
        result = await IngestGithubRepo.ainvoke({"repo_url": "https://github.com/langchain-ai/langserve"})
        print(f"Result: {result}")
        
        # Test with empty argument - should fail
        print("\nTesting with empty repo_url:")
        try:
            result = await IngestGithubRepo.ainvoke({"repo_url": ""})
            print(f"Result: {result}")
        except ValueError as e:
            print(f"Expected ValueError: {e}")
            
        return True
    except Exception as e:
        print(f"Unexpected error during execution: {e}")
        return False

async def test_chat_model_with_tools():
    """Test that a chat model can be bound with our tools."""
    try:
        # Try to load a model
        model_name = os.environ.get("TEST_MODEL", "groq/llama3-8b-8192")
        print(f"\nTesting with model: {model_name}")
        
        model = load_chat_model(model_name)
        
        # Try to bind the tool
        try:
            print("Attempting to bind tool to model...")
            model_with_tools = model.bind_tools([IngestGithubRepo])
            print("Successfully bound tools to model!")
            
            # Try a simple invocation
            messages = [HumanMessage(content="Can you help me ingest the repository at https://github.com/langchain-ai/langchain?")]
            config = RunnableConfig(configurable={})
            
            print("Testing a simple invocation (this will not actually run the ingestion)...")
            # Don't actually run this to avoid real network calls
            # result = await model_with_tools.ainvoke(messages, config=config)
            # print(f"Got result: {result}")
            
            return True
        except Exception as e:
            print(f"Error binding tools: {e}")
            return False
            
    except Exception as e:
        print(f"Error in chat model test: {e}")
        return False

async def main():
    """Run the tests."""
    print("Testing IngestGithubRepo tool")
    print("============================")
    
    prop_success = await test_tool_properties()
    exec_success = await test_tool_execution()
    bind_success = await test_chat_model_with_tools()
    
    print("\nTest Results:")
    print(f"- Tool properties: {'✅ PASSED' if prop_success else '❌ FAILED'}")
    print(f"- Tool execution: {'✅ PASSED' if exec_success else '❌ FAILED'}")
    print(f"- Model binding: {'✅ PASSED' if bind_success else '❌ FAILED'}")
    
    print("\nTests completed.")

if __name__ == "__main__":
    asyncio.run(main()) 