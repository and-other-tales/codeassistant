#!/usr/bin/env python3
"""Test script for GitHub repository ingestion.

This script tests the GitHub repository ingestion functionality directly
by calling the ingest_github_repo function with test mode enabled.
"""

import asyncio
import os
import logging
from code_assistant.utils import ingest_github_repo

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

async def test_github_ingestion():
    """Test the GitHub repository ingestion function directly with test mode."""
    # Set up test parameters
    repo_url = "https://github.com/langchain-ai/langserve"  # Small public repo
    mongodb_uri = "mongodb://localhost:27017/codeassistant"
    pinecone_index = "test-index"
    pinecone_api_key = "test-api-key"
    embedding_model_name = "all-MiniLM-L6-v2"
    
    print(f"Testing GitHub ingestion with repo: {repo_url}")
    print(f"MongoDB URI: {mongodb_uri}")
    print(f"Pinecone Index: {pinecone_index}")
    print(f"Embedding Model: {embedding_model_name}")
    
    # Check for required dependencies
    try:
        import git
        print("✅ GitPython is installed")
    except ImportError:
        print("❌ GitPython is not installed. Installing...")
        import subprocess
        subprocess.check_call(["pip", "install", "gitpython"])
        print("✅ GitPython has been installed")
        
    try:
        from langchain_community.document_loaders import GitLoader
        print("✅ langchain_community is installed")
    except ImportError:
        print("❌ langchain_community is not installed or GitLoader is not available")
        return False
    
    # Call the function directly with test_mode=True
    try:
        result = await ingest_github_repo(
            repo_url=repo_url,
            mongodb_uri=mongodb_uri,
            pinecone_index=pinecone_index,
            pinecone_api_key=pinecone_api_key,
            embedding_model_name=embedding_model_name,
            test_mode=True  # Enable test mode to bypass DB operations
        )
        
        if result:
            print("✅ GitHub ingestion successful!")
        else:
            print("❌ GitHub ingestion failed.")
        
        return result
    except Exception as e:
        print(f"❌ Error during ingestion: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run the test."""
    print("Testing GitHub repository ingestion")
    print("==================================")
    
    success = await test_github_ingestion()
    
    print("\nTest Result:")
    print(f"- GitHub Ingestion: {'✅ PASSED' if success else '❌ FAILED'}")
    
    print("\nTest completed.")

if __name__ == "__main__":
    asyncio.run(main()) 