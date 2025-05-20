#!/usr/bin/env python3
"""Test script for GitHub repository ingestion.

This script tests the GitHub repository ingestion functionality directly
by calling the ingest_github_repo function.
"""

import asyncio
import os
from code_assistant.utils import ingest_github_repo

async def test_github_ingestion():
    """Test the GitHub repository ingestion function directly."""
    # Set up test parameters
    repo_url = "https://github.com/langchain-ai/langchain"
    # Use environment variables or defaults for configuration
    mongodb_uri = os.environ.get("MONGODB_URI", "mongodb://localhost:27017/test")
    pinecone_index = os.environ.get("PINECONE_INDEX", "test-index")
    pinecone_api_key = os.environ.get("PINECONE_API_KEY", "test-api-key")
    embedding_model_name = "all-MiniLM-L6-v2"  # Use a small model for testing
    
    print(f"Testing GitHub ingestion with repo: {repo_url}")
    print(f"MongoDB URI: {mongodb_uri}")
    print(f"Pinecone Index: {pinecone_index}")
    print(f"Embedding Model: {embedding_model_name}")
    
    # Call the function directly
    try:
        result = await ingest_github_repo(
            repo_url=repo_url,
            mongodb_uri=mongodb_uri,
            pinecone_index=pinecone_index,
            pinecone_api_key=pinecone_api_key,
            embedding_model_name=embedding_model_name
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