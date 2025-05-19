"""
Server script for running the Code Assistant API.

This script sets up a FastAPI server with the LangGraph application.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import after loading environment variables
import uvicorn
from langgraph.server import app as langgraph_app

if __name__ == "__main__":
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8080"))
    
    print(f"Starting server on {host}:{port}")
    uvicorn.run(langgraph_app, host=host, port=port)