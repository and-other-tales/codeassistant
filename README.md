# Code Assistant

A powerful automated code assistant built with LangChain and LangGraph that can:

1. Generate code based on user requirements and best practices
2. Ingest documentation and code repositories from GitHub
3. Check and validate required module knowledge before code generation
4. Test and verify generated code with quality gates

## Features

- **GitHub Repository Ingestion**: Automatically ingests documentation, examples, and cookbooks from GitHub repositories on user request
- **Intelligent Knowledge Management**: Stores and retrieves documentation in MongoDB and Pinecone to ensure code generation with proper documentation
- **Automatic Module Dependency Analysis**: Detects modules in user queries and automatically ingests missing documentation
- **Multi-Model Support**: Works with OpenAI, Anthropic, and Groq models with proper tool support
- **Code Quality Control**: Verifies imports and tests code execution before providing it to the user
- **Memory Management**: Tracks previously ingested documentation and conversation history to maintain context

## Enhanced Groq Support

This implementation provides full support for Groq models with:

- Tool calling and binding capabilities
- Streaming for more responsive interactions
- Proper error handling
- Support for both synchronous and asynchronous operations

## Technical Details

- Built with LangGraph for state management and workflow orchestration
- Uses LangChain Core for foundational LLM interactions
- Stores documentation in MongoDB and Pinecone for efficient retrieval
- Implements proper memory and state management for maintaining context
- Provides quality gates for generated code

## Environment Variables

The following environment variables should be set:

```
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
GROQ_API_KEY=your_groq_key
MONGODB_URI=your_mongodb_uri
PINECONE_API_KEY=your_pinecone_key
PINECONE_INDEX_NAME=your_pinecone_index
GITHUB_TOKEN=your_github_token
```

## Requirements

See `requirements.txt` for detailed package requirements. Main dependencies:

- langchain
- langchain-core
- langgraph
- langchain-groq (for Groq model support)
- pymongo
- pinecone-client
- pydantic

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python server.py
```

The API will be accessible at `http://localhost:8000` by default.

## Example Workflow

1. User requests code generation for a specific task
2. System detects required modules and checks for documentation
3. If documentation is missing, system automatically ingests relevant repos
4. Code is generated with proper imports and structure
5. System verifies the code's correctness before returning it
6. Code is returned to the user along with any relevant context or explanations