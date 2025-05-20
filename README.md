# Code Assistant: Knowledge-Driven Agentic Workflow

This repository contains a knowledge-driven agentic workflow for code generation, testing, and refinement, backed by a knowledge graph for contextual awareness.

## Project Overview

Code Assistant is a research project that explores the integration of large language models (LLMs) with knowledge graphs, document retrieval systems, and automated code quality gates to create a highly effective code assistant system. The system aims to improve code generation quality by leveraging contextual knowledge about libraries, dependencies, and code patterns.

## Architecture

## Architecture

The system architecture consists of several core components:

### 1. Agentic Workflow

The Code Assistant employs a LangGraph-based agent workflow with defined states and transitions to create a structured approach to code generation:

- **Knowledge Retrieval**: Dynamically retrieves relevant documentation for the task and dependencies
- **Planning**: Creates a strategy for code implementation
- **Generation**: Produces code based on knowledge and requirements
- **Testing**: Validates the code against quality standards
- **Refinement**: Iteratively improves the code based on test feedback

This state-based architecture allows the system to make informed decisions about when to proceed to the next step or when to iterate within a step.

```
[User Query] → [Knowledge Retrieval] → [Planning] → [Generation] → [Testing] → [Refinement] → [Output]
                       ↑                                  |             |           |
                       └──────────────────────────────────┴─────────────┴───────────┘
```

### 2. Knowledge Graph

The knowledge graph component serves as the memory system for the agent:

- **Documentation Storage**: Library documentation, API references, and code examples indexed in vector stores
- **Relationship Mapping**: Connections between libraries, dependencies, and common usage patterns
- **Contextual Recall**: Ability to retrieve relevant context based on the coding task

The system uses MongoDB Atlas Vector Search and Pinecone for efficient vector storage and retrieval, allowing it to maintain a rich knowledge representation.

### 3. Documentation Ingestion

A critical aspect of the system is its ability to ingest and process documentation:

- **Automated Discovery**: Identifies required modules and libraries from user queries
- **Documentation Ingestion**: Processes documentation from GitHub repositories, official docs, and other sources
- **Chunking and Embedding**: Breaks down documentation into semantic chunks and creates embeddings
- **Knowledge Integration**: Adds new information to the knowledge graph

When a user asks for code involving libraries the system doesn't know about, it can automatically fetch and process the relevant documentation.

### 4. Code Testing & Evaluation

The system implements multiple layers of code validation:

- **Static Analysis**: Syntax checking and import validation
- **Runtime Testing**: Execution validation in isolated environments
- **Quality Gates**: Enforced standards that code must pass before being presented
- **Iterative Refinement**: Feedback loops for improving code quality

### 5. Tool Integration

The agent can leverage various tools to enhance its capabilities:

- **GitHub Tools**: Access to repositories for examples and reference implementations
- **Module Documentation Tools**: Automatic retrieval of library documentation
- **Code Testing Tools**: Verification of generated code quality
- **Format and Presentation Tools**: Consistent presentation of results

## Setup

### Prerequisites

- Python 3.9+
- LangGraph 0.2.6+
- Access to OpenAI, Anthropic, or other supported LLM providers

### Environment Variables

Create a `.env` file with the following variables:

```
# LLM API Keys
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key

# For Pinecone integration
PINECONE_API_KEY=your_pinecone_key
PINECONE_INDEX_NAME=your_index_name

# For MongoDB integration
MONGODB_URI=your_mongodb_uri
```

### Installation

```bash
pip install -e .
```

## Usage

### Running the Assistant

```python
from code_assistant import graph, InputState
from langchain_core.messages import HumanMessage

# Initialize the graph with a user question
response = graph.invoke(
    {
        "messages": [
            HumanMessage(content="I need to create a REST API with FastAPI that connects to MongoDB. Here's the MongoDB documentation: [documentation provided here]")
        ],
        "documentation": None  # Optional: provide pre-processed documentation
    }
)

# Print the generated code
solution = response["generation"]
print(f"Solution description: {solution.prefix}")
print(f"Imports:\n{solution.imports}")
print(f"Code:\n{solution.code}")
```

### Configuration

You can customize the behavior of the Code Assistant through the configuration:

```python
from code_assistant import graph, InputState, Configuration

# Custom configuration
config = Configuration(
    user_id="user123",
    code_gen_model="anthropic/claude-3-sonnet-20240229",
    max_iterations=5,
    reflection_enabled=True
)

# Run with custom configuration
response = graph.invoke(
    {
        "messages": [HumanMessage(content="...")] 
    },
    config={"configurable": config.__dict__}
)
```

## Extending the Assistant

### Adding New Document Sources

You can extend the `process_documentation` function in `graph.py` to handle different documentation sources:

- URLs (using web scrapers)
- Local files (PDF, Markdown, etc.)
- API documentation
- GitHub repositories

### Supporting New Vector Stores

The assistant currently supports Pinecone and MongoDB for document retrieval. To add a new vector store:

1. Create a new retriever function in `utils.py`
2. Update the `make_retriever` function in `utils.py`
3. Add the new provider to the `Configuration` class in `configuration.py`

## License

This project is licensed under the MIT License - see the LICENSE file for details.