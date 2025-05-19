# Code Assistant with LangGraph

A powerful code generation assistant built with LangGraph that:

- Ingests documentation about libraries or modules from user input
- Generates production-ready code based on user requirements
- Tests the generated code for correctness
- Analyzes and fixes errors when they occur

The assistant uses a graph-based approach that separates concerns into distinct processing nodes, allowing for a systematic approach to code generation and testing.

## Architecture

The Code Assistant graph consists of the following nodes:

1. **Process Documentation**: Extracts and processes documentation provided by the user
2. **Generate Code**: Creates production-ready code based on the user requirements and documentation
3. **Check Code**: Tests the generated code for correctness and identifies errors
4. **Reflect**: (Optional) Analyzes errors to improve subsequent generation attempts

![Code Assistant Flow](./assets/code_assistant_flow.png)

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