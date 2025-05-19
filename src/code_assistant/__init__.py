"""Code Assistant using LangGraph.

This package provides a graph-based implementation of a Code Assistant that can:
1. Ingest documentation from various sources
2. Generate production-ready code based on user requirements
3. Test and validate the code for correctness
4. Analyze and fix errors when they occur

The assistant is built using the LangGraph framework and can work with
various vector stores for document retrieval (Pinecone, MongoDB).
"""

from code_assistant.configuration import Configuration, DocumentConfiguration
from code_assistant.graph import graph
from code_assistant.state import CodeSolution, GraphState, InputState

__all__ = [
    "graph",
    "Configuration",
    "DocumentConfiguration",
    "CodeSolution",
    "GraphState",
    "InputState",
]