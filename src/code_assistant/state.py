"""State management for the code assistant graph.

This module defines the state structures for the code assistant graph, which handles
the ingestion of documentation, code generation, code testing, and error correction
processes.

Classes:
    InputState: Represents the restricted input state for the graph.
    GraphState: Full state representation for the code assistant graph.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Annotated, List, Optional, Sequence

from langchain_core.documents import Document
from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages

from pydantic import BaseModel


# Define Pydantic model for code solution
class CodeSolution(BaseModel):
    """Schema for code solutions generated from the documentation."""

    prefix: str
    imports: str
    code: str


def add_documents(
    existing: Optional[Sequence[Document]], 
    new: Sequence[Document]
) -> Sequence[Document]:
    """Add new documents to the existing set of documents.
    
    Args:
        existing (Optional[Sequence[Document]]): The existing documents, if any.
        new (Sequence[Document]): New documents to add.
        
    Returns:
        Sequence[Document]: A sequence containing all documents.
    """
    return list(existing or []) + list(new)


@dataclass(kw_only=True)
class InputState:
    """Represents the input state for the agent.
    
    This is a restricted version of the full State that provides a narrower
    interface to the outside world.
    """
    
    messages: Annotated[Sequence[AnyMessage], add_messages]
    """Messages exchanged between the user and the agent."""
    
    documentation: Optional[Sequence[Document]] = None
    """Documentation provided by the user to inform code generation."""


@dataclass(kw_only=True)
class GraphState(InputState):
    """The full state of the code assistant graph.
    
    Includes all components needed for code generation, testing, and refinement.
    """
    
    error: str = field(default="")
    """Error flag to control code generation flow."""
    
    iterations: int = field(default=0)
    """Number of generation attempts made."""
    
    generation: Optional[CodeSolution] = None
    """The latest code solution generated."""
    
    documents: Annotated[Sequence[Document], add_documents] = field(default_factory=list)
    """Documents used for context in code generation."""
    
    def get_documents(self) -> List[Document]:
        """Fetch the state documents.
        
        Returns:
            List[Document]: A list of documents used for context in code generation.
        """
        return list(self.documents or [])