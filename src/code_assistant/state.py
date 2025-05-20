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
from typing import Annotated, Dict, List, Optional, Sequence, Any

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


def add_memory_item(
    existing: Optional[List[Dict[str, Any]]], 
    new_item: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Add a new memory item to the memory store.
    
    Args:
        existing (Optional[List[Dict[str, Any]]]): Existing memory items, if any.
        new_item (Dict[str, Any]): New memory item to add.
        
    Returns:
        List[Dict[str, Any]]: Updated list of memory items.
    """
    memory_list = list(existing or [])
    memory_list.append(new_item)
    return memory_list


def summarize_messages(messages: Sequence[AnyMessage], max_length: int = 1000) -> str:
    """Summarize a sequence of messages to keep context size manageable.
    
    Args:
        messages (Sequence[AnyMessage]): Messages to summarize.
        max_length (int, optional): Maximum length of the summary. Defaults to 1000.
        
    Returns:
        str: A summary of the messages.
    """
    # This is a simple implementation - in practice, you'd use an LLM to create a better summary
    all_content = " ".join([str(msg.content) for msg in messages])
    if len(all_content) <= max_length:
        return all_content
    
    # Simple truncation strategy
    return all_content[:max_length] + "... [truncated]"


def trim_messages_by_count(messages: Sequence[AnyMessage], max_count: int = 10) -> Sequence[AnyMessage]:
    """Trim messages to keep only the most recent ones.
    
    Args:
        messages (Sequence[AnyMessage]): Messages to trim.
        max_count (int, optional): Maximum number of messages to keep. Defaults to 10.
        
    Returns:
        Sequence[AnyMessage]: Trimmed sequence of messages.
    """
    if len(messages) <= max_count:
        return messages
    
    return messages[-max_count:]


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
    
    memory: Annotated[List[Dict[str, Any]], add_memory_item] = field(default_factory=list)
    """Long-term memory for the agent to remember important information across sessions."""
    
    conversation_summary: str = field(default="")
    """Summary of the conversation history for context management."""
    
    def get_documents(self) -> List[Document]:
        """Fetch the state documents.
        
        Returns:
            List[Document]: A list of documents used for context in code generation.
        """
        return list(self.documents or [])
    
    def add_to_memory(self, key: str, value: Any, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add an item to the agent's long-term memory.
        
        Args:
            key (str): Key to identify the memory item.
            value (Any): The value to store.
            metadata (Optional[Dict[str, Any]], optional): Additional metadata about this memory item.
        """
        metadata = metadata or {}
        self.memory = add_memory_item(self.memory, {
            "key": key,
            "value": value,
            "metadata": metadata,
            "timestamp": __import__('datetime').datetime.now().isoformat()
        })
    
    def get_from_memory(self, key: str) -> Optional[Any]:
        """Retrieve an item from memory by key.
        
        Args:
            key (str): The key to look for.
            
        Returns:
            Optional[Any]: The value if found, None otherwise.
        """
        for item in self.memory:
            if item.get("key") == key:
                return item.get("value")
        return None
    
    def search_memory(self, query: str) -> List[Dict[str, Any]]:
        """Simple search through memory items.
        
        In a real implementation, this would use vector-based semantic search.
        
        Args:
            query (str): The search query.
            
        Returns:
            List[Dict[str, Any]]: List of matching memory items.
        """
        # Simple implementation - in practice, you'd use vector similarity search
        results = []
        for item in self.memory:
            if query.lower() in str(item.get("value")).lower():
                results.append(item)
        return results
    
    def update_conversation_summary(self) -> None:
        """Update the conversation summary based on the current messages."""
        self.conversation_summary = summarize_messages(self.messages)
    
    def trim_messages(self, max_count: int = 10) -> None:
        """Trim the message history to keep context size manageable.
        
        Args:
            max_count (int, optional): Maximum number of messages to keep. Defaults to 10.
        """
        self.messages = trim_messages_by_count(self.messages, max_count)