"""Define the configurable parameters for the code assistant."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Annotated, Any, Literal, Optional, Type, TypeVar
import os

from langchain_core.runnables import RunnableConfig, ensure_config

from code_assistant import prompts


@dataclass(kw_only=True)
class DocumentConfiguration:
    """Configuration class for document indexing and retrieval operations.

    This class defines the parameters needed for storing and retrieving
    documentation for the code assistant, including user identification,
    embedding model selection, retriever provider choice, and search parameters.
    """

    user_id: str = field(metadata={"description": "Unique identifier for the user."})

    embedding_model: Annotated[
        str,
        {"__template_metadata__": {"kind": "embeddings"}},
    ] = field(
        default="openai/text-embedding-3-small",
        metadata={
            "description": "Name of the embedding model to use. Must be a valid embedding model name."
        },
    )

    retriever_provider: Annotated[
        Literal["pinecone", "mongodb"],
        {"__template_metadata__": {"kind": "retriever"}},
    ] = field(
        default="pinecone",
        metadata={
            "description": "The vector store provider to use for retrieval. Options are 'pinecone' or 'mongodb'."
        },
    )

    search_kwargs: dict[str, Any] = field(
        default_factory=dict,
        metadata={
            "description": "Additional keyword arguments to pass to the search function of the retriever."
        },
    )

    @classmethod
    def from_runnable_config(
        cls: Type[T], config: Optional[RunnableConfig] = None
    ) -> T:
        """Create a DocumentConfiguration instance from a RunnableConfig object.

        Args:
            cls (Type[T]): The class itself.
            config (Optional[RunnableConfig]): The configuration object to use.

        Returns:
            T: An instance of DocumentConfiguration with the specified configuration.
        """
        config = ensure_config(config)
        configurable = config.get("configurable") or {}
        _fields = {f.name for f in fields(cls) if f.init}
        return cls(**{k: v for k, v in configurable.items() if k in _fields})


T = TypeVar("T", bound=DocumentConfiguration)


@dataclass(kw_only=True)
class Configuration(DocumentConfiguration):
    """The configuration for the code assistant."""

    # Code generation model
    code_gen_system_prompt: str = field(
        default=prompts.CODE_GEN_SYSTEM_PROMPT,
        metadata={"description": "The system prompt used for generating code solutions."},
    )

    code_gen_system_prompt_claude: str = field(
        default=prompts.CODE_GEN_SYSTEM_PROMPT_CLAUDE,
        metadata={"description": "The system prompt used for generating code solutions with Claude."},
    )

    code_gen_model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default_factory=lambda: os.getenv("CODE_GEN_MODEL", "anthropic/claude-3-opus-20240229"),
        metadata={
            "description": "The language model used for generating code. Should be in the form: provider/model-name. Can be overridden with CODE_GEN_MODEL env variable."
        },
    )

    # Reflection model
    reflection_prompt: str = field(
        default=prompts.REFLECTION_PROMPT,
        metadata={
            "description": "The prompt used for reflection on code errors."
        },
    )

    reflection_model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default_factory=lambda: os.getenv("REFLECTION_MODEL", "anthropic/claude-3-haiku-20240307"),
        metadata={
            "description": "The language model used for reflection. Should be in the form: provider/model-name. Can be overridden with REFLECTION_MODEL env variable."
        },
    )

    # Code testing parameters
    max_iterations: int = field(
        default=3,
        metadata={
            "description": "Maximum number of code generation attempts before returning the final solution."
        },
    )

    reflection_enabled: bool = field(
        default=False,
        metadata={
            "description": "Whether to enable the reflection step for error analysis."
        },
    )

    pinecone_api_key: str = field(
        default_factory=lambda: os.getenv("PINECONE_API_KEY", ""),
        metadata={"description": "API key for Pinecone vector database."},
    )
    pinecone_index: str = field(
        default_factory=lambda: os.getenv("PINECONE_INDEX_NAME", "codeassistant"),
        metadata={"description": "Pinecone index name for vector storage."},
    )
    mongodb_uri: str = field(
        default_factory=lambda: os.getenv("MONGODB_URI", ""),
        metadata={"description": "MongoDB URI for document storage."},
    )
    embedding_model_name: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL_NAME", "openai/text-embedding-3-small"),
        metadata={"description": "Embedding model name for vectorization."},
    )