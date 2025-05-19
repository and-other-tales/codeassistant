"""Utility functions for the code assistant.

This module contains utility functions for handling documents, messages,
code testing, and other common operations in the code assistant project.

Functions:
    get_message_text: Extract text content from various message formats.
    format_docs: Convert documents to an xml-formatted string.
    load_chat_model: Load a chat model by provider/model name.
    check_imports: Test if imports in generated code are valid.
    check_code_execution: Test if generated code executes without errors.
    extract_documentation: Extract documentation from messages or documents.
"""

import os
from typing import Any, Optional, Sequence

from bs4 import BeautifulSoup as Soup
from langchain.chat_models import init_chat_model
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AnyMessage
from langchain_core.retrievers import VectorStoreRetriever
from langchain_core.vectorstores import VectorStoreRetriever


def get_message_text(msg: AnyMessage) -> str:
    """Get the text content of a message.

    This function extracts the text content from various message formats.

    Args:
        msg (AnyMessage): The message object to extract text from.

    Returns:
        str: The extracted text content of the message.

    Examples:
        >>> from langchain_core.messages import HumanMessage
        >>> get_message_text(HumanMessage(content="Hello"))
        'Hello'
        >>> get_message_text(HumanMessage(content={"text": "World"}))
        'World'
        >>> get_message_text(HumanMessage(content=[{"text": "Hello"}, " ", {"text": "World"}]))
        'Hello World'
    """
    content = msg.content
    if isinstance(content, str):
        return content
    elif isinstance(content, dict):
        return content.get("text", "")
    else:
        txts = [c if isinstance(c, str) else (c.get("text") or "") for c in content]
        return "".join(txts).strip()


def _format_doc(doc: Document) -> str:
    """Format a single document as XML.

    Args:
        doc (Document): The document to format.

    Returns:
        str: The formatted document as an XML string.
    """
    metadata = doc.metadata or {}
    meta = "".join(f" {k}={v!r}" for k, v in metadata.items())
    if meta:
        meta = f" {meta}"

    return f"<document{meta}>\n{doc.page_content}\n</document>"


def format_docs(docs: Optional[Sequence[Document]]) -> str:
    """Format a list of documents as XML.

    This function takes a list of Document objects and formats them into a single XML string.

    Args:
        docs (Optional[list[Document]]): A list of Document objects to format, or None.

    Returns:
        str: A string containing the formatted documents in XML format.

    Examples:
        >>> docs = [Document(page_content="Hello"), Document(page_content="World")]
        >>> print(format_docs(docs))
        <documents>
        <document>
        Hello
        </document>
        <document>
        World
        </document>
        </documents>

        >>> print(format_docs(None))
        <documents></documents>
    """
    if not docs:
        return "<documents></documents>"
    formatted = "\n".join(_format_doc(doc) for doc in docs)
    return f"""<documents>
{formatted}
</documents>"""


def load_chat_model(fully_specified_name: str) -> BaseChatModel:
    """Load a chat model from a fully specified name.

    Args:
        fully_specified_name (str): String in the format 'provider/model'.
    """
    if "/" in fully_specified_name:
        provider, model = fully_specified_name.split("/", maxsplit=1)
    else:
        provider = ""
        model = fully_specified_name
    return init_chat_model(model, model_provider=provider)


def check_imports(imports: str) -> tuple[bool, Optional[str]]:
    """Check if the provided imports execute without errors.
    
    Args:
        imports (str): The import statements to test.
        
    Returns:
        tuple[bool, Optional[str]]: A tuple containing a boolean indicating success
                                   and an optional error message.
    """
    try:
        exec(imports)
        return True, None
    except Exception as e:
        return False, str(e)


def check_code_execution(imports: str, code: str) -> tuple[bool, Optional[str]]:
    """Check if the provided code executes without errors.
    
    Args:
        imports (str): The import statements.
        code (str): The code to test.
        
    Returns:
        tuple[bool, Optional[str]]: A tuple containing a boolean indicating success
                                   and an optional error message.
    """
    try:
        exec(imports + "\n" + code)
        return True, None
    except Exception as e:
        return False, str(e)


def extract_documentation_from_url(url: str) -> list[Document]:
    """Extract documentation from a URL.
    
    Args:
        url (str): The URL to extract documentation from.
        
    Returns:
        list[Document]: A list of Document objects containing the documentation.
    """
    from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader
    
    loader = RecursiveUrlLoader(
        url=url, 
        max_depth=5, 
        extractor=lambda x: Soup(x, "html.parser").text
    )
    docs = loader.load()
    
    # Sort and process the documents as needed
    return sorted(docs, key=lambda x: x.metadata.get("source", ""))


def make_text_encoder(model: str) -> Embeddings:
    """Connect to the configured text encoder.
    
    Args:
        model (str): String in the format 'provider/model'.
        
    Returns:
        Embeddings: The configured embedding model.
    """
    provider, model = model.split("/", maxsplit=1)
    match provider:
        case "openai":
            from langchain_openai import OpenAIEmbeddings
            return OpenAIEmbeddings(model=model)
        case "cohere":
            from langchain_cohere import CohereEmbeddings
            return CohereEmbeddings(model=model)
        case _:
            raise ValueError(f"Unsupported embedding provider: {provider}")


def make_pinecone_retriever(
    user_id: str, 
    embedding_model: Embeddings,
    search_kwargs: dict[str, Any] = None
) -> VectorStoreRetriever:
    """Configure a Pinecone retriever.
    
    Args:
        user_id (str): User ID for filtering results.
        embedding_model (Embeddings): The embedding model to use.
        search_kwargs (dict[str, Any], optional): Additional search parameters.
        
    Returns:
        VectorStoreRetriever: The configured retriever.
    """
    from langchain_pinecone import PineconeVectorStore
    
    search_kwargs = search_kwargs or {}
    
    search_filter = search_kwargs.setdefault("filter", {})
    search_filter.update({"user_id": user_id})
    
    vstore = PineconeVectorStore.from_existing_index(
        os.environ["PINECONE_INDEX_NAME"], embedding=embedding_model
    )
    return vstore.as_retriever(search_kwargs=search_kwargs)


def make_mongodb_retriever(
    user_id: str, 
    embedding_model: Embeddings,
    search_kwargs: dict[str, Any] = None
) -> VectorStoreRetriever:
    """Configure a MongoDB Atlas retriever.
    
    Args:
        user_id (str): User ID for filtering results.
        embedding_model (Embeddings): The embedding model to use.
        search_kwargs (dict[str, Any], optional): Additional search parameters.
        
    Returns:
        VectorStoreRetriever: The configured retriever.
    """
    from langchain_mongodb.vectorstores import MongoDBAtlasVectorSearch
    
    search_kwargs = search_kwargs or {}
    
    pre_filter = search_kwargs.setdefault("pre_filter", {})
    pre_filter["user_id"] = {"$eq": user_id}
    
    vstore = MongoDBAtlasVectorSearch.from_connection_string(
        os.environ["MONGODB_URI"],
        namespace="code_assistant.documentation",
        embedding=embedding_model,
    )
    return vstore.as_retriever(search_kwargs=search_kwargs)