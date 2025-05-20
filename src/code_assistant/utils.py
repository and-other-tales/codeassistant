"""Utility functions for the code assistant.

This module contains utility functions for handling documents, messages,
code testing, and other common operations in the code assistant project.

Functions:
    get_message_text: Extract text content from various message formats.
    format_docs: Convert documents to an XML format.
    load_chat_model: Load a chat model by provider/model name.
    check_imports: Test if imports in generated code are valid.
    check_code_execution: Test if generated code executes without errors.
    extract_documentation_from_url: Extract documentation from a URL.
    make_text_encoder: Create a text encoder for embeddings.
    make_pinecone_retriever: Create a Pinecone vector store retriever.
    make_mongodb_retriever: Create a MongoDB Atlas vector store retriever.
    get_document_from_mongodb: Retrieve a document from MongoDB.
    ingest_github_repo: Ingest a GitHub repository into the document storage.
"""

import os
import tempfile
import shutil
import requests
import glob
import git
import re
import subprocess
import json
import asyncio
import logging
from typing import Any, Optional, Sequence, List, Dict, Union, Iterable, Tuple

from bs4 import BeautifulSoup as Soup
from langchain.chat_models import init_chat_model
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AnyMessage, BaseMessage
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_mongodb import MongoDBAtlasVectorSearch
from pymongo import MongoClient
import nbformat
from pydantic import SecretStr

logger = logging.getLogger(__name__)

__all__ = [
    'get_message_text',
    'format_docs',
    'load_chat_model',
    'check_imports',
    'check_code_execution',
    'extract_documentation_from_url',
    'make_text_encoder',
    'make_pinecone_retriever',
    'make_mongodb_retriever',
    'get_document_from_mongodb',
    'ingest_github_repo'
]


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
    """Load a chat model from a fully specified name, including Groq support.
    
    This function loads a chat model based on a fully qualified name in the format
    provider/model, such as 'openai/gpt-4', 'anthropic/claude-3-opus', or 'groq/llama3-8b-8192'.
    
    Args:
        fully_specified_name (str): The fully qualified model name in provider/model format.
        
    Returns:
        BaseChatModel: An initialized chat model ready to use.
        
    Raises:
        ValueError: If the model provider is not supported or if required API keys are missing.
    """
    try:
        if "/" in fully_specified_name:
            provider, model = fully_specified_name.split("/", maxsplit=1)
        else:
            provider = ""
            model = fully_specified_name

        if provider.lower() == "groq":
            # Use our custom ChatGroq implementation for better tool support
            from code_assistant.groq_tools import ChatGroq
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                raise ValueError("GROQ_API_KEY environment variable must be set for Groq models.")
            
            from langchain.pydantic_v1 import SecretStr
            secret_key = SecretStr(api_key)
            return ChatGroq(model=model, groq_api_key=secret_key)
        elif provider.lower() == "openai":
            return init_chat_model(model, model_provider="openai")
        elif provider.lower() == "anthropic":
            return init_chat_model(model, model_provider="anthropic")
        else:
            # If provider is not recognized, try passing the full name (for future compatibility)
            return init_chat_model(fully_specified_name)
    except Exception as e:
        print(f"Error loading chat model: {e}")
        raise ValueError(f"Failed to load chat model: {e}")


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


async def check_code_execution(code: str) -> Dict[str, Any]:
    """Check if the provided code executes without errors.
    
    Args:
        code (str): The code to test.
        
    Returns:
        Dict[str, Any]: A dictionary with the following keys:
            - success (bool): Whether the code executed successfully
            - error (Optional[str]): Error message if execution failed
            - description (str): Description of the execution result
    """
    try:
        # Extract imports from code
        import re
        import_lines = []
        code_lines = []
        
        for line in code.split('\n'):
            if line.strip().startswith('import ') or line.strip().startswith('from '):
                import_lines.append(line)
            else:
                code_lines.append(line)
                
        imports = '\n'.join(import_lines)
        code_without_imports = '\n'.join(code_lines)
        
        # Check imports first
        imports_success, imports_error = check_imports(imports)
        if not imports_success:
            return {
                "success": False,
                "error": f"Import error: {imports_error}",
                "description": f"Failed to import required modules: {imports_error}"
            }
        
        # Create a temporary globals dictionary to avoid modifying the global state
        globals_dict = {}
        
        # Execute the imports in the isolated globals dictionary
        exec(imports, globals_dict)
        
        # Execute the code in the same globals dictionary
        try:
            exec(code_without_imports, globals_dict)
            return {
                "success": True,
                "error": None,
                "description": "Code executed successfully"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "description": f"Runtime error: {str(e)}"
            }
    except Exception as e:
        # Catch any other exceptions that might occur during the checking process
        return {
            "success": False,
            "error": str(e),
            "description": f"Error during code verification: {str(e)}"
        }


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
            import cohere
            cohere_api_key = os.environ.get("COHERE_API_KEY")
            if not cohere_api_key:
                raise ValueError("COHERE_API_KEY environment variable must be set for Cohere embeddings.")
            client = cohere.Client(cohere_api_key)
            return CohereEmbeddings(model=model, client=client, async_client=None)
        case _:
            raise ValueError(f"Unsupported embedding provider: {provider}")


def make_pinecone_retriever(
    user_id: str, 
    embedding_model: Embeddings,
    search_kwargs: Optional[dict[str, Any]] = None
) -> VectorStoreRetriever:
    """Configure a Pinecone retriever.
    
    Args:
        user_id (str): User ID for filtering results.
        embedding_model: Embeddings: The embedding model to use.
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
    search_kwargs: Optional[dict[str, Any]] = None
) -> VectorStoreRetriever:
    """Configure a MongoDB Atlas retriever.
    
    Args:
        user_id (str): User ID for filtering results.
        embedding_model: Embeddings: The embedding model to use.
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


def get_document_from_mongodb(doc_id: str):
    """Fetch a document from MongoDB by its ID. Returns a langchain Document or None."""
    import os
    from pymongo import MongoClient
    from langchain_core.documents import Document
    MONGODB_URI = os.environ.get("MONGODB_URI")
    if not MONGODB_URI:
        return None
    client = MongoClient(MONGODB_URI)
    db = client["codeassist"]  # Specify the database name
    collection = db["docs"]  # Specify the collection name
    doc = collection.find_one({"_id": doc_id})
    if not doc:
        return None
    return Document(page_content=doc.get("page_content", ""), metadata=doc.get("metadata", {}))


async def ingest_github_repo(
    repo_url: str,
    mongodb_uri: str,
    pinecone_index: str,
    pinecone_api_key: str,
    embedding_model_name: str
) -> bool:
    """Ingest a GitHub repository into the document storage.
    
    Args:
        repo_url: URL of the GitHub repository to ingest
        mongodb_uri: MongoDB connection URI
        pinecone_index: Name of the Pinecone index to use
        pinecone_api_key: Pinecone API key
        embedding_model_name: Name of the embedding model to use
        
    Returns:
        bool: True if ingestion was successful, False otherwise
    """
    try:
        logger.info(f"Ingesting GitHub repository: {repo_url}")
        
        # Import necessary modules
        from langchain_community.document_loaders import GitLoader
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from langchain_mongodb import MongoDBAtlasVectorSearch
        from pymongo import MongoClient
        from langchain_community.embeddings import HuggingFaceEmbeddings
        import tempfile
        
        # Extract repo owner and name from URL
        repo_parts = repo_url.strip('/').split('/')
        if len(repo_parts) < 5 or repo_parts[2] != 'github.com':
            logger.error(f"Invalid GitHub repository URL: {repo_url}")
            return False
            
        repo_owner = repo_parts[3]
        repo_name = repo_parts[4]
        
        # Create a temporary directory for cloning the repo
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                # Clone the repository using GitLoader
                loader = GitLoader(
                    clone_url=repo_url,
                    repo_path=temp_dir,
                    branch="main"
                )
                
                # Load and split documents
                documents = loader.load()
                
                # Add metadata to documents
                for doc in documents:
                    metadata = doc.metadata or {}
                    metadata.update({
                        "source": "github",
                        "repo_owner": repo_owner,
                        "repo_name": repo_name,
                        "module": repo_name
                    })
                    doc.metadata = metadata
                
                # Split documents into chunks
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200
                )
                chunks = text_splitter.split_documents(documents)
                
                # Initialize MongoDB client
                client = MongoClient(mongodb_uri)
                db = client.get_default_database()
                collection = db.documents
                
                # Check if documents from this repo already exist
                existing = collection.count_documents({
                    "metadata.repo_owner": repo_owner,
                    "metadata.repo_name": repo_name
                })
                
                if existing > 0:
                    logger.info(f"Repository {repo_owner}/{repo_name} already exists in MongoDB")
                    # Optionally, delete existing documents before re-ingesting
                    collection.delete_many({
                        "metadata.repo_owner": repo_owner,
                        "metadata.repo_name": repo_name
                    })
                
                # Initialize embeddings
                embeddings = HuggingFaceEmbeddings(
                    model_name=embedding_model_name,
                    encode_kwargs={"normalize_embeddings": True}
                )
                
                # Initialize vector store
                vector_store = MongoDBAtlasVectorSearch.from_documents(
                    chunks,
                    embeddings,
                    collection=collection,
                    index_name=pinecone_index
                )
                
                logger.info(f"Successfully ingested {len(chunks)} chunks from {repo_owner}/{repo_name}")
                return True
                
            except Exception as e:
                logger.error(f"Error cloning or processing repository: {e}")
                return False
    except Exception as e:
        logger.error(f"Error in ingest_github_repo: {e}")
        return False