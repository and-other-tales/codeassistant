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
import tempfile
import shutil
import requests
import glob
import git
import re
from typing import Any, Optional, Sequence

from bs4 import BeautifulSoup as Soup
from langchain.chat_models import init_chat_model
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AnyMessage
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_mongodb import MongoDBAtlasVectorSearch
from pymongo import MongoClient
import nbformat
from pydantic import SecretStr


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
    """Load a chat model from a fully specified name, including Groq support."""
    if "/" in fully_specified_name:
        provider, model = fully_specified_name.split("/", maxsplit=1)
    else:
        provider = ""
        model = fully_specified_name

    if provider == "groq":
        # Support for ChatGroq
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable must be set for Groq models.")
        return ChatGroq(model=model, api_key=SecretStr(api_key))
    elif provider == "openai":
        return init_chat_model(model, model_provider="openai")
    elif provider == "anthropic":
        return init_chat_model(model, model_provider="anthropic")
    else:
        # If provider is not recognized, try passing the full name (for future compatibility)
        return init_chat_model(fully_specified_name)


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


def ingest_github_repo(repo_url: str, mongodb_uri: str, pinecone_index: str, pinecone_api_key: str, embedding_model_name: str = "openai/text-embedding-3-small") -> dict:
    """
    Ingests a GitHub repository: clones it, extracts docs/examples/cookbooks/ipynb, stores in MongoDB, and creates Pinecone vectors.
    Returns a summary dict.
    """
    temp_dir = tempfile.mkdtemp()
    try:
        # Clone the repo
        repo = git.Repo.clone_from(repo_url, temp_dir)
        # Find all relevant files
        folders = ["docs", "examples", "cookbooks"]
        file_patterns = [
            os.path.join(temp_dir, f, "**", "*.md") for f in folders
        ] + [
            os.path.join(temp_dir, f, "**", "*.rst") for f in folders
        ] + [
            os.path.join(temp_dir, f, "**", "*.ipynb") for f in folders
        ] + [
            os.path.join(temp_dir, f, "**", "*.py") for f in folders
        ]
        files = []
        for pattern in file_patterns:
            files.extend(glob.glob(pattern, recursive=True))
        # Parse files into documents
        docs = []
        for file in files:
            ext = os.path.splitext(file)[1]
            with open(file, "r", encoding="utf-8", errors="ignore") as f:
                if ext == ".ipynb":
                    nb = nbformat.read(f, as_version=4)
                    text = "\n".join(cell['source'] for cell in nb.cells if cell.cell_type == 'markdown' or cell.cell_type == 'code')
                else:
                    text = f.read()
            docs.append(Document(page_content=text, metadata={"source": file, "repo_url": repo_url}))
        # Store in MongoDB
        client = MongoClient(mongodb_uri)
        db = client["codeassist"]
        collection = db["documentation"]
        for doc in docs:
            collection.insert_one({"page_content": doc.page_content, "metadata": doc.metadata})
        # Create Pinecone vectors
        if docs:
            embeddings = OpenAIEmbeddings(model=embedding_model_name)
            pinecone_vs = PineconeVectorStore.from_existing_index(pinecone_index, embedding=embeddings)
            pinecone_vs.add_documents(docs)
        return {"status": "success", "files_ingested": len(docs)}
    except Exception as e:
        return {"status": "error", "error": str(e)}
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def ingest_repository(repo_url: str, mongodb_uri: str, pinecone_index: str, pinecone_api_key: str, embedding_model_name: str = "openai/text-embedding-3-small") -> dict:
    """
    Ingest any GitHub repository: clones, extracts docs/examples/cookbooks/notebooks, stores in MongoDB, and creates Pinecone vectors.
    """
    return ingest_github_repo(
        repo_url=repo_url,
        mongodb_uri=mongodb_uri,
        pinecone_index=pinecone_index,
        pinecone_api_key=pinecone_api_key,
        embedding_model_name=embedding_model_name
    )


def extract_required_modules(text: str) -> list[str]:
    """Extracts required module names from Python import statements in the given text."""
    pattern = r"(?:import\s+([\w_\.]+))|(?:from\s+([\w_\.]+)\s+import)"
    matches = re.findall(pattern, text)
    modules = set()
    for imp, frm in matches:
        if imp:
            modules.add(imp.split('.')[0])
        if frm:
            modules.add(frm.split('.')[0])
    return list(modules)


def documentation_exists(module: str, mongodb_uri: str) -> bool:
    """Checks if documentation for the given module exists in MongoDB."""
    client = MongoClient(mongodb_uri)
    db = client["codeassist"]
    collection = db["documentation"]
    # Check for any document with the module name in metadata or content
    doc = collection.find_one({
        "$or": [
            {"metadata.module": module},
            {"page_content": {"$regex": rf"\\b{module}\\b", "$options": "i"}}
        ]
    })
    return doc is not None


def fetch_and_ingest_repos(
    repo_urls: list[str],
    mongodb_uri: str,
    pinecone_index: str,
    pinecone_api_key: str,
    embedding_model_name: str = "openai/text-embedding-3-small"
) -> dict:
    """
    Ingest documentation and examples from a list of GitHub repository URLs.
    Returns a dict with ingestion results for each repo.
    """
    from code_assistant.utils import ingest_github_repo
    results = {}
    for repo_url in repo_urls:
        result = ingest_github_repo(
            repo_url=repo_url,
            mongodb_uri=mongodb_uri,
            pinecone_index=pinecone_index,
            pinecone_api_key=pinecone_api_key,
            embedding_model_name=embedding_model_name
        )
        results[repo_url] = result
    return results


def generate_code_with_doc_check(
    code_task: str,
    mongodb_uri: str,
    pinecone_index: str,
    pinecone_api_key: str,
    embedding_model_name: str = "openai/text-embedding-3-small",
    repo_url_map: Optional[dict] = None,
    code_generator=None,
    code_tester=None
) -> str:
    """
    Orchestrates code generation with documentation check and ingestion:
    - Extracts required modules from the code task.
    - For each module, checks if documentation exists in MongoDB.
    - If not, ingests the relevant repo (blocking until done).
    - Only proceeds with code generation when all docs are present.
    - Tests all generated code.
    - Returns the generated code.
    repo_url_map: Dict mapping module names to repo URLs. Must be provided for missing modules.
    code_generator: Callable that takes code_task and returns code (default: NotImplementedError).
    code_tester: Callable that takes code and returns test result (default: NotImplementedError).
    """
    if repo_url_map is None:
        repo_url_map = {}
    required_modules = extract_required_modules(code_task)
    for module in required_modules:
        if not documentation_exists(module, mongodb_uri):
            repo_url = repo_url_map.get(module)
            if not repo_url:
                raise ValueError(f"No repository URL provided for missing module: {module}. Please specify in repo_url_map.")
            ingest_repository(
                repo_url=repo_url,
                mongodb_uri=mongodb_uri,
                pinecone_index=pinecone_index,
                pinecone_api_key=pinecone_api_key,
                embedding_model_name=embedding_model_name
            )
    if code_generator is None:
        raise NotImplementedError("You must provide a code_generator callable.")
    code = code_generator(code_task)
    if code_tester is None:
        raise NotImplementedError("You must provide a code_tester callable.")
    test_result = code_tester(code)
    if not test_result:
        raise RuntimeError("Generated code did not pass tests.")
    return code


# --- SANDBOXED CODE EXECUTION ---

def run_code_in_pyodide_sandbox(code: str, allow_net: bool = False):
    """
    Executes code in a PyodideSandbox (langchain-sandbox) for secure, isolated execution.
    Returns the CodeExecutionResult object.

    NOTE: langchain-sandbox requires langgraph<0.4.0, which may conflict with your main project.
    If ImportError occurs, run this in a separate virtual environment or subprocess.
    """
    try:
        from langchain_sandbox import PyodideSandbox
        import asyncio
        import tempfile
        import shutil
    except ImportError as e:
        raise ImportError("langchain-sandbox is not installed or incompatible with current langgraph version. "
                          "Use a separate environment for sandboxed execution.\n" + str(e))
    sessions_dir = tempfile.mkdtemp()
    try:
        sandbox = PyodideSandbox(sessions_dir=sessions_dir, allow_net=allow_net)
        async def _run():
            return await sandbox.execute(code)
        return asyncio.run(_run())
    finally:
        shutil.rmtree(sessions_dir, ignore_errors=True)


# --- CODEACT AGENT INTEGRATION ---

def create_codeact_agent(model, tools, allow_net: bool = False):
    """
    Creates a CodeAct agent using PyodideSandbox for secure code execution.

    NOTE: langgraph-codeact requires langgraph<0.4.0, which may conflict with your main project.
    If ImportError occurs, run this in a separate virtual environment or subprocess.
    """
    try:
        from langgraph_codeact import create_codeact
        from langgraph.checkpoint.memory import MemorySaver
    except ImportError as e:
        raise ImportError("langgraph-codeact is not installed or incompatible with current langgraph version. "
                          "Use a separate environment for codeact agent execution.\n" + str(e))
    def pyodide_sandbox_eval(code: str, _locals: dict):
        result = run_code_in_pyodide_sandbox(code, allow_net=allow_net)
        return result.stdout or result.result, _locals
    code_act = create_codeact(model, tools, pyodide_sandbox_eval)
    agent = code_act.compile(checkpointer=MemorySaver())
    return agent


# --- EXAMPLE TEST TEMPLATE ---

def test_generated_code_with_codeact(model, tools, code: str, expected_output: str):
    """
    Test generated code using CodeAct agent and PyodideSandbox for correctness and safety.
    This function will raise ImportError if dependencies are not compatible.
    """
    agent = create_codeact_agent(model, tools)
    messages = [{"role": "user", "content": code}]
    result = agent.invoke({"messages": messages})
    # result is typically a list of message dicts; check all for expected output
    found = any(expected_output in (msg.get("content", "") or str(msg)) for msg in (result if isinstance(result, list) else [result]))
    assert found, f"Expected output not found. Got: {result}"


def get_github_tools(user_consent: bool = False):
    """
    Conditionally load GitHubToolkit tools if the user has provided GitHub credentials and consented.
    Returns a list of GitHub tools, or an empty list if not enabled.
    """
    if not user_consent:
        return []
    required_env = ["GITHUB_APP_ID", "GITHUB_APP_PRIVATE_KEY", "GITHUB_REPOSITORY"]
    if not all(os.getenv(var) for var in required_env):
        return []
    try:
        from langchain_community.agent_toolkits.github.toolkit import GitHubToolkit
        from langchain_community.utilities.github import GitHubAPIWrapper
        github = GitHubAPIWrapper()
        toolkit = GitHubToolkit.from_github_api_wrapper(github)
        return toolkit.get_tools()
    except ImportError:
        return []


def is_tool_use_supported(model_name: str) -> bool:
    """
    Returns True if the model supports tool use (function calling), e.g. ChatGroq Llama4, OpenAI GPT-4o, etc.
    Update this as new models are supported.
    """
    # Groq Llama-3/4, OpenAI GPT-4o, etc. (update as needed)
    model_name = model_name.lower()
    return any(
        kw in model_name for kw in ["llama-4", "llama-3", "gpt-4o", "gpt-4-turbo", "tool-use"]
    )


def build_agent_tools(user_tools: list, github_tools: list = None, ingestion_tools: list = None) -> list:
    """
    Build the list of tools to pass to the agent, including user, github, and ingestion tools.
    """
    tools = list(user_tools) if user_tools else []
    if github_tools:
        tools.extend(github_tools)
    if ingestion_tools:
        tools.extend(ingestion_tools)
    return tools