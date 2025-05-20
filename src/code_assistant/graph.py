"""Main entrypoint for the code assistant graph.

This module defines the core structure and functionality of the code assistant graph.
It includes the main graph definition, state management, and key functions for
processing documentation, generating code, testing code, and incorporating feedback
for improved solutions.
"""

from typing import Dict, List, Optional, cast

from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph, START

from code_assistant.configuration import Configuration
from code_assistant.state import CodeSolution, GraphState, InputState
from code_assistant.utils import (
    check_code_execution,
    check_imports,
    format_docs, 
    get_message_text,
    load_chat_model,
)


class SearchQuery(BaseModel):
    """Search the indexed documents for a query."""

    query: str


async def process_documentation(
    state: GraphState, *, config: RunnableConfig
) -> Dict:
    """Process documentation from the user input and state.
    
    This function extracts or formats documentation either from the user's messages
    or from documents already in the state.
    
    Args:
        state (GraphState): The current state containing messages and/or documents.
        config (RunnableConfig): Configuration for the processing.
        
    Returns:
        Dict: A dictionary with updated documents.
    """
    messages = state.messages
    
    # If we already have documentation, use it
    if state.documentation:
        return {"documents": state.documentation}

    # If a MongoDB document ID/reference is present in the state, fetch from MongoDB
    doc_id = getattr(state, 'mongodb_doc_id', None)
    if doc_id:
        from code_assistant.utils import get_document_from_mongodb
        mongo_doc = get_document_from_mongodb(doc_id)
        if mongo_doc:
            return {"documents": [mongo_doc]}

    # Extract documentation from the latest message
    user_input = get_message_text(messages[-1])
    # For now, we'll just create a document from the user input
    from langchain_core.documents import Document
    doc = Document(page_content=user_input)
    return {"documents": [doc]}


async def generate_code(
    state: GraphState, *, config: RunnableConfig
) -> Dict:
    """Generate code based on the user's question and documentation.
    
    This function uses an LLM to generate code that addresses the user's requirements
    while utilizing the provided documentation.
    
    Args:
        state (GraphState): The current state with user messages and documentation.
        config (RunnableConfig): Configuration for code generation.
        
    Returns:
        Dict: A dictionary with the generated code solution and updated messages.
    """
    print("---GENERATING CODE SOLUTION---")
    
    # Get configuration
    configuration = Configuration.from_runnable_config(config)
    
    # Get state components
    messages = state.messages
    iterations = state.iterations
    error = state.error
    documents = state.documents
    
    # Format documentation
    formatted_docs = format_docs(documents)
    
    # Detect if the user is just greeting or starting a chat
    user_text = get_message_text(messages[-1]).strip().lower()
    greeting_phrases = ["hi", "hello", "hey", "good morning", "good afternoon", "good evening"]
    if any(user_text.startswith(greet) for greet in greeting_phrases):
        # Respond with a natural language introduction
        intro = (
            "Hello! I'm your code assistant. "
            "You can ask me to generate code, ingest documentation, or help with your coding tasks. "
            "For example, you can say: 'Please ingest the GitHub repository at ...', "
            "or 'I'd like you to create a FastAPI endpoint.'\n\nHow can I help you today?"
        )
        # Use AIMessage for consistency with message types
        messages = list(messages) + [AIMessage(content=intro)]
        # Set special error flag to end the graph after intro
        return {"messages": messages, "iterations": iterations, "error": "__intro__", "generation": None}
    
    # If we have an error, prepare to regenerate with error info
    if error and error != "__intro__":
        messages = list(messages) + [
            AIMessage(content=(
                f"Your solution failed with the following error: {error}. "
                "Please fix the issues and provide a corrected solution. "
                "Make sure to invoke the code tool to structure your output correctly."
            ))
        ]
    
    # Prepare prompt and model
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                configuration.code_gen_system_prompt_claude,
            ),
            ("placeholder", "{messages}"),
        ]
    )
    
    model = load_chat_model(configuration.code_gen_model)
    structured_model = model.with_structured_output(CodeSolution, include_raw=True)
    
    # Generate code solution
    message_value = await prompt.ainvoke(
        {
            "messages": state.messages,
            "context": formatted_docs,
        },
        config,
    )
    
    raw_result = await structured_model.ainvoke(message_value, config)
    
    # Handle potential tool invocation failures (Claude sometimes struggles with tool use)
    parsing_error = getattr(raw_result, 'parsing_error', None)
    if parsing_error:
        # Fallback to retry with stronger tool use instruction
        fallback_messages = list(messages)
        raw_content = getattr(getattr(raw_result, 'raw', None), 'content', '')
        fallback_messages.append(
            AIMessage(content=f"I'll help create code based on the documentation. {raw_content}")
        )
        fallback_messages.append(
            AIMessage(content="Please try again, and make sure to invoke the 'code' tool to structure your response with prefix, imports, and code fields.")
        )
        # Try again with the fallback model
        message_value = await prompt.ainvoke(
            {
                "messages": fallback_messages,
                "context": formatted_docs,
            },
            config,
        )
        raw_result = await structured_model.ainvoke(message_value, config)
        parsing_error = getattr(raw_result, 'parsing_error', None)
        if parsing_error:
            print("---TOOL PARSING FAILED, USING FALLBACK EXTRACTION---")
            raw_content = getattr(getattr(raw_result, 'raw', None), 'content', '')
            import re
            prefix_match = re.search(r"(.*?)(?=```|Imports:|imports:)", raw_content, re.DOTALL)
            prefix = prefix_match.group(1).strip() if prefix_match else "Code solution generated"
            imports_match = re.search(r"(?:Imports:|imports:)(.*?)(?=```|Code:|code:)", raw_content, re.DOTALL)
            imports = imports_match.group(1).strip() if imports_match else ""
            code_match = re.search(r"```(?:python)?\s*(.*?)```", raw_content, re.DOTALL)
            code = code_match.group(1).strip() if code_match else ""
            code_solution = CodeSolution(
                prefix=prefix,
                imports=imports,
                code=code
            )
        else:
            code_solution = getattr(raw_result, 'parsed', None)
    else:
        code_solution = getattr(raw_result, 'parsed', None)
    
    # Append solution to messages only if a solution exists
    if code_solution:
        formatted_solution = (
            f"{code_solution.prefix} \n\nImports:\n```python\n{code_solution.imports}\n```"
            f"\n\nCode:\n```python\n{code_solution.code}\n```"
        )
        messages = list(messages) + [AIMessage(content=formatted_solution)]
        # Increment iterations
        iterations = iterations + 1
    
    return {
        "generation": code_solution, 
        "messages": messages, 
        "iterations": iterations,
        "error": "" if code_solution is not None else error  # Only reset error if code_solution exists
    }


async def check_code(
    state: GraphState, *, config: RunnableConfig
) -> Dict:
    """Check if the generated code runs without errors.
    
    This function tests the imports and code execution of the generated solution
    to ensure it works correctly.
    
    Args:
        state (GraphState): The current state with the generated code solution.
        config (RunnableConfig): Configuration for code checking.
        
    Returns:
        Dict: A dictionary with test results and potential error information.
    """
    print("---CHECKING CODE---")
    
    # Guard: if the special intro error flag is set, end immediately
    if state.error == "__intro__":
        return {"error": "__intro__"}
    
    # Get state components
    code_solution = state.generation
    messages = state.messages
    iterations = state.iterations
    
    if not code_solution:
        return {"error": "no_solution"}
    
    # Extract code components
    imports = code_solution.imports
    code = code_solution.code
    
    # Check imports
    imports_ok, import_error = check_imports(imports)
    if not imports_ok:
        print("---CODE IMPORT CHECK: FAILED---")
        return {
            "error": f"Import error: {import_error}"
        }
    
    # Check code execution
    code_ok, code_error = check_code_execution(imports, code)
    if not code_ok:
        print("---CODE BLOCK CHECK: FAILED---")
        return {
            "error": f"Code execution error: {code_error}"
        }
    
    # No errors
    print("---NO CODE TEST FAILURES---")
    return {"error": ""}


async def reflect(
    state: GraphState, *, config: RunnableConfig
) -> Dict:
    """Reflect on code errors and provide analysis for improvement.
    
    This function analyzes errors in the code generation process and provides
    insights to improve the next generation attempt.
    
    Args:
        state (GraphState): The current state with error information.
        config (RunnableConfig): Configuration for reflection.
        
    Returns:
        Dict: A dictionary with updated messages including reflection.
    """
    print("---REFLECTING ON ERRORS---")
    
    # Get configuration
    configuration = Configuration.from_runnable_config(config)
    
    # Get state components
    messages = state.messages
    error = state.error
    
    # Create reflection prompt
    reflection_prompt = ChatPromptTemplate.from_template(configuration.reflection_prompt)
    
    # Create reflection message
    reflection_model = load_chat_model(configuration.reflection_model)
    
    # Add error information and request reflection
    reflection_messages = list(messages)
    reflection_messages.append(
        AIMessage(content=f"I encountered this error while trying to generate code: {error}")
    )
    
    reflection = await reflection_model.ainvoke(
        reflection_prompt.format_messages(error=error),
        config,
    )
    
    # Add reflection to messages
    messages = list(messages) + [
        AIMessage(content=f"Here's my analysis of the error: {reflection.content}")
    ]
    
    return {"messages": messages}


def decide_next_step(state: GraphState) -> str:
    """Determine the next node in the graph based on the current state.
    
    Args:
        state (GraphState): The current graph state.
        
    Returns:
        str: The name of the next node to execute.
    """
    # Get configuration (using default values since we can't access config here)
    max_iterations = 3  # Default, will be overridden by config when available
    reflection_enabled = False  # Default, will be overridden by config
    
    # Get state components
    error = state.error
    iterations = state.iterations
    
    # End the graph if the special intro error flag is set
    if error == "__intro__":
        print("---DECISION: END AFTER INTRO---")
        return "end"
    
    # Decision logic
    if not error:
        print("---DECISION: FINISH---")
        return "end"
    
    if iterations >= max_iterations:
        print("---DECISION: MAX ITERATIONS REACHED---")
        return "end"
    
    if error and reflection_enabled:
        print("---DECISION: REFLECT ON ERROR---")
        return "reflect"
    
    print("---DECISION: RETRY GENERATION---")
    return "generate"


# Build the graph
builder = StateGraph(GraphState, input=InputState, config_schema=Configuration)

# Add nodes
builder.add_node("process_documentation", process_documentation)
builder.add_node("generate", generate_code)
builder.add_node("check_code", check_code)
builder.add_node("reflect", reflect)

# Add edges
builder.add_edge(START, "process_documentation")
builder.add_edge("process_documentation", "generate")
builder.add_edge("generate", "check_code")
builder.add_conditional_edges(
    "check_code",
    decide_next_step,
    {
        "end": END,
        "reflect": "reflect",
        "generate": "generate",
    },
)
builder.add_edge("reflect", "generate")

# Compile the graph
graph = builder.compile()
graph.name = "CodeAssistant"