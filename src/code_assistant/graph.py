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
from langchain_core.pydantic_v1 import BaseModel
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
    
    # Extract documentation from the latest message
    user_input = get_message_text(messages[-1])
    
    # For now, we'll just create a document from the user input
    # In a real application, you would parse URLs, file paths, etc.
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
    
    # If we have an error, prepare to regenerate with error info
    if error:
        messages += [
            (
                "user",
                f"Your solution failed with the following error: {error}. "
                "Please fix the issues and provide a corrected solution. "
                "Make sure to invoke the code tool to structure your output correctly.",
            )
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
    if raw_result.get("parsing_error"):
        # Fallback to retry with stronger tool use instruction
        fallback_messages = list(messages)
        fallback_messages.append(
            AIMessage(content=f"I'll help create code based on the documentation. {raw_result['raw'].content}")
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
        
        # If still failing, create a basic structure
        if raw_result.get("parsing_error"):
            print("---TOOL PARSING FAILED, USING FALLBACK EXTRACTION---")
            raw_content = raw_result["raw"].content
            
            # Simple extraction logic - would be more robust in production
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
            code_solution = cast(CodeSolution, raw_result["parsed"])
    else:
        code_solution = cast(CodeSolution, raw_result["parsed"])
    
    # Append solution to messages
    formatted_solution = (
        f"{code_solution.prefix} \n\nImports:\n```python\n{code_solution.imports}\n```"
        f"\n\nCode:\n```python\n{code_solution.code}\n```"
    )
    
    messages += [
        (
            "assistant",
            formatted_solution,
        )
    ]
    
    # Increment iterations
    iterations = iterations + 1
    
    return {
        "generation": code_solution, 
        "messages": messages, 
        "iterations": iterations,
        "error": ""  # Reset error state
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
    messages += [
        (
            "assistant",
            f"Here's my analysis of the error: {reflection.content}",
        )
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