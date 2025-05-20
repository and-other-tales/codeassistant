"""Groq integration for LangChain with enhanced tool support.

This module provides improved Groq chat model integration for LangChain,
with proper support for tool calling and structured outputs.
"""

from __future__ import annotations

import json
import logging
import asyncio
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Type, Union, cast, Callable, AsyncIterator, Coroutine
from uuid import UUID

import requests
from langchain.pydantic_v1 import BaseModel, Field, SecretStr, root_validator
from langchain_core.callbacks import Callbacks
from langchain_core.callbacks.manager import CallbackManagerForLLMRun, AsyncCallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    BaseMessageChunk,
    ChatMessage,
    FunctionMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult, ChatGenerationChunk
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool

logger = logging.getLogger(__name__)

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"


class ChatGroq(BaseChatModel):
    """Groq chat model with tool calling support."""

    model: str = "llama3-8b-8192"
    """Model name to use."""
    
    temperature: float = 0.7
    """What sampling temperature to use."""
    
    max_tokens: Optional[int] = None
    """Maximum number of tokens to generate."""
    
    top_p: float = 1.0
    """Total probability mass of tokens to consider at each step."""
    
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """Holds any model parameters valid for the Groq API not explicitly specified."""
    
    groq_api_key: Optional[SecretStr] = None
    """Groq API key."""
    
    streaming: bool = False
    """Whether to stream the response."""

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that the API key is provided."""
        groq_api_key = values.get("groq_api_key")
        if groq_api_key is None or groq_api_key.get_secret_value() == "":
            try:
                import os
                groq_api_key = os.environ["GROQ_API_KEY"]
                values["groq_api_key"] = SecretStr(groq_api_key)
            except KeyError:
                raise ValueError(
                    "Groq API key not found. Please set the GROQ_API_KEY environment "
                    "variable or pass it to the constructor as groq_api_key."
                )
        return values

    @property
    def _llm_type(self) -> str:
        """Return type of LLM."""
        return "groq-chat"

    def _convert_messages_to_groq_format(
        self, messages: List[BaseMessage], tools: Optional[List[Dict]] = None
    ) -> Dict:
        """Convert messages to the format expected by the Groq API."""
        message_dicts = []
        for message in messages:
            message_dict = {"role": self._convert_message_to_role(message), "content": message.content}
            if isinstance(message, AIMessage) and message.tool_calls:
                groq_tool_calls = []
                for tool_call in message.tool_calls:
                    groq_tool_call = {
                        "id": tool_call.get("id", f"call_{len(groq_tool_calls)}"),
                        "type": "function",
                        "function": {
                            "name": tool_call.get("name", ""),
                            "arguments": json.dumps(tool_call.get("args", {})),
                        },
                    }
                    groq_tool_calls.append(groq_tool_call)
                if groq_tool_calls:
                    message_dict["tool_calls"] = groq_tool_calls
            if isinstance(message, ToolMessage):
                message_dict["role"] = "tool"
                message_dict["tool_call_id"] = message.tool_call_id
                message_dict["name"] = getattr(message, "name", None)
            message_dicts.append(message_dict)

        payload = {
            "model": self.model,
            "messages": message_dicts,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "stream": self.streaming,
            **self.model_kwargs,
        }

        if self.max_tokens is not None:
            payload["max_tokens"] = self.max_tokens
            
        if tools:
            # Llama 3 models in Groq support tool_choice "auto" which is the default
            payload["tools"] = tools

        return payload

    def _convert_message_to_role(self, message: BaseMessage) -> str:
        """Convert message to role for the Groq API."""
        if isinstance(message, ChatMessage):
            return message.role
        elif isinstance(message, HumanMessage):
            return "user"
        elif isinstance(message, AIMessage):
            return "assistant"
        elif isinstance(message, SystemMessage):
            return "system"
        elif isinstance(message, ToolMessage):
            return "tool"
        elif isinstance(message, FunctionMessage):
            return "function"
        else:
            raise ValueError(f"Unknown message type: {type(message)}")

    def _create_chat_result(self, response: Mapping[str, Any]) -> ChatResult:
        """Create a ChatResult from the response."""
        message = response["choices"][0]["message"]
        role = message.get("role", "assistant")
        content = message.get("content", "")
        
        tool_calls = None
        if "tool_calls" in message:
            tool_calls = []
            for tool_call in message["tool_calls"]:
                if tool_call["type"] == "function":
                    function_call = tool_call["function"]
                    tool_calls.append({
                        "id": tool_call["id"],
                        "type": "function",
                        "name": function_call["name"],
                        "args": json.loads(function_call["arguments"])
                    })
        
        additional_kwargs = {}
        if tool_calls:
            additional_kwargs["tool_calls"] = tool_calls
            
        message = AIMessage(content=content, additional_kwargs=additional_kwargs)

        return ChatResult(
            generations=[ChatGeneration(message=message)],
            llm_output={
                "token_usage": response.get("usage", {}),
                "model_name": self.model,
            },
        )

    def _generate(
        self, 
        messages: List[BaseMessage], 
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate with tools if provided."""
        if stop is not None:
            raise ValueError("Stop sequences are not yet supported for Groq.")
        
        tools = kwargs.pop("tools", None)
        
        if self.groq_api_key is None:
            raise ValueError("groq_api_key cannot be None")
            
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.groq_api_key.get_secret_value()}",
        }
        
        payload = self._convert_messages_to_groq_format(messages, tools)
        
        response = requests.post(GROQ_API_URL, headers=headers, json=payload)
        
        if response.status_code != 200:
            raise ValueError(
                f"Groq API returned error status code: {response.status_code}. "
                f"Response content: {response.text}"
            )
        
        return self._create_chat_result(response.json())

    def _stream(
        self, 
        messages: List[BaseMessage], 
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """Stream the response from the model."""
        if stop is not None:
            raise ValueError("Stop sequences are not yet supported for Groq.")
        
        tools = kwargs.pop("tools", None)
        
        if self.groq_api_key is None:
            raise ValueError("groq_api_key cannot be None")
            
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.groq_api_key.get_secret_value()}",
            "Accept": "text/event-stream",
        }
        
        payload = self._convert_messages_to_groq_format(messages, tools)
        payload["stream"] = True
        
        with requests.post(GROQ_API_URL, headers=headers, json=payload, stream=True) as response:
            if response.status_code != 200:
                raise ValueError(
                    f"Groq API returned error status code: {response.status_code}. "
                    f"Response content: {response.text}"
                )
            
            content_buffer = ""
            function_calls_buffer = []
            
            for line in response.iter_lines():
                if line:
                    line = line.decode("utf-8")
                    if line.startswith("data: "):
                        line = line[6:]  # Remove "data: " prefix
                        if line.strip() == "[DONE]":
                            break
                        
                        try:
                            chunk = json.loads(line)
                            delta = chunk["choices"][0]["delta"]
                            
                            if "content" in delta and delta["content"] is not None:
                                content = delta["content"]
                                content_buffer += content
                                # Create proper chunk with AIMessageChunk
                                message_chunk = AIMessageChunk(content=content)
                                yield ChatGenerationChunk(message=message_chunk)
                            
                            if "tool_calls" in delta:
                                # Process tool calls in the stream
                                for tool_call in delta["tool_calls"]:
                                    # Check if this is a new tool call or update to existing one
                                    tool_call_id = tool_call.get("id")
                                    
                                    # Find existing tool call or create a new one
                                    existing_call = next((
                                        c for c in function_calls_buffer 
                                        if c.get("id") == tool_call_id
                                    ), None)
                                    
                                    if not existing_call:
                                        # New tool call
                                        function_calls_buffer.append({
                                            "id": tool_call_id,
                                            "type": "function",
                                            "name": tool_call.get("function", {}).get("name", ""),
                                            "args": tool_call.get("function", {}).get("arguments", "")
                                        })
                                    else:
                                        # Update existing call
                                        if "function" in tool_call:
                                            if "name" in tool_call["function"]:
                                                existing_call["name"] += tool_call["function"]["name"]
                                            if "arguments" in tool_call["function"]:
                                                existing_call["args"] += tool_call["function"]["arguments"]
                                
                                # Yield a chunk for tool calls with the correct message type
                                tool_calls_kwargs = {"tool_calls": [
                                    {
                                        "id": tc.get("id", ""),
                                        "type": "function",
                                        "function": {
                                            "name": tc.get("name", ""),
                                            "arguments": tc.get("args", "{}")
                                        }
                                    }
                                    for tc in function_calls_buffer
                                ]}
                                tool_message_chunk = AIMessageChunk(content="", additional_kwargs=tool_calls_kwargs)
                                yield ChatGenerationChunk(message=tool_message_chunk)
                        
                        except (json.JSONDecodeError, KeyError) as e:
                            logger.error(f"Error parsing Groq streaming response: {e}, line: {line}")
                            continue

    def bind_tools(
        self, tools: Sequence[Union[Dict[str, Any], Type, Callable, BaseTool]], **kwargs: Any
    ) -> BaseChatModel:
        """Bind tools to the model.
        
        This makes the tools available to the model during generation.
        
        Args:
            tools: A list of tools to bind to the model.
            **kwargs: Additional parameters to pass to the model.
            
        Returns:
            A new instance of the model with the tools bound.
        """
        converted_tools = []
        for tool in tools:
            if isinstance(tool, dict):
                converted_tools.append(tool)
            elif isinstance(tool, type) and issubclass(tool, BaseModel):
                converted_tools.append(convert_to_openai_tool(tool))
            elif isinstance(tool, BaseTool):
                converted_tools.append(convert_to_openai_tool(tool))
            elif callable(tool):
                # Handle callable tools
                tool_name = getattr(tool, "__name__", "tool")
                tool_description = getattr(tool, "__doc__", "A callable tool")
                converted_tools.append({
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "description": tool_description,
                        "parameters": {
                            "type": "object",
                            "properties": {},
                            "required": []
                        }
                    }
                })
            else:
                # Try to convert unknown tool types
                try:
                    converted_tools.append(convert_to_openai_tool(tool))
                except Exception as e:
                    logger.warning(f"Could not convert tool {tool}: {e}")
                    continue
        
        def new_generate(
            messages: List[BaseMessage],
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
        ) -> ChatResult:
            return self._generate(
                messages=messages,
                stop=stop,
                run_manager=run_manager,
                tools=converted_tools,
                **kwargs,
            )
        
        def new_stream(
            messages: List[BaseMessage],
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
        ) -> Iterator[ChatGenerationChunk]:
            return self._stream(
                messages=messages,
                stop=stop,
                run_manager=run_manager,
                tools=converted_tools,
                **kwargs,
            )
        
        # Make a copy of the model to avoid modifying the original
        model_copy = cast(ChatGroq, self.copy())
        model_copy._generate = new_generate  # type: ignore
        model_copy._stream = new_stream  # type: ignore
        
        return model_copy

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Async generate chat completion."""
        # Convert AsyncCallbackManagerForLLMRun to CallbackManagerForLLMRun if needed
        sync_manager = None
        if run_manager:
            from langchain_core.callbacks.manager import CallbackManager
            handlers = getattr(run_manager, "handlers", [])
            tags = getattr(run_manager, "tags", [])
            metadata = getattr(run_manager, "metadata", {})
            run_id = getattr(run_manager, "run_id", None) or UUID(int=0)
            
            sync_manager = CallbackManagerForLLMRun(
                handlers=handlers,
                tags=tags,
                metadata=metadata,
                inheritable_handlers=[],
                parent_run_id=None,
                run_id=run_id,
            )
        
        # Run synchronous _generate in a thread pool
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self._generate(
                messages=messages,
                stop=stop,
                run_manager=sync_manager,
                **kwargs
            )
        )

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        """Async stream chat completion."""
        # Convert AsyncCallbackManagerForLLMRun to CallbackManagerForLLMRun if needed
        sync_manager = None
        if run_manager:
            from langchain_core.callbacks.manager import CallbackManager
            handlers = getattr(run_manager, "handlers", [])
            tags = getattr(run_manager, "tags", [])
            metadata = getattr(run_manager, "metadata", {})
            run_id = getattr(run_manager, "run_id", None) or UUID(int=0)
            
            sync_manager = CallbackManagerForLLMRun(
                handlers=handlers,
                tags=tags,
                metadata=metadata,
                inheritable_handlers=[],
                parent_run_id=None,
                run_id=run_id,
            )
            
        # Create async generator to yield chunks from synchronous iterator
        iterator = self._stream(
            messages=messages,
            stop=stop,
            run_manager=sync_manager,
            **kwargs
        )
        
        # Process iterator in a way that properly implements async iteration
        loop = asyncio.get_event_loop()
        for chunk in iterator:
            # Use yield to create an async generator
            yield chunk