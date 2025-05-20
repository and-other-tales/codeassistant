"""Groq integration for LangChain with enhanced tool support.

This module provides improved Groq chat model integration for LangChain,
with proper support for tool calling and structured outputs.
"""

from __future__ import annotations

import json
import logging
import asyncio
import os
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Type, Union, cast, Callable, AsyncIterator, Coroutine
from uuid import UUID

import requests
from pydantic import BaseModel, Field, model_validator
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
    
    groq_api_key: Optional[str] = None
    """Groq API key."""
    
    streaming: bool = False
    """Whether to stream the response."""

    @model_validator(mode='after')
    def validate_environment(cls, values):
        """Validate that the API key is provided."""
        if values.groq_api_key is None or values.groq_api_key == "":
            try:
                groq_api_key = os.environ["GROQ_API_KEY"]
                values.groq_api_key = groq_api_key
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
            "Authorization": f"Bearer {self.groq_api_key}",
        }

        # Convert LangChain tools to OpenAI format
        openai_tools = None
        if tools:
            openai_tools = [convert_to_openai_tool(tool) for tool in tools]
        
        payload = self._convert_messages_to_groq_format(messages, tools=openai_tools)
        response = requests.post(GROQ_API_URL, headers=headers, json=payload)
        
        if response.status_code != 200:
            raise ValueError(
                f"Groq API returned error {response.status_code}: {response.text}"
            )
            
        response_json = response.json()
        return self._create_chat_result(response_json)

    def _stream(
        self, 
        messages: List[BaseMessage], 
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """Stream responses from Groq API."""
        if stop is not None:
            raise ValueError("Stop sequences are not yet supported for Groq.")
        
        tools = kwargs.pop("tools", None)
        
        if self.groq_api_key is None:
            raise ValueError("groq_api_key cannot be None")
            
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.groq_api_key}",
        }

        # Convert LangChain tools to OpenAI format
        openai_tools = None
        if tools:
            openai_tools = [convert_to_openai_tool(tool) for tool in tools]
        
        payload = self._convert_messages_to_groq_format(messages, tools=openai_tools)
        payload["stream"] = True  # Ensure streaming is enabled
        
        with requests.post(GROQ_API_URL, headers=headers, json=payload, stream=True) as response:
            if response.status_code != 200:
                raise ValueError(
                    f"Groq API returned error {response.status_code}: {response.text}"
                )

            # Process the streamed response
            content = ""
            tool_calls_buffer = []
            current_tool_call = None
            
            for line in response.iter_lines():
                if not line:
                    continue
                    
                # Remove 'data: ' prefix and skip ping/end lines
                line = line.decode("utf-8")
                if line == "data: [DONE]":
                    break
                if not line.startswith("data: "):
                    continue
                    
                data = json.loads(line[6:])  # Remove 'data: ' prefix and parse JSON
                
                delta = data.get("choices", [{}])[0].get("delta", {})
                finish_reason = data.get("choices", [{}])[0].get("finish_reason")
                
                chunk_content = delta.get("content", "")
                if chunk_content:
                    content += chunk_content
                    yield ChatGenerationChunk(message=AIMessageChunk(content=chunk_content))

                # Handle tool calls in the stream
                if "tool_calls" in delta:
                    for tool_call_delta in delta["tool_calls"]:
                        tool_call_index = tool_call_delta.get("index", 0)
                        
                        # Extend the tool_calls_buffer if needed
                        while len(tool_calls_buffer) <= tool_call_index:
                            tool_calls_buffer.append({
                                "id": "",
                                "type": "function",
                                "function": {"name": "", "arguments": ""}
                            })
                        
                        # Update the tool call at the index
                        if "id" in tool_call_delta:
                            tool_calls_buffer[tool_call_index]["id"] = tool_call_delta["id"]
                        
                        if "function" in tool_call_delta:
                            function_delta = tool_call_delta["function"]
                            if "name" in function_delta:
                                tool_calls_buffer[tool_call_index]["function"]["name"] = function_delta["name"]
                            if "arguments" in function_delta:
                                tool_calls_buffer[tool_call_index]["function"]["arguments"] += function_delta["arguments"]
                
                # Yield tool call chunks
                if tool_calls_buffer:
                    complete_tool_calls = []
                    for tc in tool_calls_buffer:
                        # Only include if we have at least an ID and name
                        if tc["id"] and tc["function"]["name"]:
                            try:
                                args = json.loads(tc["function"]["arguments"]) if tc["function"]["arguments"] else {}
                                complete_tool_calls.append({
                                    "id": tc["id"],
                                    "type": "function",
                                    "name": tc["function"]["name"],
                                    "args": args
                                })
                            except json.JSONDecodeError:
                                # The arguments might not be complete JSON yet
                                pass
                    
                    if complete_tool_calls:
                        yield ChatGenerationChunk(
                            message=AIMessageChunk(
                                content="", 
                                additional_kwargs={"tool_calls": complete_tool_calls}
                            )
                        )

    def bind_tools(
        self, tools: Sequence[Union[Dict[str, Any], Type, Callable, BaseTool]], **kwargs: Any
    ) -> BaseChatModel:
        """Bind tools to this chat model.

        Args:
            tools: A list of tools to bind to this chat model.
            **kwargs: Additional parameters to pass to the chat model.

        Returns:
            A new instance of this chat model with the tools bound.
        """
        # Convert tools to the format expected by the model
        openai_tools = [convert_to_openai_tool(tool) for tool in tools]
        
        # Create a copy of the current model
        new_model = self.copy()

        # Create new _generate and _stream methods with the tools
        def new_generate(
            messages: List[BaseMessage],
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs_inner: Any,
        ) -> ChatResult:
            """Call _generate with the bound tools."""
            merged_kwargs = {**kwargs, **kwargs_inner}
            return self._generate(
                messages, stop=stop, run_manager=run_manager, tools=openai_tools, **merged_kwargs
            )

        def new_stream(
            messages: List[BaseMessage],
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs_inner: Any,
        ) -> Iterator[ChatGenerationChunk]:
            """Call _stream with the bound tools."""
            merged_kwargs = {**kwargs, **kwargs_inner}
            return self._stream(
                messages, stop=stop, run_manager=run_manager, tools=openai_tools, **merged_kwargs
            )

        # Monkey patch the new model with the bound tools methods
        setattr(new_model, "_generate", new_generate)
        setattr(new_model, "_stream", new_stream)
        
        # Also patch the async methods to use the tools
        async def new_agenerate(
            messages: List[BaseMessage],
            stop: Optional[List[str]] = None,
            run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
            **kwargs_inner: Any,
        ) -> ChatResult:
            """Call _agenerate with the bound tools."""
            merged_kwargs = {**kwargs, **kwargs_inner}
            return await self._agenerate(
                messages, stop=stop, run_manager=run_manager, tools=openai_tools, **merged_kwargs
            )

        async def new_astream(
            messages: List[BaseMessage],
            stop: Optional[List[str]] = None,
            run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
            **kwargs_inner: Any,
        ) -> AsyncIterator[ChatGenerationChunk]:
            """Call _astream with the bound tools."""
            merged_kwargs = {**kwargs, **kwargs_inner}
            async for chunk in self._astream(
                messages, stop=stop, run_manager=run_manager, tools=openai_tools, **merged_kwargs
            ):
                yield chunk

        # Monkey patch the new model with the async bound tools methods
        setattr(new_model, "_agenerate", new_agenerate)
        setattr(new_model, "_astream", new_astream)
        
        return new_model

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Asynchronously generate a response."""
        if stop is not None:
            raise ValueError("Stop sequences are not yet supported for Groq.")
        
        tools = kwargs.pop("tools", None)
        
        if self.groq_api_key is None:
            raise ValueError("groq_api_key cannot be None")
        
        # Convert async arguments to synchronous ones
        # In this case, we pass None for run_manager since we can't convert it
        # Run in a thread to avoid blocking the event loop
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, lambda: self._generate(messages, None, None, tools=tools, **kwargs)
        )

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        """Asynchronously stream a response."""
        if stop is not None:
            raise ValueError("Stop sequences are not yet supported for Groq.")
        
        tools = kwargs.pop("tools", None)
        
        if self.groq_api_key is None:
            raise ValueError("groq_api_key cannot be None")
        
        # Create HTTP session and prepare request
        import aiohttp
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.groq_api_key}",
        }
        
        # Convert LangChain tools to OpenAI format
        openai_tools = None
        if tools:
            openai_tools = [convert_to_openai_tool(tool) for tool in tools]
        
        payload = self._convert_messages_to_groq_format(messages, tools=openai_tools)
        payload["stream"] = True  # Ensure streaming is enabled
        
        async with aiohttp.ClientSession() as session:
            async with session.post(GROQ_API_URL, headers=headers, json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise ValueError(
                        f"Groq API returned error {response.status}: {error_text}"
                    )
                
                # Process the streamed response
                content = ""
                tool_calls_buffer = []
                
                async for line in response.content:
                    line = line.decode("utf-8").strip()
                    if not line or not line.startswith("data: "):
                        continue
                    
                    if line == "data: [DONE]":
                        break
                        
                    data = json.loads(line[6:])  # Remove 'data: ' prefix and parse JSON
                    
                    delta = data.get("choices", [{}])[0].get("delta", {})
                    finish_reason = data.get("choices", [{}])[0].get("finish_reason")
                    
                    chunk_content = delta.get("content", "")
                    if chunk_content:
                        content += chunk_content
                        yield ChatGenerationChunk(message=AIMessageChunk(content=chunk_content))
    
                    # Handle tool calls in the stream
                    if "tool_calls" in delta:
                        for tool_call_delta in delta["tool_calls"]:
                            tool_call_index = tool_call_delta.get("index", 0)
                            
                            # Extend the tool_calls_buffer if needed
                            while len(tool_calls_buffer) <= tool_call_index:
                                tool_calls_buffer.append({
                                    "id": "",
                                    "type": "function",
                                    "function": {"name": "", "arguments": ""}
                                })
                            
                            # Update the tool call at the index
                            if "id" in tool_call_delta:
                                tool_calls_buffer[tool_call_index]["id"] = tool_call_delta["id"]
                            
                            if "function" in tool_call_delta:
                                function_delta = tool_call_delta["function"]
                                if "name" in function_delta:
                                    tool_calls_buffer[tool_call_index]["function"]["name"] = function_delta["name"]
                                if "arguments" in function_delta:
                                    tool_calls_buffer[tool_call_index]["function"]["arguments"] += function_delta["arguments"]
                    
                    # Yield tool call chunks
                    if tool_calls_buffer:
                        complete_tool_calls = []
                        for tc in tool_calls_buffer:
                            # Only include if we have at least an ID and name
                            if tc["id"] and tc["function"]["name"]:
                                try:
                                    args = json.loads(tc["function"]["arguments"]) if tc["function"]["arguments"] else {}
                                    complete_tool_calls.append({
                                        "id": tc["id"],
                                        "type": "function",
                                        "name": tc["function"]["name"],
                                        "args": args
                                    })
                                except json.JSONDecodeError:
                                    # The arguments might not be complete JSON yet
                                    pass
                        
                        if complete_tool_calls:
                            yield ChatGenerationChunk(
                                message=AIMessageChunk(
                                    content="", 
                                    additional_kwargs={"tool_calls": complete_tool_calls}
                                )
                            )