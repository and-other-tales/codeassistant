from typing import Any, Dict, List, Optional, Union, cast, Callable, Sequence, Iterator, Mapping, AsyncIterator, Type
from langchain_core.callbacks.base import Callbacks, BaseCallbackManager, BaseCallbackHandler
from langchain_core.callbacks.manager import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, BaseMessageChunk, AIMessageChunk, HumanMessage, SystemMessage, ToolMessage
from langchain_core.pydantic_v1 import root_validator, SecretStr, Field, BaseModel
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult, Generation
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, ToolException
import asyncio
import os
import json
from uuid import UUID
from copy import deepcopy
from typing_extensions import TypedDict

# Define a type that matches RunnableConfig structure for type checking
class ConfigDict(TypedDict, total=False):
    callbacks: Any
    tags: List[str]
    metadata: Dict[str, Any]
    run_name: str
    configurable: Dict[str, Any]

class ChatGroq(BaseChatModel):
    """Chat model implementation for Groq.
    
    This implementation provides support for the Groq API with 
    proper tool calling capabilities.
    """
    
    api_key: Optional[SecretStr] = None
    model: str = Field(..., description="The name of the Groq model to use")
    temperature: float = Field(0.7, description="Sampling temperature to use")
    top_p: float = Field(0.7, description="The top-p sampling parameter")
    max_tokens: Optional[int] = Field(None, description="Maximum number of tokens to generate")
    
    @root_validator(pre=True)
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key exists in environment."""
        api_key = values.get("api_key") or os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "Either `api_key` parameter or `GROQ_API_KEY` environment variable must be set"
            )
        values["api_key"] = SecretStr(api_key) if isinstance(api_key, str) else api_key
        
        # Validate model
        if not values.get("model"):
            values["model"] = os.environ.get("GROQ_MODEL", "llama3-8b-8192")
        
        return values
    
    @property
    def _llm_type(self) -> str:
        """Return the type of LLM."""
        return "groq"
        
    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get the identifying parameters."""
        return {
            "model": self.model,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
        }
        
    def _invoke_tools(
        self, 
        messages: List[BaseMessage], 
        tools: List[dict], 
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs
    ) -> AIMessage:
        """Process tool invocations for Groq models."""
        try:
            # Import at runtime to avoid dependency issues
            from langchain_groq import ChatGroq as LangchainGroq
            
            # Collect parameters
            params = {
                "temperature": kwargs.get("temperature", self.temperature),
                "top_p": kwargs.get("top_p", self.top_p),
                "max_tokens": kwargs.get("max_tokens", self.max_tokens)
            }
            
            # Convert SecretStr to string for the API
            api_key_value = self.api_key.get_secret_value() if self.api_key else None
            
            # Pass api_key_value directly - langchain_groq will handle SecretStr conversion
            groq = LangchainGroq(
                api_key=api_key_value,
                model=self.model,
                **{k: v for k, v in params.items() if v is not None}
            )
            
            # Create a proper callbacks configuration using BaseCallbackManager if available
            from langchain_core.callbacks.manager import CallbackManager
            callbacks = None
            if run_manager:
                handlers = getattr(run_manager, "handlers", [])
                callbacks = CallbackManager(handlers=handlers)
            
            config = RunnableConfig(callbacks=callbacks)
            
            # Bind tools to model
            model_with_tools = groq.bind_tools(tools)
            result = model_with_tools.invoke(messages, config=config)
            
            # Ensure the result is an AIMessage
            if isinstance(result, AIMessage):
                return result
            else:
                return AIMessage(content=str(result))
        except Exception as e:
            raise ValueError(f"Error invoking Groq tools: {str(e)}")
            
    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs
    ) -> Iterator[ChatGenerationChunk]:
        """Stream the response for the given messages."""
        try:
            # Import required modules
            from langchain_groq import ChatGroq as LangchainGroq
            from langchain_core.callbacks.manager import CallbackManager
            
            # Extract parameters
            params = {
                "temperature": kwargs.get("temperature", self.temperature),
                "top_p": kwargs.get("top_p", self.top_p),
                "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            }
            
            # Convert SecretStr to string
            api_key_value = self.api_key.get_secret_value() if self.api_key else None
            
            # Create the model - pass api_key_value directly
            groq = LangchainGroq(
                api_key=api_key_value,
                model=self.model,
                streaming=True,
                **{k: v for k, v in params.items() if v is not None}
            )
            
            # Create a proper callbacks configuration using CallbackManager
            callbacks = None
            if run_manager:
                handlers = getattr(run_manager, "handlers", [])
                callbacks = CallbackManager(handlers=handlers)
            
            config = RunnableConfig(callbacks=callbacks)
            
            # Handle tools if present
            tools = kwargs.get("tools")
            if tools:
                model = groq.bind_tools(tools)
            else:
                model = groq
            
            # Stream the response and handle BaseMessageChunk compatibility
            for chunk in model.stream(messages, stop=stop, config=config):
                if isinstance(chunk, AIMessage):
                    # Create a MessageChunk compatible with ChatGenerationChunk
                    from langchain_core.messages import AIMessageChunk
                    content = chunk.content if isinstance(chunk.content, str) else str(chunk.content)
                    message = AIMessageChunk(content=content)
                    yield ChatGenerationChunk(message=message)
                else:
                    from langchain_core.messages import AIMessageChunk
                    message = AIMessageChunk(content=str(chunk))
                    yield ChatGenerationChunk(message=message)
                    
        except Exception as e:
            raise ValueError(f"Error streaming with Groq: {str(e)}")
    
    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs
    ) -> AsyncIterator[ChatGenerationChunk]:
        """Asynchronously stream the response for the given messages."""
        from langchain_core.outputs import ChatGenerationChunk
        
        # Create a new callback manager compatible with the synchronous _stream method
        sync_manager = None
        if run_manager:
            handlers = getattr(run_manager, "handlers", [])
            # Create with proper arguments
            sync_manager = CallbackManagerForLLMRun(
                handlers=handlers,
                inheritable_handlers=[],
                parent_run_id=None, 
                tags=getattr(run_manager, "tags", []),
                metadata=getattr(run_manager, "metadata", {}),
                run_id=run_manager.run_id
            )
        
        # Use sync streaming in an executor
        for chunk in self._stream(
            messages=messages,
            stop=stop,
            run_manager=sync_manager,
            **kwargs
        ):
            yield chunk

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs
    ) -> ChatResult:
        """Asynchronously generate a response from Groq."""
        
        # Create a new callback manager compatible with the synchronous _generate method
        sync_manager = None
        if run_manager:
            handlers = getattr(run_manager, "handlers", [])
            # Create with proper arguments
            sync_manager = CallbackManagerForLLMRun(
                handlers=handlers,
                inheritable_handlers=[],
                parent_run_id=None, 
                tags=getattr(run_manager, "tags", []),
                metadata=getattr(run_manager, "metadata", {}),
                run_id=run_manager.run_id
            )
            
        return await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self._generate(
                messages=messages,
                stop=stop,
                run_manager=sync_manager,
                **kwargs
            )
        )

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs
    ) -> ChatResult:
        """Generate a response from Groq."""
        tools = kwargs.pop("tools", None)
        if tools and isinstance(tools, list):
            # Use tool invocation path
            try:
                result = self._invoke_tools(messages, tools, run_manager=run_manager, **kwargs)
                # Convert to ChatResult for compatibility
                return ChatResult(generations=[ChatGeneration(message=result)])
            except Exception as e:
                raise ValueError(f"Error invoking Groq tools: {str(e)}")
        else:
            # Regular chat completion path
            try:
                from langchain_groq import ChatGroq as LangchainGroq
                from langchain_core.callbacks.manager import CallbackManager
                
                # Collect parameters
                params = {
                    "temperature": kwargs.get("temperature", self.temperature),
                    "top_p": kwargs.get("top_p", self.top_p),
                    "max_tokens": kwargs.get("max_tokens", self.max_tokens)
                }
                
                # Convert SecretStr to string for the API
                api_key_value = self.api_key.get_secret_value() if self.api_key else None
                
                # Pass api_key_value directly
                groq = LangchainGroq(
                    api_key=api_key_value,
                    model=self.model,
                    **{k: v for k, v in params.items() if v is not None}
                )
                
                # Create a proper callbacks configuration using CallbackManager
                callbacks = None
                if run_manager:
                    handlers = getattr(run_manager, "handlers", [])
                    callbacks = CallbackManager(handlers=handlers)
                
                config = RunnableConfig(callbacks=callbacks)
                result = groq.invoke(messages, config=config, stop=stop)
                
                # Convert to ChatResult for compatibility
                if isinstance(result, AIMessage):
                    return ChatResult(generations=[ChatGeneration(message=result)])
                else:
                    return ChatResult(generations=[ChatGeneration(message=AIMessage(content=str(result)))])
                    
            except Exception as e:
                raise ValueError(f"Error generating with Groq: {str(e)}")
    
    def bind_tools(
        self, 
        tools: Sequence[Union[Dict[str, Any], Type, Callable, BaseTool]],
        *,
        tool_choice: Optional[Union[str, Dict[str, str]]] = None,
        **kwargs
    ) -> BaseChatModel:
        """Bind tools to the model."""
        from langchain_core.tools import BaseTool
        
        # Process tools to ensure they're in the right format
        processed_tools = []
        for tool in tools:
            if isinstance(tool, dict):
                # Assume the dictionary is already in the right format
                processed_tools.append(tool)
            elif isinstance(tool, BaseTool):
                # Create a tool dictionary with required fields
                tool_dict = {}
                # Set name and description
                tool_dict["name"] = tool.name
                tool_dict["description"] = tool.description
                
                # Handle the args_schema carefully
                if hasattr(tool, "args_schema") and tool.args_schema is not None:
                    args_schema = tool.args_schema
                    # Check if it has a schema method
                    if hasattr(args_schema, "schema") and callable(getattr(args_schema, "schema")):
                        # Convert schema to a dictionary - handle this with try/except
                        try:
                            schema_dict = args_schema.schema()
                            tool_dict["parameters"] = schema_dict
                        except Exception:
                            # Fallback if schema() method fails or doesn't exist
                            try:
                                # Try to convert to a dict using __dict__
                                tool_dict["parameters"] = args_schema.__dict__
                            except AttributeError:
                                # Final fallback
                                tool_dict["parameters"] = {}
                    elif isinstance(args_schema, dict):
                        # If it's already a dict, use it directly
                        tool_dict["parameters"] = args_schema
                    else:
                        # Try to convert to a dict using __dict__
                        try:
                            tool_dict["parameters"] = args_schema.__dict__
                        except AttributeError:
                            # Fallback
                            tool_dict["parameters"] = {}
                            
                processed_tools.append(tool_dict)
            elif callable(tool):
                # Convert callable to a dictionary (best effort)
                import inspect
                signature = inspect.signature(tool)
                name = tool.__name__
                description = tool.__doc__ or f"Function {name}"
                parameters = {}
                for param_name, param in signature.parameters.items():
                    parameters[param_name] = {
                        "type": "string",
                        "description": f"Parameter {param_name}"
                    }
                
                tool_dict = {
                    "name": name,
                    "description": description,
                    "parameters": {
                        "type": "object",
                        "properties": parameters,
                        "required": list(parameters.keys())
                    }
                }
                processed_tools.append(tool_dict)
        
        # Create a new instance with the tools bound
        chat_model = deepcopy(self)
        
        # This gets passed to _generate via **kwargs
        def _generate_with_tools(
            self,
            messages: List[BaseMessage],
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs
        ) -> ChatResult:
            return self._generate(
                messages=messages,
                stop=stop,
                run_manager=run_manager,
                tools=processed_tools,
                tool_choice=tool_choice,
                **kwargs
            )
        
        # Override _generate with our wrapped version
        chat_model._generate = _generate_with_tools.__get__(chat_model)
        return chat_model