from typing import Dict, List, Optional
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

class ChatGroq(BaseChatModel):
    api_key: str
    model: str
    
    @property
    def _llm_type(self) -> str:
        return "groq"

    def _invoke_tools(self, messages: List[BaseMessage], tools: List[dict], **kwargs) -> AIMessage:
        """Process tool invocations for Groq models."""
        try:
            from langchain_groq import ChatGroq as LangchainGroq
            groq = LangchainGroq(
                api_key=self.api_key,
                model=self.model
            )
            # Bind tools to model
            model_with_tools = groq.bind_tools(tools)
            result = model_with_tools.invoke(messages, **kwargs)
            return result
        except Exception as e:
            raise ValueError(f"Error invoking Groq tools: {str(e)}")

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Dict,
    ) -> str:
        """Generate a response from Groq."""
        tools = kwargs.pop("tools", None)
        if tools:
            # Use tool invocation path
            return self._invoke_tools(messages, tools, stop=stop, run_manager=run_manager, **kwargs)
        else:
            # Regular chat completion path
            try:
                from langchain_groq import ChatGroq as LangchainGroq
                groq = LangchainGroq(
                    api_key=self.api_key,
                    model=self.model
                )
                result = groq.invoke(messages, stop=stop, run_manager=run_manager, **kwargs)
                return result
            except Exception as e:
                raise ValueError(f"Error generating with Groq: {str(e)}")