from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any, AsyncIterator

class Model(ABC):
    """
    Abstract base class for language model engines.
    Defines interface for interacting with different LLM providers.
    """

    @abstractmethod
    async def call(self, messages: List[Dict[str, str]]) -> str:
        """Generate response from message history asynchronously."""
        pass
    
    @abstractmethod
    async def stream(self, messages: List[Dict[str, str]]) -> AsyncIterator[str]:
        """Stream response tokens from message history asynchronously."""
        pass

class OpenAIServerModel(Model):
    """
    OpenAI-compatible LLM engine implementation.
    Supports OpenAI API and compatible endpoints.
    """
    
    def __init__(
            self, 
            model_id: str,
            base_url: Optional[str] = None,
            api_key: Optional[str] = None,
            organization: Optional[str] = None,
            project: Optional[str] = None,
            **kwargs
        ):
        """Initialize OpenAI LLM engine.
        
        Args:
            model_id: Model identifier
            api_key: API authentication key
            base_url: Optional API endpoint URL
            organization: Optional organization ID
            project: Optional project ID
            **kwargs: Additional parameters to pass to the OpenAI API
        """
        try:
            import openai
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "Please install 'openai' extra to use OpenAIServerModel: `pip install 'cave_agent[openai]'`"
            )

        self.kwargs = kwargs
        self.model_id = model_id
        self.client = openai.AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
            organization=organization,
            project=project,
        )
    
    def _prepare_params(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """Prepare parameters for OpenAI API call."""
        params = {
            "model": self.model_id,
            "messages": messages,
            **self.kwargs,
        }
            
        return params

    async def call(self, messages: List[Dict[str, str]]) -> str:
        """Generate response using OpenAI API asynchronously."""
        response = await self.client.chat.completions.create(
            **self._prepare_params(messages),
            stream=False
        )
        if hasattr(response, "choices") and len(response.choices) > 0:
            return response.choices[0].message.content
    
    async def stream(self, messages: List[Dict[str, str]]) -> AsyncIterator[str]:
        """Stream response tokens using OpenAI API."""
        response = await self.client.chat.completions.create(
            **self._prepare_params(messages),
            stream=True
        )
        
        async for chunk in response:
            if hasattr(chunk, "choices") and len(chunk.choices) > 0:
                delta = chunk.choices[0].delta
                if delta.content:
                    yield delta.content

class LiteLLMModel(Model):
    """
    LiteLLM model implementation that provides a unified interface to hundreds of LLM providers.
    
    LiteLLM is a library that standardizes the API for different LLM providers, allowing you to
    easily switch between OpenAI, Anthropic, Google, Azure, and many other providers with a
    consistent interface. This model acts as a gateway to access any LLM supported by LiteLLM.
    
    See https://www.litellm.ai/ for more information about supported providers and models.
    """
    
    def __init__(
            self, 
            model_id: str,
            base_url: Optional[str] = None,
            api_key: Optional[str] = None,
            **kwargs
        ):
        """Initialize OpenAI LLM engine.
        
        Args:
            model_id: Model identifier
            api_key: API authentication key
            base_url: Optional API endpoint URL
            **kwargs: Additional parameters to pass to the API
        """
        try:
            import litellm
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "Please install 'litellm' extra to use LiteLLMModel: `pip install 'cave_agent[litellm]'`"
            )
        self.kwargs = kwargs
        self.model_id = model_id
        self.base_url = base_url
        self.api_key = api_key

    def _prepare_params(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """Prepare parameters for API call"""
        params = {
            "model": self.model_id,
            "api_base": self.base_url,
            "api_key": self.api_key,
            "messages": messages,
            **self.kwargs,
        }
        
        return params

    async def call(self, messages: List[Dict[str, str]]) -> str:
        """Generate response."""
        import litellm
        response = await litellm.acompletion(**self._prepare_params(messages), stream=False)

        if hasattr(response, "choices") and len(response.choices) > 0:
            return response.choices[0].message.content
    
    async def stream(self, messages: List[Dict[str, str]]) -> AsyncIterator[str]:
        """Stream response tokens"""
        import litellm
        response = await litellm.acompletion(**self._prepare_params(messages), stream=True)
        
        async for chunk in response:
            if hasattr(chunk, "choices") and len(chunk.choices) > 0:
                delta = chunk.choices[0].delta
                if delta.content:
                    yield delta.content