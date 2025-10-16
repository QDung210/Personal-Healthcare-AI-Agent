"""
Custom Pydantic AI Model wrapper for Amazon Bedrock imported models
"""

from pydantic_ai.models import Model, KnownModelName
from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    TextPart,
    UserPromptPart,
    SystemPromptPart,
)
from pydantic_ai.settings import ModelSettings
from pydantic_ai.usage import RunUsage
import boto3
import json
from typing import Any


class BedrockCustomModel(Model):
    """Custom Pydantic AI Model wrapper for Bedrock imported models"""
    
    def __init__(self, model_arn: str):
        self.model_arn = model_arn
        self._client = None  # Lazy initialization
        self._model_name = f"bedrock-custom-{model_arn.split('/')[-1]}"
    
    @property
    def client(self):
        """Lazy initialization of boto3 client to ensure credentials are set"""
        if self._client is None:
            import boto3
            self._client = boto3.client('bedrock-runtime', region_name='us-east-1')
        return self._client
    
    @property
    def model_name(self) -> KnownModelName | str:
        """Return model name"""
        return self._model_name
    
    @property 
    def system(self) -> bool:
        """Whether model supports system messages"""
        return True
    
    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None = None,
        model_request_parameters: dict[str, Any] | None = None,
    ) -> ModelResponse:
        """Make request to Bedrock custom model"""
        
        # Convert messages to prompt
        prompt = self._messages_to_prompt(messages)
        
        # Get settings
        max_tokens = getattr(model_settings, 'max_tokens', 512) if model_settings else 512
        temperature = getattr(model_settings, 'temperature', 0.7) if model_settings else 0.7
        
        # Call Bedrock asynchronously
        import asyncio
        import concurrent.futures
        
        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor() as pool:
            response = await loop.run_in_executor(
                pool, self._invoke_model, prompt, max_tokens, temperature
            )
        
        # Parse response
        result = json.loads(response['body'].read())
        text = result['choices'][0]['text'].strip()
        
        # Create RunUsage
        usage = RunUsage(
            input_tokens=len(prompt.split()),
            output_tokens=len(text.split()),
            requests=1,
            tool_calls=0,
        )
        
        # Create ModelResponse with usage
        model_response = ModelResponse(
            parts=[TextPart(content=text)],
            timestamp=None,
        )
        model_response.usage = usage
        
        return model_response
    
    def _invoke_model(self, prompt: str, max_tokens: int, temperature: float):
        """Synchronous invoke to Bedrock"""
        return self.client.invoke_model(
            modelId=self.model_arn,
            body=json.dumps({
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature
            })
        )
    
    def _messages_to_prompt(self, messages: list[ModelMessage]) -> str:
        """Convert Pydantic AI messages to Qwen chat template format"""
        prompt_parts = []
        
        for msg in messages:
            role = getattr(msg, 'role', 'user')
            
            if hasattr(msg, 'parts'):
                content_parts = []
                for part in msg.parts:
                    if isinstance(part, (TextPart, UserPromptPart, SystemPromptPart)):
                        content_parts.append(part.content)
                    elif hasattr(part, 'content'):
                        content_parts.append(str(part.content))
                
                if content_parts:
                    content = ' '.join(content_parts)
                    
                    # Format according to Qwen chat template
                    if role == 'system':
                        prompt_parts.append(f"<|im_start|>system\n{content}<|im_end|>")
                    elif role == 'user':
                        prompt_parts.append(f"<|im_start|>user\n{content}<|im_end|>")
                    elif role == 'assistant':
                        prompt_parts.append(f"<|im_start|>assistant\n{content}<|im_end|>")
        
        # Add assistant start token
        prompt_parts.append("<|im_start|>assistant\n")
        
        return '\n'.join(prompt_parts)
