from typing import (
    Callable,
    List,
)
import openai
import tiktoken
from llm.basellm import BaseLLM
from retry import retry

import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

class Gemini(BaseLLM):
    """Wrapper around Gemini large language models."""

    def __init__(
        self,
        google_api_key: str = 'AIzaSyAnT0-DpdDE63wJpH51BT3GiB1n8e_tFNo',
        model_name: str = "gemini-pro",
        max_tokens: int = 1000,
        temperature: float = 0.0,
    ) -> None:
        genai.configure(api_key=google_api_key)
        self.model_name = model_name
        self.model=genai.GenerativeModel(
            model_name=model_name,
            generation_config=genai.types.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens
                ),
            safety_settings={
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }
        )

    @retry(tries=3, delay=1)
    def generate(
        self,
        messages: str,
    ) -> str:
        
        completions = self.model.generate_content(
            messages
        )
        
        return completions.text
    
    async def generateStreaming(
        self,
        messages: List[str],
        onTokenCallback=Callable[[str], None],
    ) -> str:
        result = []
        completions = openai.ChatCompletion.create(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            messages=messages,
            stream=True,
        )
        result = []
        for message in completions:
            # Process the streamed messages or perform any other desired action
            delta = message["choices"][0]["delta"]
            if "content" in delta:
                result.append(delta["content"])
            await onTokenCallback(message)
        return result

    def num_tokens_from_string(self, string: str) -> int:
        encoding = tiktoken.encoding_for_model(self.model_name)
        num_tokens = len(encoding.encode(string))
        return num_tokens

    def max_allowed_token_length(self) -> int:
        # TODO: list all models and their max tokens from api
        return 8193
