import traceback
from typing import Any, Dict
import logging
import aiohttp

class LLM_analyzer:
    def __init__(self, system_prompt: str, gemini_api_key: str, open_router_api_key: str):
        self.system_prompt = system_prompt
        self.allowed_models = [
            "gemini-2.0-flash-lite",
            "deepseek/deepseek-chat-v3.1",
            "gemini-2.0-flash",
            "openai/gpt-oss-120b",
            "gemini-2.5-flash",
            "sarvamai/sarvam-m",
            "z-ai/glm-4.5-air", 
            "moonshotai/kimi-k2",
            "tngtech/deepseek-r1t2-chimera", 
            "deepseek/deepseek-r1-0528-qwen3-8b",  
            "tngtech/deepseek-r1t-chimera", 
            "microsoft/mai-ds-r1",
            "gemini-2.5-flash-lite",
            "gemini-2.5-pro",
            "venice/uncensored"
        ]
        self.max_retries = 5
        self.gemini_api_key = gemini_api_key
        self.open_router_api_key = open_router_api_key

    def _extract_text(self, json_str: Dict[str, Any]) -> str | None:
        try:
            return json_str["candidates"][0]["content"]["parts"][0]["text"]
        except Exception as e:
            # try with open-router response format
            try:
                print(json_str)
                return json_str['choices'][0]['message']['content']
            except Exception as e:
                logging.error(f"Error extracting text from response: {e}")
            return None
    
    async def _llm_call(
        self,
        user_prompt: str, model: str):
        try:
            if model.startswith("gemini"):
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
                headers = {
                    "Content-Type": "application/json",
                    "x-goog-api-key": self.gemini_api_key,
                }
                data = {
                    "systemInstruction": {
                        "parts": [
                            {"text": self.system_prompt}
                        ]
                    },
                    "contents": [
                        {
                            "parts": [
                                {"text": user_prompt}
                            ]
                        }
                    ],
                    "generationConfig": {
                        "responseMimeType": "text/x.enum",
                        "responseSchema": {
                        "type": "STRING",
                        "enum": ["APPROVE", "REJECT"]
                        }
                    }
                }
                response = await self._api_call(url, data, headers)
                return response
            else:
                # Using open-router for non-Gemini models
                url = "https://openrouter.ai/api/v1/chat/completions"
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.open_router_api_key}",
                }
                data = {
                    "model": model,
                    "messages": [
                        {
                            "role": "system",
                            "content": self.system_prompt + "\nRemember, your final output is ONLY `APPROVE` or `REJECT`.\n",
                        },
                        {
                            "role": "user",
                            "content": user_prompt,
                        }
                    ]
                }
                response = await self._api_call(url, data, headers)
                return response
        except Exception as e:
            logging.error(f"Error occurred at analyze_confession_with_llm: {e}")
            raise e

    async def _api_call(self, url: str, data: Dict[str, str], headers: Dict[str, str]) -> Dict[str, Any]:
        """
        Placeholder function simulating a call to an LLM API (like Gemini).

        Args:
            data (Dict[str, str]): The payload for the API request.
            url (str): The API endpoint URL.
            headers (Dict[str, str]): Headers for the API request.

        Returns:
            dict: Simulated response from the LLM API.
        """
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=data) as resp:
                    resp.raise_for_status()
                    res = await resp.json()
                    return res
        except Exception as e:
            logging.error(f"Error occurred while calling LLM API: {e}")
            raise e

    async def analyze_confession(self, user_prompt: str):
        """ Analyzes a confession using the specified LLM model."""
        i, max_retries = 0, self.max_retries
        while max_retries != i:
            try:
                res = await self._llm_call(
                    user_prompt=user_prompt,
                    model=self.allowed_models[i % len(self.allowed_models)],
                )
                return self._extract_text(res)
            except Exception as e:
                logging.error(f"Error occurred at analyze_confession: {e}")
                i += 1
                if max_retries == i:
                    raise e
    