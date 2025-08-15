from typing import Optional

import httpx


class LLMService:
    def __init__(
            self,
            base_url: str = "http://localhost:11434",
            model: str = "llama2"):
        self.base_url = base_url
        self.model = model
        self.timeout = 30.0

    async def is_available(self) -> bool:
        """Check if LLM service is available."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                return response.status_code == 200
        except Exception as e:
            print(f"LLM service not available: {e}")
            return False

    async def generate_response(self, prompt: str) -> Optional[str]:
        """
        Generate response from LLM for the given prompt.
        :param prompt: The input prompt for LLM
        :return: Generated response or None if failed
        """
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                payload = {
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7
                    }
                }

                response = await client.post(
                    f"{self.base_url}/api/generate",
                    json=payload
                )

                if response.status_code != 200:
                    print(f"LLM API error: {response.status_code} - {response.text}")
                    return None

                result = response.json()
                return result.get("response", "").strip();
        except TimeoutError:
            print("LLM request timeout")
            return None
        except Exception as e:
            print(f"Error calling LLM: {e}")
            return None
