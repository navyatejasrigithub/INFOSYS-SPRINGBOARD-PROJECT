# src/llm/llm_client.py

import os
import time
from dotenv import load_dotenv
import openai
from openai import OpenAI

load_dotenv()


class LLMClient:
    """
    Wrapper for calling LLM inside RAG pipeline
    """

    def __init__(self, model="gpt-4o-mini"):
        self.offline_mode = os.getenv("OFFLINE_MODE", "false").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

        if self.offline_mode:
            print("OFFLINE_MODE is enabled. OpenAI API calls are disabled.")
            self.client = None
            self.model = model
            self.fallback_models = []
            return

        api_key = os.getenv("OPENAI_API_KEY")

        if not api_key:
            print(
                "Warning: OPENAI_API_KEY not found. Set it in a .env file or environment."
            )
            self.client = None
        else:
            try:
                self.client = OpenAI(api_key=api_key)
            except Exception as e:
                print("Failed to initialize OpenAI client:", e)
                self.client = None

        # Primary model (can be overridden via env)
        env_model = os.getenv("OPENAI_MODEL")
        self.model = env_model or model

        # Optional comma-separated fallbacks: OPENAI_MODEL_FALLBACKS="gpt-4o,gpt-3.5-turbo"
        fallbacks = os.getenv("OPENAI_MODEL_FALLBACKS", "")
        if fallbacks:
            self.fallback_models = [m.strip() for m in fallbacks.split(",") if m.strip()]
        else:
            # sensible defaults to try if primary fails
            self.fallback_models = ["gpt-4o", "gpt-3.5-turbo"]

    def generate(self, prompt, temperature=0.2, max_tokens=800):
        """
        Generate response from LLM
        """

        if self.offline_mode:
            return "LLM generation failed: offline mode enabled."

        if self.client is None:
            print(
                "LLM client not initialized. Set `OPENAI_API_KEY` in .env or the environment."
            )
            return "LLM generation failed: missing API key."

        # Try primary model then fallbacks on recoverable errors (e.g., quota/rate limit)
        models_to_try = [self.model] + [m for m in self.fallback_models if m != self.model]
        last_error = None

        for idx, mdl in enumerate(models_to_try):
            try:
                response = self.client.chat.completions.create(
                    model=mdl,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a research assistant that answers using provided context only."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens
                )

                return response.choices[0].message.content

            except openai.OpenAIError as e:
                print(f"LLM error with model {mdl}:", e)
                last_error = e
                # If quota / rate limit, try next fallback
                status_code = getattr(e, 'status_code', None)
                err_code = getattr(e, 'code', None)
                if err_code == 'insufficient_quota':
                    print("Account has insufficient quota; skipping model fallbacks.")
                    break
                if status_code == 429:
                    print(f"Model {mdl} hit quota/rate limit; trying next model if available.")
                    # small backoff before retrying another model
                    time.sleep(1 + idx)
                    continue
                # For authentication or other unrecoverable errors, stop
                if status_code == 401:
                    print("Authentication error: check OPENAI_API_KEY.")
                    break
                # otherwise, try next model once
                continue

            except Exception as e:
                print(f"Unexpected LLM error with model {mdl}:", e)
                last_error = e
                continue

        print("All configured models failed or quota exceeded.")
        if last_error:
            return f"LLM generation failed: {last_error}"
        return "LLM generation failed."
