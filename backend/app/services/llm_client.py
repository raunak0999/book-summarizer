"""
Single choke point for all model calls, so swapping providers (Azure
OpenAI <-> AWS Bedrock) never touches agent/business logic (requirement
#6.iii "modularity for future expansion").
"""
from app.core.config import get_settings
from app.core.logging import log

settings = get_settings()


_st_model = None


def _get_st_model():
    global _st_model
    if _st_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            _st_model = SentenceTransformer(settings.local_embedding_model)
        except ImportError:
            raise RuntimeError("sentence-transformers package is required for local embeddings. Set EMBEDDING_PROVIDER=gemini to use API embeddings.")
    return _st_model


import re
import time
from tenacity import retry, retry_if_exception, stop_after_attempt

def _is_rate_limit_error(exc: Exception) -> bool:
    exc_str = str(exc).lower()
    if "429" in exc_str or "rate limit" in exc_str or "too many requests" in exc_str or "resource_exhausted" in exc_str or "resourceexhausted" in exc_str:
        return True
    if "remote end closed connection" in exc_str or "connection aborted" in exc_str or "remotedisconnected" in exc_str or "protocolerror" in exc_str:
        return True
    if hasattr(exc, "status_code") and getattr(exc, "status_code") == 429:
        return True
    if hasattr(exc, "code") and getattr(exc, "code") == 429:
        return True
    return False

def _wait_rate_limit(retry_state) -> float:
    exc = retry_state.outcome.exception()
    if exc:
        msg = str(exc)
        # Parse patterns like "try again in 56m52.8s" or "Please retry in 13.8s"
        match = re.search(r"(?:try again|please retry) in (?:(\d+)m)?(\d+(?:\.\d+)?)s", msg, re.IGNORECASE)
        if match:
            minutes = float(match.group(1)) if match.group(1) else 0.0
            seconds = float(match.group(2))
            total_wait = (minutes * 60.0) + seconds + 1.0
            if total_wait > 120.0:
                log.warning("rate_limit_wait_too_long_aborting_retry", total_wait_s=total_wait)
                raise exc
            log.warning("rate_limit_429_retrying", parsed_wait_s=total_wait, attempt=retry_state.attempt_number)
            return total_wait
    attempt = retry_state.attempt_number
    default_wait = min(5.0 * attempt, 30.0)
    log.warning("transient_or_rate_limit_retrying", default_wait_s=default_wait, attempt=attempt)
    return default_wait






class LLMClient:
    def __init__(self):
        self.chat_provider = getattr(settings, "chat_provider", settings.llm_provider)
        self.embedding_provider = getattr(settings, "embedding_provider", settings.llm_provider)

        self._chat_client = self._init_client(self.chat_provider)
        self._embed_client = self._init_client(self.embedding_provider)

    def _init_client(self, provider: str):
        import httpx
        if provider == "azure_openai":
            from openai import AzureOpenAI
            return AzureOpenAI(
                api_key=settings.azure_openai_api_key,
                azure_endpoint=settings.azure_openai_endpoint,
                api_version=settings.azure_openai_api_version,
                http_client=httpx.Client(),
            )
        elif provider == "github_models":
            from openai import OpenAI
            return OpenAI(
                base_url=settings.github_models_endpoint,
                api_key=settings.github_models_token,
                http_client=httpx.Client(),
            )
        elif provider == "groq":
            from openai import OpenAI
            return OpenAI(
                base_url=settings.groq_endpoint,
                api_key=settings.groq_api_key or "placeholder_key",
                http_client=httpx.Client(),
            )
        elif provider == "gemini":
            from google import genai
            return genai.Client(api_key=settings.gemini_api_key or "placeholder_key")
        elif provider == "aws_bedrock":
            import boto3
            return boto3.client("bedrock-runtime", region_name=settings.aws_region)
        else:
            from openai import OpenAI
            return OpenAI(http_client=httpx.Client())

    # ---- Chat completion ----
    @retry(
        retry=retry_if_exception(_is_rate_limit_error),
        stop=stop_after_attempt(3),
        wait=_wait_rate_limit,
        reraise=True,
    )
    def chat(self, messages: list[dict], temperature: float = 0.2, max_tokens: int = 800) -> str:

        log.info("llm_call", provider=self.chat_provider, n_messages=len(messages))
        if self.chat_provider == "azure_openai":
            resp = self._chat_client.chat.completions.create(
                model=settings.azure_openai_deployment_chat,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content

        if self.chat_provider == "github_models":
            resp = self._chat_client.chat.completions.create(
                model=settings.github_models_chat_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content

        if self.chat_provider == "groq":
            try:
                resp = self._chat_client.chat.completions.create(
                    model=settings.groq_chat_model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                msg = resp.choices[0].message
                content = getattr(msg, "content", None) or getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None) or ""
                if content and content.strip():
                    return content.strip()
            except Exception as e:
                log.warning("groq_chat_failed_falling_back_to_gemini_immediately", error=str(e))
            
            # Immediate Fallback to Gemini with model rotation
            try:
                from google.genai import types
                gemini_client = self._init_client("gemini")
                system_instruction = next((m["content"] for m in messages if m["role"] == "system"), None)
                contents = []
                for m in messages:
                    if m["role"] == "system":
                        continue
                    role = "model" if m["role"] == "assistant" else m["role"]
                    contents.append(
                        types.Content(
                            role=role,
                            parts=[types.Part.from_text(text=m["content"])],
                        )
                    )
                config = types.GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                    system_instruction=system_instruction,
                )
                gemini_models = [
                    settings.gemini_chat_model,
                    "gemini-3.5-flash-lite",
                    "gemini-3.1-flash-lite",
                    "gemini-3.8-flash",
                    "gemini-3.7-flash",
                ]
                for model_name in gemini_models:
                    try:
                        resp = gemini_client.models.generate_content(
                            model=model_name, contents=contents, config=config,
                        )
                        text = resp.text
                        if text and text.strip():
                            return text
                    except Exception as me:
                        if "429" in str(me) or "RESOURCE_EXHAUSTED" in str(me):
                            log.warning("groq_fallback_gemini_rotating", model=model_name)
                            continue
                        raise me
            except Exception as ge:
                log.error("gemini_fallback_failed", error=str(ge))
                raise ge

        if self.chat_provider == "gemini":
            from google.genai import types
            system_instruction = next((m["content"] for m in messages if m["role"] == "system"), None)
            contents = []
            for m in messages:
                if m["role"] == "system":
                    continue
                role = "model" if m["role"] == "assistant" else m["role"]
                contents.append(
                    types.Content(
                        role=role,
                        parts=[types.Part.from_text(text=m["content"])],
                    )
                )
            config = types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
                system_instruction=system_instruction,
            )
            # Rotate across multiple Gemini models to avoid per-model RPD limits
            # Each free-tier model gets ~20 RPD; cycling gives ~100+ RPD total
            gemini_models = [
                settings.gemini_chat_model,  # primary: gemini-3.6-flash
                "gemini-3.5-flash-lite",
                "gemini-3.1-flash-lite",
                "gemini-3.8-flash",
                "gemini-3.7-flash",
            ]
            last_err = None
            for model_name in gemini_models:
                try:
                    resp = self._chat_client.models.generate_content(
                        model=model_name,
                        contents=contents,
                        config=config,
                    )
                    text = resp.text
                    if text and text.strip():
                        return text
                except Exception as e:
                    last_err = e
                    err_str = str(e)
                    if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                        log.warning("gemini_model_rate_limited_rotating", model=model_name, error=err_str[:100])
                        continue
                    else:
                        log.warning("gemini_model_error_rotating", model=model_name, error=err_str[:100])
                        continue
            # All models exhausted
            if last_err:
                raise last_err
            raise RuntimeError("All Gemini models exhausted")

        if self.chat_provider == "aws_bedrock":
            import json
            body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": [m for m in messages if m["role"] != "system"],
                "system": next((m["content"] for m in messages if m["role"] == "system"), ""),
            }
            resp = self._chat_client.invoke_model(
                modelId=settings.bedrock_model_id,
                body=json.dumps(body),
            )
            payload = json.loads(resp["body"].read())
            return payload["content"][0]["text"]

        resp = self._chat_client.chat.completions.create(
            model="gpt-4o-mini", messages=messages, temperature=temperature, max_tokens=max_tokens
        )
        return resp.choices[0].message.content

    # ---- Embeddings ----
    @retry(
        retry=retry_if_exception(_is_rate_limit_error),
        stop=stop_after_attempt(8),
        wait=_wait_rate_limit,
        reraise=True,
    )
    def embed(self, texts: list[str]) -> list[list[float]]:
        if self.embedding_provider == "local":
            model = _get_st_model()
            return model.encode(texts).tolist()

        if self.embedding_provider == "azure_openai":
            resp = self._embed_client.embeddings.create(
                model=settings.azure_openai_deployment_embedding, input=texts
            )
            return [d.embedding for d in resp.data]

        if self.embedding_provider == "github_models":
            resp = self._embed_client.embeddings.create(
                model=settings.github_models_embedding_model, input=texts
            )
            return [d.embedding for d in resp.data]

        if self.embedding_provider == "aws_bedrock":
            import json
            out = []
            for t in texts:
                resp = self._embed_client.invoke_model(
                    modelId="amazon.titan-embed-text-v2:0",
                    body=json.dumps({"inputText": t}),
                )
                out.append(json.loads(resp["body"].read())["embedding"])
            return out

        if self.embedding_provider in ("gemini", "api"):
            from google.genai import types
            config = types.EmbedContentConfig(output_dimensionality=settings.embedding_dim)
            res = self._embed_client.models.embed_content(
                model=settings.gemini_embedding_model,
                contents=texts,
                config=config,
            )
            return [e.values for e in res.embeddings]

        resp = self._embed_client.embeddings.create(model="text-embedding-3-small", input=texts)
        return [d.embedding for d in resp.data]




_llm_singleton: LLMClient | None = None


def get_llm_client() -> LLMClient:
    global _llm_singleton
    if _llm_singleton is None:
        _llm_singleton = LLMClient()
    return _llm_singleton
