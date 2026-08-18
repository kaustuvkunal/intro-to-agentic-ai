"""
Provider-agnostic LLM client.

Supported providers
-------------------
  openai            -> OpenAI Responses API
  openai-chat       -> OpenAI Chat Completions API
  ollama            -> OpenAI Chat Completions via Ollama
  groq              -> OpenAI Chat Completions via Groq
  gemini            -> OpenAI Chat Completions via Gemini
  anthropic         -> Anthropic Messages API

Usage
-----
    from mls_agents.config.llm_client import get_langchain_llm

    llm = get_langchain_llm()
    llm = get_langchain_llm(provider="groq", model="llama-3.3-70b-versatile")
"""

from __future__ import annotations

import base64
import io
from typing import Any

from src.config.settings import settings

# -- Provider -> default base_url --
PROVIDER_BASE_URLS: dict[str, str] = {
    "ollama": settings.ollama_base_url,
    "openai": settings.openai_base_url,
    "groq": settings.groq_base_url,
    "gemini": settings.gemini_base_url,
}

# -- Provider -> default API key env-var name --
PROVIDER_KEY_ENV: dict[str, str | None] = {
    "ollama": None,
    "openai": "OPENAI_API_KEY",
    "groq": "GROQ_API_KEY",
    "gemini": "GEMINI_API_KEY",
}

Message = dict[str, str]


# ---------------------------------------------------------------------------
# 1. Raw OpenAI / Anthropic client factories
# ---------------------------------------------------------------------------


def get_client_for_provider(
    provider: str,
    api_key: str | None = None,
    base_url: str | None = None,
) -> Any:
    """Return a raw SDK client for the given provider.

    Parameters
    ----------
    provider : str
        One of ``openai``, ``openai-chat``, ``ollama``, ``groq``,
        ``gemini``, ``anthropic``.
    api_key : str, optional
        Override the key resolved from settings / env.
    base_url : str, optional
        Override the base URL resolved from settings / env.

    Returns
    -------
    object
        An ``openai.OpenAI`` or ``anthropic.Anthropic`` client.
    """
    import os

    from openai import OpenAI

    provider = (provider or settings.llm_provider).lower()

    # -- Anthropic - separate SDK --
    if provider == "anthropic":
        from anthropic import Anthropic

        key = (
            api_key
            or settings.anthropic_api_key
            or os.getenv("ANTHROPIC_API_KEY")
        )
        if not key:
            raise ValueError("API key required for Anthropic provider")
        return Anthropic(api_key=key)

    # -- All OpenAI-compatible providers --
    if provider == "openai-chat":
        provider = "openai"

    url = (base_url or "").strip() or PROVIDER_BASE_URLS.get(provider)
    if not url:
        raise ValueError(f"No base_url configured for provider: {provider}")

    if provider == "ollama":
        key = api_key or "ollama"
    else:
        key = api_key or settings.openai_api_key
        if not key:
            raise ValueError(f"API key required for provider: {provider}")

    return OpenAI(base_url=url, api_key=key)


def get_openai_client(
    *,
    base_url: str | None = None,
    api_key: str | None = None,
) -> Any:
    """Return a raw OpenAI-compatible client (default provider)."""
    from openai import OpenAI

    url = base_url or PROVIDER_BASE_URLS.get(
        settings.llm_provider, settings.openai_base_url
    )
    key = api_key or settings.openai_api_key or "ollama"
    return OpenAI(base_url=url, api_key=key)


# ---------------------------------------------------------------------------
# 2. LangChain-compatible wrapper (used by LangGraph nodes)
# ---------------------------------------------------------------------------


def get_langchain_llm(
    provider: str | None = None,
    model: str | None = None,
    temperature: float | None = None,
) -> Any:
    """Return a LangChain chat model for the selected provider.

    The provider is resolved from the ``provider`` argument, falling
    back to ``settings.llm_provider``.

    Parameters
    ----------
    provider : str, optional
        Override the provider from settings.
    model : str, optional
        Override the model from settings.
    temperature : float, optional
        Override the temperature from settings.

    Returns
    -------
    BaseChatModel
        A LangChain chat model ready for ``prompt | llm`` chains.
    """
    provider = (provider or settings.llm_provider).lower()
    model = model or settings.llm_model
    temperature = (
        temperature
        if temperature is not None
        else settings.llm_temperature
    )

    # -- Anthropic needs its own LangChain class --
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=model,
            temperature=temperature,
            api_key=settings.anthropic_api_key,
        )

    # -- Everything else is OpenAI-compatible --
    from langchain_openai import ChatOpenAI

    base_url = PROVIDER_BASE_URLS.get(provider, settings.openai_base_url)
    api_key = "ollama" if provider == "ollama" else (
        settings.openai_api_key or "sk-placeholder"
    )

    return ChatOpenAI(
        model=model,
        base_url=base_url,
        api_key=api_key,
        temperature=temperature,
    )


# ---------------------------------------------------------------------------
# 3. Raw response helpers (non-LangChain usage)
# ---------------------------------------------------------------------------


def get_openai_response(
    messages: list[Message] | None = None,
    *,
    model: str | None = None,
    client: Any = None,
    instructions: str | None = None,
    input: str | None = None,
    reasoning_effort: str | None = None,
    use_responses_api: bool = True,
    include: list[str] | None = None,
    **kwargs: Any,
) -> str:
    """Generate a text response via the raw OpenAI/Anthropic SDK.

    Tries the Responses API first, then falls back to Chat Completions.
    """
    provider = settings.llm_provider.lower()
    model = model or settings.llm_model
    client = client or get_client_for_provider(provider)

    # -- Anthropic branch --
    if provider == "anthropic":
        answer, _ = get_openai_response_with_trace(
            input=input or "",
            instructions=instructions,
            client=client,
            model=model,
            provider=provider,
        )
        return answer

    # -- Gemini has no Responses API --
    if provider == "gemini":
        use_responses_api = False

    # -- Responses API --
    if use_responses_api:
        if not input and not messages:
            raise ValueError(
                "Either 'input' or 'messages' must be provided"
            )

        create_kwargs: dict[str, Any] = {
            "model": model,
            "input": input or (
                messages[0]["content"] if messages else ""
            ),
        }
        if instructions:
            create_kwargs["instructions"] = instructions
        if reasoning_effort:
            create_kwargs["reasoning"] = {"effort": reasoning_effort}
        if include is not None:
            create_kwargs["include"] = include
        create_kwargs.update(kwargs)

        try:
            response = client.responses.create(**create_kwargs)
            return response.output_text
        except Exception:
            pass  # fall through to Chat Completions

    # -- Chat Completions fallback --
    if not messages:
        raise ValueError("'messages' required for Chat Completions")

    final_messages = list(messages)
    if instructions:
        final_messages = [
            {"role": "system", "content": instructions},
            *final_messages,
        ]

    response = client.chat.completions.create(
        model=model,
        messages=final_messages,
        **kwargs,
    )
    return response.choices[0].message.content or ""


def get_reasoning_trace_from_response(response: Any) -> str:
    """Extract reasoning text from a Responses-API response object."""
    if response is None:
        return ""

    parts: list[str] = []
    output = getattr(response, "output", None) or []

    for item in output:
        item_type = getattr(item, "type", None) or (
            item.get("type") if isinstance(item, dict) else None
        )
        if item_type != "reasoning":
            continue

        # OpenAI-style summary
        summary = getattr(item, "summary", None)
        if summary is None and isinstance(item, dict):
            summary = item.get("summary")
        if summary:
            for s in summary:
                text = getattr(s, "text", None) or (
                    s.get("text") if isinstance(s, dict) else None
                )
                if text and str(text).strip():
                    parts.append(str(text).strip())

        # Groq-style: content[].type == "reasoning_text"
        content = getattr(item, "content", None)
        if content is None and isinstance(item, dict):
            content = item.get("content")
        if content and isinstance(content, list):
            for chunk in content:
                ctype = getattr(chunk, "type", None) or (
                    chunk.get("type")
                    if isinstance(chunk, dict)
                    else None
                )
                if ctype == "reasoning_text":
                    text = getattr(chunk, "text", None) or (
                        chunk.get("text")
                        if isinstance(chunk, dict)
                        else None
                    )
                    if text and str(text).strip():
                        parts.append(str(text).strip())
        elif content and isinstance(content, str) and content.strip():
            parts.append(content.strip())

    return "\n\n".join(parts).strip()


def _anthropic_generate(
    client: Any,
    *,
    model: str,
    input: str,
    instructions: str | None,
) -> tuple[str, str]:
    """Anthropic Messages API -> (answer_text, reasoning_trace).

    Content blocks may include:
        - type == "thinking" -> .thinking
        - type == "text"     -> .text
    """
    kwargs: dict[str, Any] = {
        "model": model,
        "max_tokens": 4096,
        "messages": [{"role": "user", "content": input}],
    }
    if instructions:
        kwargs["system"] = instructions

    msg = client.messages.create(**kwargs)

    text_parts: list[str] = []
    think_parts: list[str] = []

    for block in msg.content or []:
        btype = getattr(block, "type", None)
        if btype == "thinking":
            t = getattr(block, "thinking", None)
            if t and str(t).strip():
                think_parts.append(str(t).strip())
        elif btype == "text":
            t = getattr(block, "text", None)
            if t and str(t).strip():
                text_parts.append(str(t).strip())
        else:
            t = getattr(block, "text", None)
            if t and str(t).strip():
                text_parts.append(str(t).strip())

    answer = "\n".join(text_parts).strip()
    trace = "\n\n".join(think_parts).strip()
    return answer, trace


def get_openai_response_with_trace(
    *,
    input: str,
    instructions: str | None = None,
    client: Any = None,
    model: str | None = None,
    provider: str | None = None,
    use_responses_api: bool = True,
    reasoning_effort: str | None = None,
) -> tuple[str, str]:
    """Return ``(answer_text, reasoning_trace)`` for any provider."""
    provider = (provider or settings.llm_provider).lower()
    model = model or settings.llm_model
    client = client or get_client_for_provider(provider)

    # -- Anthropic --
    if provider == "anthropic":
        return _anthropic_generate(
            client,
            model=model,
            input=input,
            instructions=instructions,
        )

    # -- Gemini has no Responses API --
    if provider == "gemini":
        use_responses_api = False

    # -- Responses API --
    if use_responses_api:
        kwargs: dict[str, Any] = {"model": model, "input": input}
        if instructions:
            kwargs["instructions"] = instructions
        effort = reasoning_effort or settings.reasoning_effort
        if effort:
            kwargs["reasoning"] = {"effort": effort}
            kwargs["include"] = ["reasoning.encrypted_content"]

        try:
            response = client.responses.create(**create_kwargs)
            answer = getattr(response, "output_text", None) or ""
            trace = get_reasoning_trace_from_response(response)
            return answer, trace
        except Exception:
            pass  # fall through to Chat Completions

    # -- Chat Completions --
    messages: list[dict[str, str]] = []
    if instructions:
        messages.append({"role": "system", "content": instructions})
    messages.append({"role": "user", "content": input})

    resp = client.chat.completions.create(model=model, messages=messages)
    return (resp.choices[0].message.content or ""), ""


def describe_image_with_vision(
    image: Any, model: str | None = None
) -> str:
    """Send a PIL image to a vision-capable model via OpenAI-compat API."""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    client = get_openai_client()
    model = model or settings.vision_model

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Describe this page in detail. Focus on any "
                            "charts, graphs, figures, diagrams, and key "
                            "visual information. Be concise but complete."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{b64}"
                        },
                    },
                ],
            }
        ],
        max_tokens=500,
    )
    return response.choices[0].message.content or ""
