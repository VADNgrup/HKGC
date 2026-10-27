"""
Ollama LLM Interface Module
==========================

This module provides interfaces for interacting with Ollama's language models,
including text generation and embedding capabilities.

Author: Lightrag team
Created: 2024-01-24
License: MIT License

Copyright (c) 2024 Lightrag

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

Version: 1.0.0

Change Log:
- 1.0.0 (2024-01-24): Initial release
    * Added async chat completion support
    * Added embedding generation
    * Added stream response capability

Dependencies:
    - ollama
    - numpy
    - pipmaster
    - Python >= 3.10

Usage:
    from llm_interfaces.ollama_interface import ollama_model_complete, ollama_embed
"""

__version__ = "1.0.0"
__author__ = "lightrag Team"
__status__ = "Production"

import os
import sys
from dotenv import load_dotenv

if sys.version_info < (3, 9):
    from typing import AsyncIterator
else:
    from collections.abc import AsyncIterator
import pipmaster as pm  # Pipmaster for dynamic library install

load_dotenv(dotenv_path=os.getenv("MINIRAG_ENV", ".env"), override=False)

# install specific modules
if not pm.is_installed("ollama"):
    pm.install("ollama")
if not pm.is_installed("tenacity"):
    pm.install("tenacity")

import httpx
import ollama
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from minirag.exceptions import (
    APIConnectionError,
    RateLimitError,
    APITimeoutError,
)
import numpy as np
from typing import Union


def _get_env_float(key: str, default: float | None = None) -> float | None:
    value = os.getenv(key)
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _get_env_bool(key: str, default: bool = False) -> bool:
    value = os.getenv(key)
    if value is None or value == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _coerce_bool(value, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _normalize_model_name(model: str | None) -> str | None:
    if not model:
        return model
    if model.startswith("ollama/"):
        return model[len("ollama/") :]
    return model


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type(
        (RateLimitError, APIConnectionError, APITimeoutError)
    ),
)
async def ollama_model_if_cache(
    model,
    prompt,
    system_prompt=None,
    history_messages=[],
    **kwargs,
) -> Union[str, AsyncIterator[str]]:
    # print("debug: calling ollama with model:", model)
    # print("debug: calling ollama with kwargs:", {k: v for k, v in kwargs.items() if k != "host"})
    
    model = _normalize_model_name(model)
    stream = True if kwargs.get("stream") else False
    kwargs.pop("max_tokens", os.getenv("LLM_MAX_TOKEN_SIZE"))
    # kwargs.pop("response_format", None) # allow json
    host = kwargs.pop("host", None) or os.getenv("LLM_HOST")
    think = kwargs.pop("think", None)
    if think is None:
        think = _get_env_bool("OLLAMA_THINK", False)
    think = _coerce_bool(think, False)
    timeout = kwargs.pop("timeout", None)
    if timeout is None:
        timeout = _get_env_float("REQUEST_TIMEOUT")
    kwargs.pop("hashing_kv", None)
    api_key = kwargs.pop("api_key", None)
    headers = (
        {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
        if api_key
        else {"Content-Type": "application/json"}
    )
    ollama_client = ollama.AsyncClient(host=host, timeout=timeout, headers=headers)
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})
    try:
        response = await ollama_client.chat(
            model=model,
            messages=messages,
            think=think,
            stream=stream,
            **kwargs,
        )
    except httpx.ReadTimeout as exc:
        raise APITimeoutError(request=exc.request) from exc
    except httpx.ConnectTimeout as exc:
        raise APITimeoutError(request=exc.request) from exc
    except httpx.ConnectError as exc:
        raise APIConnectionError(request=exc.request) from exc
    if stream:
        """cannot cache stream response"""

        async def inner():
            async for chunk in response:
                yield chunk["message"]["content"]

        return inner()
    else:
        return response["message"]["content"]


async def ollama_model_complete(
    prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
) -> Union[str, AsyncIterator[str]]:
    keyword_extraction = kwargs.pop("keyword_extraction", None)
    if keyword_extraction:
        kwargs["format"] = "json"
        if _get_env_bool("OLLAMA_KEYWORDS_NO_THINK", False):
            kwargs["think"] = False
            no_think_prompt = (
                "Return JSON only. Do not output reasoning, analysis, or extra text."
            )
            if system_prompt:
                system_prompt = f"{system_prompt}\n{no_think_prompt}"
            else:
                system_prompt = no_think_prompt
    model_name = kwargs["hashing_kv"].global_config["llm_model_name"]
    return await ollama_model_if_cache(
        model_name,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )


async def ollama_embedding(texts: list[str], embed_model, **kwargs) -> np.ndarray:
    """
    Deprecated in favor of `embed`.
    """
    embed_text = []
    embed_model = _normalize_model_name(embed_model)
    ollama_client = ollama.Client(**kwargs)
    for text in texts:
        data = ollama_client.embeddings(model=embed_model, prompt=text)
        embed_text.append(data["embedding"])

    return embed_text


async def ollama_embed(texts: list[str], embed_model, **kwargs) -> np.ndarray:
    embed_model = _normalize_model_name(embed_model)
    api_key = kwargs.pop("api_key", None)
    headers = (
        {"Content-Type": "application/json", "Authorization": api_key}
        if api_key
        else {"Content-Type": "application/json"}
    )
    kwargs["headers"] = headers
    if "host" not in kwargs or not kwargs.get("host"):
        kwargs["host"] = os.getenv("LLM_HOST")
    if "timeout" not in kwargs or kwargs.get("timeout") is None:
        kwargs["timeout"] = _get_env_float("REQUEST_TIMEOUT")
    ollama_client = ollama.Client(**kwargs)
    data = ollama_client.embed(model=embed_model, input=texts)
    return data["embeddings"]
