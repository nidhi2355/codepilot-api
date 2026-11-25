import os
import re
import time
import logging
from typing import Type, TypeVar
import httpx
from pydantic import BaseModel, ValidationError
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import JsonOutputParser

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

# Active Groq self-serve / free-tier text models (Nov 2026)
DEFAULT_GROQ_MODELS = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "qwen/qwen3.8-27b",
]

NON_CHAT_PREFIXES = (
    "whisper-",
    "canopylabs/",
    "meta-llama/llama-prompt-guard",
    "openai/gpt-oss-safeguard",
)

_AVAILABLE_MODELS_CACHE: tuple[float, set[str]] | None = None
_CACHE_TTL_SECONDS = 600.0


def _fetch_account_models(api_key: str) -> set[str]:
    """Queries Groq's /openai/v1/models endpoint to discover models enabled for this API key."""
    global _AVAILABLE_MODELS_CACHE
    now = time.time()
    if _AVAILABLE_MODELS_CACHE and (now - _AVAILABLE_MODELS_CACHE[0]) < _CACHE_TTL_SECONDS:
        return _AVAILABLE_MODELS_CACHE[1]

    try:
        res = httpx.get(
            "https://api.groq.com/openai/v1/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=10.0,
        )
        if res.status_code == 200:
            data = res.json().get("data", [])
            ids = {
                m["id"]
                for m in data
                if isinstance(m, dict)
                and m.get("id")
                and m.get("active", True)
                and not m["id"].startswith(NON_CHAT_PREFIXES)
            }
            if ids:
                _AVAILABLE_MODELS_CACHE = (now, ids)
                return ids
    except Exception as e:
        logger.debug(f"Could not fetch live Groq model list: {e}")

    return set()


def _get_model_candidates(api_key: str) -> list[str]:
    primary = os.getenv("GROQ_MODEL", "").strip()
    ordered = [primary] if primary else []
    for m in DEFAULT_GROQ_MODELS:
        if m not in ordered:
            ordered.append(m)

    available = _fetch_account_models(api_key)
    if not available:
        return ordered

    filtered = [m for m in ordered if m in available]
    for m in sorted(available):
        if m not in filtered:
            filtered.append(m)

    return filtered or ordered


def invoke_structured_groq(
    prompt: ChatPromptTemplate,
    payload: dict,
    schema: Type[T],
    temperature: float = 0.1,
    max_json_retries: int = 2,
) -> T:
    """
    Invokes ChatGroq with structured JSON parsing, short TPM backoff, and automatic
    failover across active Groq self-serve models when a model's TPD quota is exhausted.
    """
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GROQ_API_KEY environment variable is not set")

    parser = JsonOutputParser(pydantic_object=schema)
    full_payload = {
        **payload,
        "format_instructions": parser.get_format_instructions(),
    }

    model_errors: list[str] = []

    for model_name in _get_model_candidates(api_key):
        base_llm = ChatGroq(
            model=model_name,
            api_key=api_key,
            temperature=temperature,
        )
        chain = prompt | base_llm.bind(tool_choice="none") | parser

        json_attempt = 0
        tpm_retries = 0

        while json_attempt < max_json_retries:
            try:
                raw = chain.invoke(full_payload)
                return schema(**raw)

            except (ValidationError, ValueError, TypeError) as val_err:
                model_errors.append(f"{model_name}: invalid JSON schema ({val_err})")
                json_attempt += 1
                logger.warning(
                    f"[{model_name}] JSON schema validation failed ({json_attempt}/{max_json_retries}): {val_err}"
                )

            except Exception as err:
                msg = str(err)
                msg_lower = msg.lower()

                is_429 = "429" in msg or "rate_limit_exceeded" in msg_lower
                wait_match = re.search(r"try again in\s+([0-9.]+)s", msg, re.IGNORECASE)
                wait_secs = float(wait_match.group(1)) if wait_match else None

                is_tpd = (
                    "tokens per day" in msg_lower
                    or "(tpd)" in msg_lower
                    or (wait_secs is not None and wait_secs > 35.0)
                )
                is_model_unavailable = any(
                    k in msg_lower
                    for k in (
                        "model_not_found",
                        "model_decommissioned",
                        "does not exist",
                        "do not have access",
                        "tool_use_failed",
                    )
                )

                if is_429 and not is_tpd and tpm_retries < 2:
                    sleep_for = min((wait_secs or 10.0) + 1.0, 30.0)
                    tpm_retries += 1
                    logger.warning(
                        f"[{model_name}] TPM rate limit hit; waiting {sleep_for:.1f}s ({tpm_retries}/2)..."
                    )
                    time.sleep(sleep_for)
                    continue

                if is_429 or is_model_unavailable:
                    reason = "daily TPD limit reached" if is_tpd else "unavailable/rate-limited"
                    model_errors.append(f"{model_name}: {reason}")
                    logger.warning(f"[{model_name}] {reason}; failing over to next Groq model...")
                    break

                raise

    summary = "; ".join(dict.fromkeys(model_errors)) or "No Groq models succeeded"
    raise RuntimeError(f"All Groq models exhausted or failed ({summary})")