import os
import re
import time
import logging
from typing import Type, TypeVar
from pydantic import BaseModel, ValidationError
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import JsonOutputParser

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

DEFAULT_GROQ_MODELS = [
    "openai/gpt-oss-120b",
    "llama-3.3-70b-versatile",
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "qwen/qwen3-32b",
    "llama-3.1-8b-instant",
]


def _get_model_candidates() -> list[str]:
    primary = os.getenv("GROQ_MODEL", "").strip()
    models = [primary] if primary else []
    for m in DEFAULT_GROQ_MODELS:
        if m not in models:
            models.append(m)
    return models


def invoke_structured_groq(
    prompt: ChatPromptTemplate,
    payload: dict,
    schema: Type[T],
    temperature: float = 0.1,
    max_json_retries: int = 2,
) -> T:
    """
    Invokes ChatGroq with structured JSON parsing, short TPM backoff, and automatic
    failover across Groq models when a model's daily token quota (TPD) is exhausted.
    """
    api_key = os.getenv("GROQ_API_KEY")
    parser = JsonOutputParser(pydantic_object=schema)
    full_payload = {
        **payload,
        "format_instructions": parser.get_format_instructions(),
    }

    last_error: Exception | None = None

    for model_name in _get_model_candidates():
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
                last_error = val_err
                json_attempt += 1
                logger.warning(
                    f"[{model_name}] JSON schema validation failed ({json_attempt}/{max_json_retries}): {val_err}"
                )

            except Exception as err:
                last_error = err
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
                    for k in ("model_not_found", "model_decommissioned", "does not exist", "tool_use_failed")
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
                    logger.warning(
                        f"[{model_name}] Quota/availability limit reached; failing over to next Groq model..."
                    )
                    break

                raise

    raise RuntimeError(f"All Groq models exhausted or failed: {last_error}")