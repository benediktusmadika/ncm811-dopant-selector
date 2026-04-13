from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .constants import (
    DEFAULT_JSON_OUTPUT_TOKENS_CAP,
    JSON_COMPACT_RETRY_SUFFIX,
    JSON_REPAIR_MAX_TRIM_CANDIDATES,
    JSON_REPAIR_TRIM_WINDOW_CHARS,
    KNOWN_MODEL_PREFIXES,
    LOGGER_NAME,
)
from .models import GenerationRequestSettings, ModelCapabilityProfile, ResponseEnvelope
from .optional_deps import jsonschema
from .utils import ensure_parent_dir, normalize_reasoning_effort

LOG = logging.getLogger(LOGGER_NAME)
_JSONSCHEMA_ABSENCE_LOGGED = False


def normalize_model_name(model: str) -> str:
    name = str(model or "").strip()
    for prefix in sorted(_KNOWN_MODEL_PREFIXES, key=len, reverse=True):
        if name == prefix or name.startswith(prefix + "-"):
            return prefix
    return name

def infer_model_capabilities(model: str) -> ModelCapabilityProfile:
    normalized = normalize_model_name(model)
    is_gpt5_family = normalized.startswith("gpt-5")
    is_o_series = bool(re.match(r"^o\d", normalized))

    supports_reasoning_config = is_gpt5_family or is_o_series
    supports_reasoning_none = normalized.startswith(("gpt-5.1", "gpt-5.2", "gpt-5.4"))
    only_high_reasoning = normalized.endswith("pro") and normalized.startswith("gpt-5")

    # Conservative temperature policy:
    # - GPT-5.4 / GPT-5.2: documented support only when reasoning effort is `none`
    # - other GPT-5 / o-series reasoning models: omit temperature unless the API
    #   explicitly proves it is accepted via future-compatible retries.
    temperature_requires_reasoning_none = normalized.startswith(("gpt-5.2", "gpt-5.4"))
    if is_gpt5_family:
        temperature_supported = temperature_requires_reasoning_none
    elif is_o_series:
        temperature_supported = False
    else:
        temperature_supported = True

    return ModelCapabilityProfile(
        raw_model=model,
        normalized_model=normalized,
        supports_reasoning_config=supports_reasoning_config,
        supports_reasoning_none=supports_reasoning_none,
        only_high_reasoning=only_high_reasoning,
        temperature_supported=temperature_supported,
        temperature_requires_reasoning_none=temperature_requires_reasoning_none,
    )

class OpenAIClient:
    """
    OpenAI wrapper supporting:
    - embeddings
    - Responses API JSON schema outputs with strict schema enforcement and safe fallback
    - model-aware request sanitization for reasoning vs non-reasoning models
    - conservative reproducibility controls
    """

    def __init__(
        self,
        api_key: Optional[str],
        model: str,
        embedding_model: str,
        timeout_s: float = 180.0,
        default_temperature: Optional[float] = 0.2,
        seed: Optional[int] = None,
    ):
        try:
            from openai import OpenAI  # type: ignore
        except Exception as e:
            raise RuntimeError("Missing dependency: openai. Install with: pip install openai") from e

        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError(
                "OpenAI API key not found. Provide --openai_api_key or set OPENAI_API_KEY in the environment."
            )

        self.client = OpenAI(timeout=timeout_s)
        self.model = model
        self.embedding_model = embedding_model
        self.default_temperature = default_temperature
        self.seed = seed
        self.model_profile = infer_model_capabilities(model)

        LOG.info(
            "OpenAI model profile | model=%s | normalized=%s | reasoning_config=%s | temp_supported=%s | seed=%s",
            self.model,
            self.model_profile.normalized_model,
            self.model_profile.supports_reasoning_config,
            self.model_profile.temperature_supported,
            self.seed,
        )

    def describe_model_profile(self) -> Dict[str, Any]:
        return self.model_profile.as_dict()

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        # Batch embeddings
        resp = self.client.embeddings.create(
            model=self.embedding_model,
            input=texts,
        )
        # OpenAI returns list of Embedding objects in resp.data
        return [d.embedding for d in resp.data]

    @staticmethod
    def _error_message(error: Exception) -> str:
        try:
            return str(error).strip().lower()
        except Exception:
            return ""

    @staticmethod
    def _looks_like_unsupported_parameter_error(message: str, parameter_name: str) -> bool:
        if not message:
            return False
        markers = (
            "unsupported",
            "not supported",
            "unknown parameter",
            "unexpected keyword",
            "invalid",
            "not allowed",
            "unrecognized",
        )
        return parameter_name in message and any(marker in message for marker in markers)

    def _normalize_reasoning_for_model(self, reasoning_effort: Optional[str]) -> Optional[str]:
        effort = normalize_reasoning_effort(reasoning_effort)
        if effort is None:
            return None
        if not self.model_profile.supports_reasoning_config:
            LOG.info(
                "Model %s does not expose reasoning config; omitting reasoning_effort=%s.",
                self.model,
                effort,
            )
            return None
        if effort == "none" and not self.model_profile.supports_reasoning_none:
            LOG.info(
                "Model %s does not support reasoning effort 'none'; omitting reasoning config and using the model default.",
                self.model,
            )
            return None
        if self.model_profile.only_high_reasoning and effort != "high":
            LOG.info(
                "Model %s only supports high reasoning; overriding reasoning_effort=%s -> high.",
                self.model,
                effort,
            )
            return "high"
        return effort

    @staticmethod
    def _normalize_temperature_value(temperature: Optional[float]) -> Optional[float]:
        if temperature is None:
            return None
        value = float(temperature)
        if value < 0 or value > 2:
            raise ValueError("temperature must be between 0 and 2")
        return value

    def _normalize_temperature_for_model(
        self,
        temperature: Optional[float],
        reasoning_effort: Optional[str],
    ) -> Optional[float]:
        value = self._normalize_temperature_value(temperature)
        if value is None:
            return None

        if self.model_profile.temperature_requires_reasoning_none:
            if reasoning_effort not in (None, "none"):
                LOG.info(
                    "Omitting temperature=%.3f for model %s because this GPT-5 configuration only documents temperature support when reasoning is none.",
                    value,
                    self.model,
                )
                return None
            return value

        if not self.model_profile.temperature_supported:
            LOG.info(
                "Omitting temperature=%.3f for model %s because this model family may reject temperature.",
                value,
                self.model,
            )
            return None

        return value

    def _initial_request_settings(
        self,
        *,
        temperature: Optional[float],
        reasoning_effort: Optional[str],
    ) -> GenerationRequestSettings:
        normalized_reasoning = self._normalize_reasoning_for_model(reasoning_effort)
        normalized_temperature = self._normalize_temperature_for_model(temperature, normalized_reasoning)
        return GenerationRequestSettings(
            temperature=normalized_temperature,
            reasoning_effort=normalized_reasoning,
        )

    def _maybe_relax_request_settings(
        self,
        settings: GenerationRequestSettings,
        error: Exception,
    ) -> bool:
        message = self._error_message(error)
        changed = False

        if settings.temperature is not None and self._looks_like_unsupported_parameter_error(message, "temperature"):
            LOG.warning("Dropping temperature and retrying after API rejected it for model=%s.", self.model)
            settings.temperature = None
            changed = True

        if settings.reasoning_effort == "xhigh" and "xhigh" in message:
            LOG.warning("Downgrading reasoning effort xhigh -> high and retrying for model=%s.", self.model)
            settings.reasoning_effort = "high"
            changed = True

        if settings.reasoning_effort is not None and (
            self._looks_like_unsupported_parameter_error(message, "reasoning")
            or self._looks_like_unsupported_parameter_error(message, "effort")
        ):
            LOG.warning("Dropping reasoning config and retrying after API rejected it for model=%s.", self.model)
            settings.reasoning_effort = None
            changed = True

        return changed

    def _build_responses_request_kwargs(
        self,
        *,
        system: str,
        user: str,
        max_output_tokens: int,
        settings: GenerationRequestSettings,
        text_format: Dict[str, Any],
    ) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "instructions": system,
            "input": user,
            "max_output_tokens": max_output_tokens,
            "text": text_format,
        }
        if settings.temperature is not None:
            kwargs["temperature"] = settings.temperature
        if settings.reasoning_effort is not None:
            kwargs["reasoning"] = {"effort": settings.reasoning_effort}
        return kwargs

    def _next_output_token_budget(self, current_max_output_tokens: int) -> int:
        proposed = current_max_output_tokens + max(512, current_max_output_tokens // 2)
        proposed = max(4096, proposed)
        return min(DEFAULT_JSON_OUTPUT_TOKENS_CAP, proposed)

    def _compact_json_retry_prompt(self, user: str) -> str:
        return user if JSON_COMPACT_RETRY_SUFFIX.strip() in user else user + JSON_COMPACT_RETRY_SUFFIX

    def _downgrade_reasoning_for_length(self, settings: GenerationRequestSettings) -> bool:
        reasoning_steps = ("xhigh", "high", "medium", "low", "minimal", "none")
        current = settings.reasoning_effort

        if current is None:
            if self.model_profile.supports_reasoning_config and not self.model_profile.only_high_reasoning:
                candidate = self._normalize_reasoning_for_model("minimal")
                if candidate is not None:
                    settings.reasoning_effort = candidate
                    LOG.warning(
                        "Structured JSON hit the token budget; constraining reasoning effort to %s for model=%s.",
                        candidate,
                        self.model,
                    )
                    return True
            return False

        if current not in reasoning_steps:
            return False

        for candidate_raw in reasoning_steps[reasoning_steps.index(current) + 1 :]:
            candidate = self._normalize_reasoning_for_model(candidate_raw)
            if candidate is None or candidate == current:
                continue
            LOG.warning(
                "Structured JSON hit the token budget; lowering reasoning effort %s -> %s for model=%s.",
                current,
                candidate,
                self.model,
            )
            settings.reasoning_effort = candidate
            return True
        return False

    def _request_and_extract_json(
        self,
        *,
        system: str,
        user: str,
        max_output_tokens: int,
        settings: GenerationRequestSettings,
        text_format: Dict[str, Any],
        debug_path: Optional[Path],
        phase: str,
        attempt: int,
    ) -> Tuple[ResponseEnvelope, Any]:
        resp = self.client.responses.create(
            **self._build_responses_request_kwargs(
                system=system,
                user=user,
                max_output_tokens=max_output_tokens,
                settings=settings,
                text_format=text_format,
            )
        )
        envelope = _extract_response_envelope(resp)
        _write_debug_response(
            debug_path,
            envelope,
            phase=phase,
            attempt=attempt,
            max_output_tokens=max_output_tokens,
            settings=settings,
        )
        return envelope, _extract_response_output_parsed(resp)

    def _finalize_json_result(self, obj: Any, schema: Dict[str, Any], *, stage: str) -> Any:
        validation_error = _validate_json_instance_if_available(obj, schema)
        if validation_error is not None:
            LOG.warning(
                "Local jsonschema validation failed before coercion at %s: %s",
                stage,
                _summarize_jsonschema_exception(validation_error),
            )
        coerced = coerce_json_to_schema(obj, schema)
        coerced_validation_error = _validate_json_instance_if_available(coerced, schema)
        if coerced_validation_error is not None:
            LOG.warning(
                "Local jsonschema validation still failed after coercion at %s: %s",
                stage,
                _summarize_jsonschema_exception(coerced_validation_error),
            )
        return coerced

    def call_json_schema(
        self,
        *,
        system: str,
        user: str,
        schema_name: str,
        schema: Dict[str, Any],
        max_output_tokens: int = 2500,
        temperature: Optional[float] = None,
        reasoning_effort: Optional[str] = None,
        max_attempts: int = 3,
        debug_path: Optional[Path] = None,
    ) -> Any:
        """
        Structured Outputs call with automatic schema strictness.
        If the schema is invalid, we auto-fix required fields and retry.
        If strict mode fails, we fall back to json_object and validate in code.

        Robustness improvements:
        - inspect Responses API status/incomplete_details before parsing JSON
        - adaptively increase max_output_tokens when output is truncated
        - reduce reasoning effort on length pressure for reasoning-capable models
        - perform best-effort JSON repair/parsing and optional local jsonschema validation
        """
        requested_temperature = self.default_temperature if temperature is None else temperature
        settings = self._initial_request_settings(
            temperature=requested_temperature,
            reasoning_effort=reasoning_effort,
        )
        schema_fixed = enforce_strict_schema(schema)
        _validate_schema_definition_if_available(schema_fixed)

        current_max_output_tokens = max_output_tokens
        current_user = user
        last_err: Optional[Exception] = None

        for attempt in range(1, max_attempts + 1):
            try:
                envelope, parsed_direct = self._request_and_extract_json(
                    system=system,
                    user=current_user,
                    max_output_tokens=current_max_output_tokens,
                    settings=settings,
                    text_format={
                        "format": {
                            "type": "json_schema",
                            "name": schema_name,
                            "schema": schema_fixed,
                            "strict": True,
                        }
                    },
                    debug_path=debug_path,
                    phase="json_schema",
                    attempt=attempt,
                )

                if parsed_direct is not None:
                    return self._finalize_json_result(parsed_direct, schema_fixed, stage="json_schema.output_parsed")

                if envelope.refusal_text:
                    raise RuntimeError(f"Model refusal during structured output: {envelope.refusal_text}")

                if envelope.status == "incomplete":
                    reason = envelope.incomplete_reason or "unknown"
                    if reason == "max_output_tokens":
                        changed = False
                        next_budget = self._next_output_token_budget(current_max_output_tokens)
                        if next_budget > current_max_output_tokens:
                            LOG.warning(
                                "OpenAI schema response incomplete attempt %d/%d because max_output_tokens was reached; retrying with budget %d -> %d.",
                                attempt,
                                max_attempts,
                                current_max_output_tokens,
                                next_budget,
                            )
                            current_max_output_tokens = next_budget
                            changed = True
                        if self._downgrade_reasoning_for_length(settings):
                            changed = True
                        current_user = self._compact_json_retry_prompt(current_user)
                        if changed and attempt < max_attempts:
                            time.sleep(0.5 * attempt)
                            continue
                        if envelope.raw_text.strip():
                            try:
                                obj = _parse_json_loose(envelope.raw_text)
                                LOG.warning(
                                    "Using salvaged partial JSON after exhausting token-budget retries during schema phase."
                                )
                                return self._finalize_json_result(
                                    obj,
                                    schema_fixed,
                                    stage="json_schema.partial_salvage",
                                )
                            except Exception as salvage_err:
                                last_err = salvage_err
                        raise RuntimeError(
                            f"OpenAI structured JSON response incomplete because max_output_tokens was reached (budget={current_max_output_tokens})."
                        )
                    if reason == "content_filter":
                        raise RuntimeError("OpenAI structured JSON response was interrupted by the content filter.")
                    raise RuntimeError(f"OpenAI structured JSON response incomplete (reason={reason}).")

                raw = envelope.raw_text or ""
                if not raw.strip():
                    raise ValueError("Structured JSON response did not contain any text output.")
                try:
                    obj = json.loads(raw)
                except Exception as parse_exc:
                    if _looks_like_truncated_json(raw, _root_exception(parse_exc)):
                        changed = False
                        next_budget = self._next_output_token_budget(current_max_output_tokens)
                        if next_budget > current_max_output_tokens:
                            LOG.warning(
                                "Structured JSON looked truncated on attempt %d/%d; retrying with budget %d -> %d.",
                                attempt,
                                max_attempts,
                                current_max_output_tokens,
                                next_budget,
                            )
                            current_max_output_tokens = next_budget
                            changed = True
                        if self._downgrade_reasoning_for_length(settings):
                            changed = True
                        current_user = self._compact_json_retry_prompt(current_user)
                        if changed and attempt < max_attempts:
                            time.sleep(0.5 * attempt)
                            continue
                    obj = _parse_json_loose(raw)
                return self._finalize_json_result(obj, schema_fixed, stage="json_schema")
            except Exception as e:
                last_err = e
                LOG.warning("OpenAI schema call failed attempt %d/%d (%s).", attempt, max_attempts, type(e).__name__)
                schema_fixed = enforce_strict_schema(schema_fixed)
                if self._maybe_relax_request_settings(settings, e):
                    time.sleep(0.4 * attempt)
                    continue
                time.sleep(0.6 * attempt)

        # Fallback: json_object mode (still GPT-driven, but we validate ourselves)
        LOG.warning("Falling back to json_object mode after schema failures: %s", last_err)
        fallback_prompt = self._compact_json_retry_prompt(
            user + "\n\nReturn ONLY valid JSON. No markdown. No extra text."
        )
        fallback_max_output_tokens = current_max_output_tokens
        last_fallback_err: Optional[Exception] = None
        for attempt in range(1, max_attempts + 1):
            try:
                envelope, parsed_direct = self._request_and_extract_json(
                    system=system,
                    user=fallback_prompt,
                    max_output_tokens=fallback_max_output_tokens,
                    settings=settings,
                    text_format={"format": {"type": "json_object"}},
                    debug_path=debug_path,
                    phase="json_object",
                    attempt=attempt,
                )

                if parsed_direct is not None:
                    return self._finalize_json_result(parsed_direct, schema_fixed, stage="json_object.output_parsed")

                if envelope.refusal_text:
                    raise RuntimeError(f"Model refusal during json_object fallback: {envelope.refusal_text}")

                if envelope.status == "incomplete":
                    reason = envelope.incomplete_reason or "unknown"
                    if reason == "max_output_tokens":
                        changed = False
                        next_budget = self._next_output_token_budget(fallback_max_output_tokens)
                        if next_budget > fallback_max_output_tokens:
                            LOG.warning(
                                "OpenAI json_object fallback incomplete attempt %d/%d because max_output_tokens was reached; retrying with budget %d -> %d.",
                                attempt,
                                max_attempts,
                                fallback_max_output_tokens,
                                next_budget,
                            )
                            fallback_max_output_tokens = next_budget
                            changed = True
                        if self._downgrade_reasoning_for_length(settings):
                            changed = True
                        fallback_prompt = self._compact_json_retry_prompt(fallback_prompt)
                        if changed and attempt < max_attempts:
                            time.sleep(0.5 * attempt)
                            continue
                        if envelope.raw_text.strip():
                            try:
                                obj = _parse_json_loose(envelope.raw_text)
                                LOG.warning(
                                    "Using salvaged partial JSON after exhausting token-budget retries during json_object fallback."
                                )
                                return self._finalize_json_result(
                                    obj,
                                    schema_fixed,
                                    stage="json_object.partial_salvage",
                                )
                            except Exception as salvage_err:
                                last_fallback_err = salvage_err
                        raise RuntimeError(
                            f"OpenAI json_object fallback incomplete because max_output_tokens was reached (budget={fallback_max_output_tokens})."
                        )
                    if reason == "content_filter":
                        raise RuntimeError("OpenAI json_object fallback was interrupted by the content filter.")
                    raise RuntimeError(f"OpenAI json_object fallback incomplete (reason={reason}).")

                raw = envelope.raw_text or ""
                if not raw.strip():
                    raise ValueError("OpenAI json_object fallback returned empty text output.")
                obj = _parse_json_loose(raw)
                return self._finalize_json_result(obj, schema_fixed, stage="json_object")
            except Exception as e:
                last_fallback_err = e
                LOG.warning(
                    "OpenAI json_object fallback failed attempt %d/%d (%s).",
                    attempt,
                    max_attempts,
                    type(e).__name__,
                )
                if self._maybe_relax_request_settings(settings, e):
                    time.sleep(0.4 * attempt)
                    continue
                time.sleep(0.6 * attempt)

        raise RuntimeError(f"OpenAI JSON generation failed after retries: {last_fallback_err or last_err}")

def _safe_get_field(obj: Any, field_name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(field_name, default)
    return getattr(obj, field_name, default)

def _get_jsonschema_validator_class() -> Any:
    if jsonschema is None:
        return None
    return getattr(jsonschema, "Draft202012Validator", None) or getattr(jsonschema, "Draft7Validator", None)

def _validate_schema_definition_if_available(schema: Dict[str, Any]) -> None:
    global _JSONSCHEMA_ABSENCE_LOGGED
    validator_cls = _get_jsonschema_validator_class()
    if validator_cls is None:
        if not _JSONSCHEMA_ABSENCE_LOGGED:
            LOG.info(
                "Optional dependency 'jsonschema' is not installed; local schema validation is disabled. Install with: pip install jsonschema"
            )
            _JSONSCHEMA_ABSENCE_LOGGED = True
        return
    try:
        validator_cls.check_schema(schema)
    except Exception as exc:
        LOG.warning("Local jsonschema schema check failed (continuing): %s", exc)

def _validate_json_instance_if_available(obj: Any, schema: Dict[str, Any]) -> Optional[Exception]:
    validator_cls = _get_jsonschema_validator_class()
    if validator_cls is None:
        return None
    try:
        validator_cls(schema).validate(obj)
        return None
    except Exception as exc:
        return exc

def _summarize_jsonschema_exception(exc: Exception) -> str:
    path = getattr(exc, "path", None)
    path_txt = ".".join(str(p) for p in path) if path else "<root>"
    return f"path={path_txt} | {exc}"

def _extract_response_output_parsed(resp: Any) -> Any:
    return _safe_get_field(resp, "output_parsed")

def _extract_response_envelope(resp: Any) -> ResponseEnvelope:
    fragments: List[str] = []
    seen: set[str] = set()
    refusal_text: Optional[str] = None

    def add_fragment(value: Any) -> None:
        text_value = value
        if text_value is None:
            return
        if not isinstance(text_value, str):
            text_value = _safe_get_field(text_value, "text", _safe_get_field(text_value, "value"))
        if not isinstance(text_value, str):
            return
        candidate = text_value.strip()
        if not candidate or candidate in seen:
            return
        seen.add(candidate)
        fragments.append(candidate)

    add_fragment(_safe_get_field(resp, "output_text"))

    output_items = _safe_get_field(resp, "output", []) or []
    if isinstance(output_items, list):
        for item in output_items:
            content_items = _safe_get_field(item, "content", []) or []
            if not isinstance(content_items, list):
                continue
            for part in content_items:
                part_type = str(_safe_get_field(part, "type", "") or "").strip().lower()
                if part_type == "refusal":
                    refusal_candidate = _safe_get_field(part, "refusal", _safe_get_field(part, "text", _safe_get_field(part, "value")))
                    if isinstance(refusal_candidate, str) and refusal_candidate.strip():
                        refusal_text = refusal_candidate.strip()
                    continue
                add_fragment(_safe_get_field(part, "text"))
                add_fragment(_safe_get_field(part, "value"))

    status = str(_safe_get_field(resp, "status", "") or "").strip().lower() or None
    incomplete_details = _safe_get_field(resp, "incomplete_details")
    incomplete_reason = str(_safe_get_field(incomplete_details, "reason", "") or "").strip().lower() or None
    raw_text = "\n".join(fragments).strip()

    return ResponseEnvelope(
        raw_text=raw_text,
        status=status,
        incomplete_reason=incomplete_reason,
        refusal_text=refusal_text,
    )

def _write_debug_response(
    debug_path: Optional[Path],
    envelope: ResponseEnvelope,
    *,
    phase: str,
    attempt: int,
    max_output_tokens: int,
    settings: GenerationRequestSettings,
) -> None:
    if not debug_path:
        return
    ensure_parent_dir(debug_path)
    header_lines = [
        f"phase={phase}",
        f"attempt={attempt}",
        f"status={envelope.status or 'unknown'}",
        f"incomplete_reason={envelope.incomplete_reason or ''}",
        f"max_output_tokens={max_output_tokens}",
        f"temperature={settings.temperature}",
        f"reasoning_effort={settings.reasoning_effort}",
        f"refusal_text={envelope.refusal_text or ''}",
        "",
    ]
    debug_path.write_text("\n".join(header_lines) + envelope.raw_text, encoding="utf-8")

def _root_exception(exc: Exception) -> Exception:
    current = exc
    while getattr(current, "__cause__", None) is not None:
        current = current.__cause__  # type: ignore[assignment]
    return current

def _looks_like_truncated_json(text: str, error: Exception) -> bool:
    raw = (text or "").rstrip()
    if not raw:
        return False
    lowered = str(error or "").lower()
    if raw[-1:] not in ("}", "]"):
        return True
    if raw.count("{") > raw.count("}") or raw.count("[") > raw.count("]"):
        return True
    if isinstance(error, json.JSONDecodeError) and error.pos >= max(0, len(raw) - 256):
        return True
    markers = (
        "unterminated string",
        "unexpected end",
        "expecting value",
        "expecting ',' delimiter",
        "expecting property name enclosed in double quotes",
        "unclosed",
    )
    return any(marker in lowered for marker in markers)

def _strip_json_fence(text: str) -> str:
    raw = (text or "").strip()
    if not raw:
        return ""
    full_match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", raw, flags=re.IGNORECASE | re.DOTALL)
    if full_match:
        return full_match.group(1).strip()
    partial_match = re.search(r"```(?:json)?\s*(.*?)\s*```", raw, flags=re.IGNORECASE | re.DOTALL)
    if partial_match:
        return partial_match.group(1).strip()
    return raw

def _extract_json_like_substring(text: str) -> str:
    raw = (text or "").strip()
    if not raw:
        return ""
    object_start = raw.find("{")
    array_start = raw.find("[")
    start_candidates = [idx for idx in (object_start, array_start) if idx >= 0]
    if not start_candidates:
        return raw
    start = min(start_candidates)
    opener = raw[start]
    closer = "}" if opener == "{" else "]"
    end = raw.rfind(closer)
    if end >= start:
        return raw[start : end + 1]
    return raw[start:]

def _repair_json_fragment(text: str) -> str:
    raw = (text or "").strip()
    if not raw:
        return raw

    out: List[str] = []
    closing_stack: List[str] = []
    in_string = False
    escaping = False

    for ch in raw:
        if in_string:
            out.append(ch)
            if escaping:
                escaping = False
            elif ch == "\\":
                escaping = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
            out.append(ch)
        elif ch == "{":
            closing_stack.append("}")
            out.append(ch)
        elif ch == "[":
            closing_stack.append("]")
            out.append(ch)
        elif ch in ("}", "]"):
            if closing_stack and closing_stack[-1] == ch:
                closing_stack.pop()
                out.append(ch)
        else:
            out.append(ch)

    repaired = "".join(out).rstrip()
    if in_string:
        if repaired.endswith("\\"):
            repaired = repaired[:-1]
        repaired += '"'
    repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
    repaired = repaired.rstrip()
    while repaired.endswith(","):
        repaired = repaired[:-1].rstrip()
    repaired += "".join(reversed(closing_stack))
    repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
    return repaired

def _candidate_trimmed_json_prefixes(text: str) -> List[str]:
    raw = (text or "").rstrip()
    if len(raw) <= 2:
        return []
    start = max(1, len(raw) - JSON_REPAIR_TRIM_WINDOW_CHARS)
    candidates: List[str] = []
    seen: set[str] = set()

    for idx in range(len(raw) - 1, start - 1, -1):
        ch = raw[idx]
        cut: Optional[int] = None
        if ch == ",":
            cut = idx
        elif ch in ("}", "]"):
            cut = idx + 1
        elif ch == "\n" and idx > 0 and raw[idx - 1] in ",}]":
            cut = idx
        if cut is None or cut <= 0:
            continue
        candidate = raw[:cut].rstrip()
        if candidate and candidate not in seen:
            seen.add(candidate)
            candidates.append(candidate)
        if len(candidates) >= JSON_REPAIR_MAX_TRIM_CANDIDATES:
            break
    return candidates

def _parse_json_loose(text: str) -> Any:
    raw = (text or "").strip()
    if not raw:
        raise ValueError("Failed to parse JSON from model output: empty response.")

    candidates: List[str] = []
    seen: set[str] = set()

    def add_candidate(candidate_text: str) -> None:
        cleaned = (candidate_text or "").strip()
        if not cleaned or cleaned in seen:
            return
        seen.add(cleaned)
        candidates.append(cleaned)

    stripped = _strip_json_fence(raw)
    extracted = _extract_json_like_substring(stripped)

    add_candidate(raw)
    add_candidate(stripped)
    add_candidate(extracted)
    add_candidate(_repair_json_fragment(raw))
    add_candidate(_repair_json_fragment(stripped))
    add_candidate(_repair_json_fragment(extracted))

    for base in list(candidates):
        for prefix in _candidate_trimmed_json_prefixes(base):
            add_candidate(prefix)
            add_candidate(_repair_json_fragment(prefix))

    first_error: Optional[Exception] = None
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except Exception as exc:
            if first_error is None:
                first_error = exc

    raise ValueError("Failed to parse JSON from model output.") from first_error

def coerce_json_to_schema(obj: Any, schema: Dict[str, Any]) -> Any:
    """
    Best-effort coercion/fill so downstream code does not crash when the model output is missing keys.
    This is NOT a full JSON-schema validator; it's a pragmatic "shape normalizer".

    Rules:
    - For objects: ensure every property key exists (fill missing with sensible defaults)
    - For arrays: ensure list; recurse into items
    - For scalars: return as-is when type-compatible, else return a safe default
    """
    def allowed_types(s: Dict[str, Any]) -> List[str]:
        t = s.get("type")
        if isinstance(t, list):
            return [str(x) for x in t]
        if isinstance(t, str):
            return [t]
        # Could be anyOf/oneOf; treat as unknown
        return []

    def default_for(s: Dict[str, Any]) -> Any:
        types = allowed_types(s)
        if "null" in types:
            return None
        if "array" in types:
            return []
        if "object" in types or "properties" in s:
            # construct empty object with filled props
            out: Dict[str, Any] = {}
            props = s.get("properties", {}) or {}
            if isinstance(props, dict):
                for k, sch in props.items():
                    out[k] = default_for(sch if isinstance(sch, dict) else {})
            return out
        if "number" in types or "integer" in types:
            return 0
        if "boolean" in types:
            return False
        # string or unknown
        return ""

    def rec(x: Any, s: Dict[str, Any]) -> Any:
        types = allowed_types(s)
        if ("object" in types) or ("properties" in s):
            props = s.get("properties", {}) or {}
            out: Dict[str, Any] = {}
            x_dict = x if isinstance(x, dict) else {}
            if isinstance(props, dict):
                for k, sch in props.items():
                    if k in x_dict:
                        out[k] = rec(x_dict.get(k), sch if isinstance(sch, dict) else {})
                    else:
                        out[k] = default_for(sch if isinstance(sch, dict) else {})
            return out
        if "array" in types:
            items = s.get("items", {}) if isinstance(s.get("items", {}), dict) else {}
            if not isinstance(x, list):
                x_list: List[Any] = []
            else:
                x_list = x
            return [rec(v, items) for v in x_list]
        # scalar handling
        if x is None:
            return default_for(s)
        if "number" in types or "integer" in types:
            try:
                return float(x)
            except Exception:
                return default_for(s)
        if "boolean" in types:
            return bool(x)
        if "string" in types:
            return str(x)
        return x

    if not isinstance(schema, dict):
        return obj
    return rec(obj, schema)

def enforce_strict_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    OpenAI Structured Outputs strict mode requires that:
    - For every object schema with properties, 'required' MUST include every key in properties.
    This function recursively enforces that requirement.

    It also sets additionalProperties=False by default for objects to reduce drift.
    """
    def rec(node: Any) -> Any:
        if not isinstance(node, dict):
            return node
        t = node.get("type")
        if t == "object" or ("properties" in node):
            props = node.get("properties", {})
            if isinstance(props, dict):
                node.setdefault("additionalProperties", False)
                # required must include all keys in properties
                node["required"] = list(props.keys())
                for k, v in props.items():
                    props[k] = rec(v)
                node["properties"] = props
            # also recurse into patternProperties etc if present
            if "items" in node:
                node["items"] = rec(node["items"])
            return node
        if t == "array":
            if "items" in node:
                node["items"] = rec(node["items"])
            return node
        # for any schema node, recurse into nested fields
        for k in list(node.keys()):
            if k in ("properties", "items", "anyOf", "oneOf", "allOf"):
                node[k] = rec(node[k])
        return node

    return rec(dict(schema))
