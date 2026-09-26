import asyncio
import json
import re
from abc import ABC, abstractmethod

from data_models.writer_data_models import GrokOutput


class JSONValidationError(ValueError):
    pass


_FENCE_PATTERN = re.compile(r"```[A-Za-z0-9_-]*\s*\n?(.*?)```", re.DOTALL)


_TRAILING_COMMA_PATTERN = re.compile(r",(\s*[}\]])")


_BOLD_PATTERN = re.compile(r"\*\*")


def _repair_variants(candidate: str) -> list[str]:
    variants: list[str] = []
    for text in (candidate, _BOLD_PATTERN.sub("", candidate)):
        for repaired in (text, _TRAILING_COMMA_PATTERN.sub(r"\1", text)):
            if repaired not in variants:
                variants.append(repaired)
    return variants


def _balanced_span(text: str) -> str | None:
    start = next((i for i, ch in enumerate(text) if ch in "{["), None)
    if start is None:
        return None
    opener = text[start]
    closer = "}" if opener == "{" else "]"
    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def extract_json_content(content: str) -> str:
    text = content.strip()
    candidates = [text]
    fenced = _FENCE_PATTERN.search(text)
    if fenced:
        candidates.append(fenced.group(1).strip())
    span = _balanced_span(text)
    if span:
        candidates.append(span)

    for candidate in candidates:
        if not candidate:
            continue
        for repaired in _repair_variants(candidate):
            try:
                json.loads(repaired)
            except (json.JSONDecodeError, ValueError):
                continue
            return repaired

    raise JSONValidationError(f"Expected valid JSON response but got: {content[:500]}")


class ModelRateLimiter:
    def __init__(self, rate_per_second: float = 45.0) -> None:
        self._interval: float = 1.0 / rate_per_second
        self._lock: asyncio.Lock = asyncio.Lock()
        self._next_allowed_time: float = 0.0

    async def wait_for_slot(self) -> None:
        loop = asyncio.get_running_loop()
        async with self._lock:
            now = loop.time()
            scheduled = max(now, self._next_allowed_time)
            self._next_allowed_time = scheduled + self._interval
        delay = scheduled - now
        if delay > 0:
            await asyncio.sleep(delay)


class LLMClient(ABC):
    def __init__(
        self,
        api_key: str,
        model: str,
        model_uri: str,
        temperature: float,
    ):
        self._api_key: str = api_key
        self._model: str = model
        self._model_uri: str = model_uri
        self._temperature: float = temperature

    @abstractmethod
    async def get_grok_response(
        self,
        prompt: str | list,
        timeout: float | None = None,
        system_prompt: str | None = None,
        expect_json: bool = False,
        retries: int = 3,
        base_delay: float = 0.0,
        post_id: int | None = None,
    ) -> GrokOutput:
        raise NotImplementedError

    @staticmethod
    async def _get_or_create_rate_limiter(
        model: str,
        rate_limiters: dict[str, ModelRateLimiter],
        rate_limiters_lock: asyncio.Lock,
    ) -> ModelRateLimiter:
        async with rate_limiters_lock:
            limiter = rate_limiters.get(model)
            if limiter is None:
                limiter = ModelRateLimiter()
                rate_limiters[model] = limiter
            return limiter
