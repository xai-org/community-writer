import asyncio
import re

from xai_sdk import AsyncClient
from xai_sdk.chat import user, system
from xai_sdk.tools import web_search, x_search

MAX_MESSAGE_LENGTH = 50 * 1024 * 1024

from data_models.writer_data_models import GrokOutput
from note_writer.llm_client import (
    JSONValidationError,
    LLMClient,
    ModelRateLimiter,
    extract_json_content,
)

from utils.log_setup import get_logger

logger = get_logger("grok_client")


def shorten_image_content(content: str, peek_length: int = 300) -> str:
    if re.search(r"data:image/[a-zA-Z]{1,5};base64", content[:peek_length]):
        return content[:peek_length] + "..."
    return content


class GrokClient(LLMClient):
    _rate_limiters: dict[str, ModelRateLimiter] = {}
    _rate_limiters_lock: asyncio.Lock = asyncio.Lock()

    def __init__(
        self,
        api_key: str,
        model: str,
        model_uri: str,
        enable_web_image_understanding: bool,
        enable_x_image_understanding: bool,
        enable_x_video_understanding: bool,
        temperature: float,
        timeout: float | None = None,
    ):
        super().__init__(
            api_key=api_key,
            model=model,
            model_uri=model_uri,
            temperature=temperature,
        )
        self._enable_web_image_understanding: bool = enable_web_image_understanding
        self._enable_x_image_understanding: bool = enable_x_image_understanding
        self._enable_x_video_understanding: bool = enable_x_video_understanding
        self._timeout: float | None = timeout
        self._client: AsyncClient | None = None

    @property
    def client(self) -> AsyncClient:
        if self._client is None:
            kwargs = dict(
                api_key=self._api_key,
                api_host=self._model_uri,
                channel_options=[
                    ("grpc.max_receive_message_length", MAX_MESSAGE_LENGTH),
                    ("grpc.max_send_message_length", MAX_MESSAGE_LENGTH),
                ],
            )
            if self._timeout is not None:
                kwargs["timeout"] = self._timeout
            self._client = AsyncClient(**kwargs)
        return self._client

    async def get_grok_response(
        self,
        prompt: str | list,
        timeout: float | None = None,
        retries: int = 3,
        system_prompt: str | None = None,
        expect_json: bool = False,
        base_delay: float = 0.0,
        post_id: int | None = None,
    ) -> GrokOutput:
        await self._rate_limit_start()

        chat = self.client.chat.create(
            model=self._model,
            temperature=self._temperature,
            tools=[
                web_search(
                    enable_image_understanding=self._enable_web_image_understanding,
                ),
                x_search(
                    enable_image_understanding=self._enable_x_image_understanding,
                    enable_video_understanding=self._enable_x_video_understanding,
                ),
            ],
        )

        chat.append(
            system(system_prompt or "You are a highly intelligent AI assistant.")
        )
        if isinstance(prompt, list):
            chat.append(user(*prompt))
        else:
            chat.append(user(prompt))

        async def sample_response() -> GrokOutput:
            trace = []
            tool_calls = []
            tool_outputs = []
            reasoning_buffer = []
            content_buffer = []
            citations = []

            final_prompt = None
            final_reasoning_content = None
            final_response_content = None
            response_id = None
            loop = asyncio.get_running_loop()
            start_time = loop.time()

            def flush_reasoning() -> None:
                if not reasoning_buffer:
                    return
                trace.append("<REASONING_CONTENT>\n")
                trace.append("".join(reasoning_buffer))
                trace.append("\n</REASONING_CONTENT>\n\n")
                reasoning_buffer.clear()

            async for response, chunk in chat.stream():
                if not trace:
                    assert chunk.debug_output
                    trace.append("<INITIAL_PROMPT>\n")
                    trace.append(chunk.debug_output.prompt)
                    trace.append("\n</INITIAL_PROMPT>\n\n")
                if chunk.reasoning_content:
                    reasoning_buffer.append(chunk.reasoning_content)
                else:
                    flush_reasoning()
                if chunk.tool_calls:
                    trace.append("<TOOL_CALLS>\n")
                    for tool_call in chunk.tool_calls:
                        trace.append(str(tool_call).strip())
                        tool_calls.append(tool_call)
                    trace.append("\n</TOOL_CALLS>\n\n")
                if chunk.tool_outputs:
                    trace.append("<TOOL_OUTPUTS>\n")
                    for output in chunk.tool_outputs:
                        trace.append(shorten_image_content(output.content))
                        tool_outputs.append(output)
                    trace.append("\n</TOOL_OUTPUTS>\n\n")
                if chunk.content:
                    content_buffer.append(chunk.content)
                if chunk.citations:
                    citations.extend(chunk.citations)

                if chunk.debug_output.prompt:
                    final_prompt = chunk.debug_output.prompt
                if response.reasoning_content:
                    final_reasoning_content = response.reasoning_content
                if response.content:
                    final_response_content = response.content
                if response.id:
                    response_id = response.id

            flush_reasoning()
            assert content_buffer, f"model {self._model} returned no content"
            trace.append("<CONTENT>\n")
            trace.append("".join(content_buffer))
            trace.append("\n</CONTENT>\n\n")
            latency = loop.time() - start_time

            assert len(tool_calls) == len(tool_outputs)
            tool_calls_dicts = [
                {
                    "id": tool_call.id,
                    "type": tool_call.type,
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments,
                    },
                    "output": shorten_image_content(tool_output.content),
                }
                for tool_call, tool_output in zip(tool_calls, tool_outputs)
            ]

            response_content = "".join(content_buffer)
            if expect_json:
                response_content = extract_json_content(response_content)
            return GrokOutput(
                content=response_content,
                citations=citations,
                tool_calls=tool_calls_dicts,
                parsed_trace="".join(trace),
                final_prompt=final_prompt,
                final_reasoning_content=final_reasoning_content,
                final_response_content=final_response_content,
                model=self._model,
                response_id=response_id,
                latency=latency,
            )

        for attempt in range(1, retries + 1):
            try:
                if timeout is not None:
                    return await asyncio.wait_for(sample_response(), timeout=timeout)
                return await sample_response()
            except JSONValidationError as e:
                if attempt == retries:
                    raise
                if attempt >= 2:
                    logger.exception(
                        f"LLM retry error: model={self._model}, post_id={post_id}, attempt={attempt}/{retries}, error={type(e).__name__}: {e}"
                    )
                await asyncio.sleep(5.0 * attempt)
            except Exception as e:
                if attempt == retries:
                    raise
                if attempt >= 2:
                    logger.exception(
                        f"LLM retry error: model={self._model}, post_id={post_id}, attempt={attempt}/{retries}, error={type(e).__name__}: {e}"
                    )
                await asyncio.sleep(base_delay + 60.0 * (2**attempt))
        raise RuntimeError("Maximum number of LLM retries exceeded.")

    async def _rate_limit_start(self) -> None:
        limiter = await self._get_or_create_rate_limiter(
            self._model,
            self._rate_limiters,
            self._rate_limiters_lock,
        )
        await limiter.wait_for_slot()
