"""
Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.

SPDX-License-Identifier: BSD-3-Clause
"""

from typing import Any, AsyncIterator, Iterator, List, Optional

from imagine import CompletionRequest
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import LLM
from langchain_core.outputs import GenerationChunk

from langchain_qualcomm_inference_suite.mixins import BaseLangChainMixin


class QISLLM(LLM, BaseLangChainMixin):
    """Qualcomm AI Inference Suite large language models.

    To use, you should have the environment variable ``IMAGINE_API_KEY``
    set with your API key, or pass it as a named parameter to the constructor.
    """

    model: str = "Llama-3.2-1B"

    temperature: float = 0.0
    max_tokens: Optional[int] = None
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    streaming: bool = False

    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    repetition_penalty: Optional[float] = None
    stop: Optional[List[str]] = None
    max_seconds: Optional[int] = None
    ignore_eos: Optional[bool] = None
    skip_special_tokens: Optional[bool] = None
    stop_token_ids: Optional[List[List[int]]] = None

    @property
    def _default_params(self) -> dict[str, Any]:
        """Get the default parameters for calling the API."""

        default_body = CompletionRequest(
            prompt="",
            model=self.model,
            stream=self.streaming,
            frequency_penalty=self.frequency_penalty,
            presence_penalty=self.presence_penalty,
            repetition_penalty=self.repetition_penalty,
            stop=self.stop,
            max_seconds=self.max_seconds,
            ignore_eos=self.ignore_eos,
            skip_special_tokens=self.skip_special_tokens,
            stop_token_ids=self.stop_token_ids,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
        ).model_dump(exclude_none=True)

        default_body.pop("prompt", "")

        return default_body

    @property
    def _identifying_params(self) -> dict[str, Any]:
        """Get the identifying parameters."""
        return self._default_params

    @property
    def _llm_type(self) -> str:
        """Return type of llm model."""
        return "imagine-llm"

    def _call(
        self,
        prompt: str,
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> str:
        params = {**self._default_params, **kwargs}
        params.pop("stream", "")

        if self.streaming:
            completion = ""
            for chunk in self._stream(
                prompt=prompt, stop=stop, run_manager=run_manager, **params
            ):
                completion += chunk.text
            return completion

        response = self.client.completion(prompt=prompt, **params)
        return response.choices[0].text

    async def _acall(
        self,
        prompt: str,
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> str:
        """Call out to Imagine's completion endpoint asynchronously."""

        params = {**self._default_params, **kwargs}
        params.pop("stream", "")

        if self.streaming:
            completion = ""

            async for chunk in self._astream(
                prompt=prompt, stop=stop, run_manager=run_manager, **params
            ):
                completion += chunk.text
            return completion

        response = await self.async_client.completion(
            prompt=prompt, stop=stop, **params
        )

        return response.choices[0].text

    def _stream(
        self,
        prompt: str,
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        params = {**self._default_params, **kwargs}
        params.pop("stream", "")

        for stream_resp in self.client.completion_stream(
            prompt=prompt, stop=stop, **params
        ):
            chunk = GenerationChunk(text=stream_resp.choices[0].delta.content)  # type: ignore

            if run_manager:
                run_manager.on_llm_new_token(chunk.text, chunk=chunk)
            yield chunk

    async def _astream(
        self,
        prompt: str,
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[GenerationChunk]:
        params = {**self._default_params, **kwargs}
        params.pop("stream", "")

        async for token in self.async_client.completion_stream(
            prompt=prompt,
            stop=stop,
            **params,
        ):
            chunk = GenerationChunk(text=token.choices[0].delta.content)  # type: ignore

            if run_manager:
                await run_manager.on_llm_new_token(chunk.text, chunk=chunk)
            yield chunk
