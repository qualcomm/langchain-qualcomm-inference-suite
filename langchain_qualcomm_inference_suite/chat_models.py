"""
Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.

SPDX-License-Identifier: BSD-3-Clause

Qualcomm Inference Suite chat models.
"""

import json
from typing import (
    Any,
    AsyncIterator,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Type,
    TypeVar,
    Union,
    cast,
)

from imagine import ChatCompletionStreamResponse
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.language_models.chat_models import (
    agenerate_from_stream,
    generate_from_stream,
)
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    BaseMessageChunk,
    ChatMessage,
    ChatMessageChunk,
    FunctionMessage,
    FunctionMessageChunk,
    HumanMessage,
    HumanMessageChunk,
    InvalidToolCall,
    SystemMessage,
    SystemMessageChunk,
    ToolCall,
    ToolMessage,
    ToolMessageChunk,
    ToolCallChunk,
)
from langchain_core.messages.ai import UsageMetadata
from langchain_core.output_parsers.openai_tools import (
    make_invalid_tool_call,
    parse_tool_call,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from pydantic import BaseModel, Field

from langchain_qualcomm_inference_suite.mixins import BaseLangChainMixin

_BM = TypeVar("_BM", bound=BaseModel)
_DictOrPydanticClass = Union[Dict[str, Any], Type[_BM], Type]
_DictOrPydantic = Union[Dict, _BM]


class OpenAIRefusalError(Exception):
    """Error raised when OpenAI Structured Outputs API returns a refusal.

    When using OpenAI's Structured Outputs API with user-generated input, the model
    may occasionally refuse to fulfill the request for safety reasons.

    See here for more on refusals:
    https://platform.openai.com/docs/guides/structured-outputs/refusals
    """


def _convert_delta_to_message_chunk(
    _dict: Mapping[str, Any], default_class: Type[BaseMessageChunk]
) -> BaseMessageChunk:
    id_ = _dict.get("id")
    role = cast(str, _dict.get("role"))
    content = cast(str, _dict.get("content") or "")
    additional_kwargs: Dict = {}
    if _dict.get("function_call"):
        function_call = dict(_dict["function_call"])
        if "name" in function_call and function_call["name"] is None:
            function_call["name"] = ""
        additional_kwargs["function_call"] = function_call
    tool_call_chunks = []
    if raw_tool_calls := _dict.get("tool_calls"):
        additional_kwargs["tool_calls"] = raw_tool_calls
        try:
            tool_call_chunks = [
                ToolCallChunk(
                    name=rtc["function"].get("name"),
                    args=rtc["function"].get("arguments"),
                    id=rtc.get("id"),
                    index=rtc["index"],
                )
                for rtc in raw_tool_calls
            ]
        except KeyError:
            pass

    if role == "user" or default_class == HumanMessageChunk:
        return HumanMessageChunk(content=content, id=id_)
    elif role == "assistant" or default_class == AIMessageChunk:
        return AIMessageChunk(
            content=content,
            additional_kwargs=additional_kwargs,
            id=id_,
            tool_call_chunks=tool_call_chunks,
        )
    elif role == "system" or default_class == SystemMessageChunk:
        return SystemMessageChunk(content=content, id=id_)
    elif role == "function" or default_class == FunctionMessageChunk:
        return FunctionMessageChunk(content=content, name=_dict["name"], id=id_)
    elif role == "tool" or default_class == ToolMessageChunk:
        return ToolMessageChunk(
            content=content, tool_call_id=_dict["tool_call_id"], id=id_
        )
    elif role or default_class == ChatMessageChunk:
        return ChatMessageChunk(content=content, role=role, id=id_)
    else:
        return default_class(content=content, id=id_)  # type: ignore


def _lc_tool_call_to_qualcomm_inference_suite_tool_call(tool_call: ToolCall) -> dict:
    return {
        "type": "function",
        "id": tool_call["id"],
        "function": {
            "name": tool_call["name"],
            "arguments": json.dumps(tool_call["args"]),
        },
    }


def _lc_invalid_tool_call_to_qualcomm_inference_suite_tool_call(
    invalid_tool_call: InvalidToolCall,
) -> dict:
    return {
        "type": "function",
        "id": invalid_tool_call["id"],
        "function": {
            "name": invalid_tool_call["name"],
            "arguments": invalid_tool_call["args"],
        },
    }


def _format_message_content(content: Any) -> Any:
    """Format message content."""
    if content and isinstance(content, list):
        # Remove unexpected block types
        formatted_content = []
        for block in content:
            if (
                isinstance(block, dict)
                and "type" in block
                and block["type"] == "tool_use"
            ):
                continue
            else:
                formatted_content.append(block)
    else:
        formatted_content = content

    return formatted_content


def _convert_message_to_dict(message: BaseMessage) -> dict:
    """Convert a LangChain message to a dictionary.

    Args:
        message: The LangChain message.

    Returns:
        The dictionary.
    """
    message_dict: Dict[str, Any] = {
        "content": _format_message_content(message.content),
    }
    if (name := message.name or message.additional_kwargs.get("name")) is not None:
        message_dict["name"] = name

    # populate role and additional message data
    if isinstance(message, ChatMessage):
        message_dict["role"] = message.role
    elif isinstance(message, HumanMessage):
        message_dict["role"] = "user"
    elif isinstance(message, AIMessage):
        message_dict["role"] = "assistant"
        if "function_call" in message.additional_kwargs:
            message_dict["function_call"] = message.additional_kwargs["function_call"]
        if message.tool_calls or message.invalid_tool_calls:
            message_dict["tool_calls"] = [
                _lc_tool_call_to_qualcomm_inference_suite_tool_call(tc)
                for tc in message.tool_calls
            ] + [
                _lc_invalid_tool_call_to_qualcomm_inference_suite_tool_call(tc)
                for tc in message.invalid_tool_calls
            ]
        elif "tool_calls" in message.additional_kwargs:
            message_dict["tool_calls"] = message.additional_kwargs["tool_calls"]
            tool_call_supported_props = {"id", "type", "function"}
            message_dict["tool_calls"] = [
                {k: v for k, v in tool_call.items() if k in tool_call_supported_props}
                for tool_call in message_dict["tool_calls"]
            ]
        else:
            pass
        # If tool calls present, content null value should be None not empty string.
        if "function_call" in message_dict or "tool_calls" in message_dict:
            message_dict["content"] = message_dict["content"] or None
    elif isinstance(message, SystemMessage):
        message_dict["role"] = "system"
    elif isinstance(message, FunctionMessage):
        message_dict["role"] = "function"
    elif isinstance(message, ToolMessage):
        message_dict["role"] = "tool"
        message_dict["tool_call_id"] = message.tool_call_id

        supported_props = {"content", "role", "tool_call_id"}
        message_dict = {k: v for k, v in message_dict.items() if k in supported_props}
    else:
        raise TypeError(f"Got unknown type {message}")
    return message_dict


def _convert_dict_to_message(_dict: Mapping[str, Any]) -> BaseMessage:
    """Convert a dictionary to a LangChain message.

    Args:
        _dict: The dictionary.

    Returns:
        The LangChain message.
    """
    role: str = _dict.get("role", "")
    name = _dict.get("name")
    id_ = _dict.get("id")
    if role == "user":
        return HumanMessage(content=_dict.get("content", ""), id=id_, name=name)
    elif role == "assistant":
        # Fix for azure
        # Also OpenAI returns None for tool invocations
        content = _dict.get("content", "") or ""
        additional_kwargs: Dict = {}
        if function_call := _dict.get("function_call"):
            additional_kwargs["function_call"] = dict(function_call)
        tool_calls = []
        invalid_tool_calls = []
        if raw_tool_calls := _dict.get("tool_calls"):
            additional_kwargs["tool_calls"] = raw_tool_calls
            for raw_tool_call in raw_tool_calls:
                try:
                    tool_calls.append(parse_tool_call(raw_tool_call, return_id=True))
                except Exception as e:
                    invalid_tool_calls.append(
                        make_invalid_tool_call(raw_tool_call, str(e))
                    )
        return AIMessage(
            content=content,
            additional_kwargs=additional_kwargs,
            name=name,
            id=id_,
            tool_calls=tool_calls,
            invalid_tool_calls=invalid_tool_calls,
        )
    elif role == "system":
        return SystemMessage(content=_dict.get("content", ""), name=name, id=id_)
    elif role == "function":
        return FunctionMessage(
            content=_dict.get("content", ""), name=cast(str, _dict.get("name")), id=id_
        )
    elif role == "tool":
        additional_kwargs = {}
        if "name" in _dict:
            additional_kwargs["name"] = _dict["name"]
        return ToolMessage(
            content=_dict.get("content", ""),
            tool_call_id=cast(str, _dict.get("tool_call_id")),
            additional_kwargs=additional_kwargs,
            name=name,
            id=id_,
        )
    else:
        return ChatMessage(content=_dict.get("content", ""), role=role, id=id_)


class ChatQIS(BaseChatModel, BaseLangChainMixin):
    """Qualcomm Inference Suite chat model integration.

    Setup:
        Install ``langchain-qualcomm-inference-suite`` and set environment variables
        ``IMAGINE_API_KEY`` and ``IMAGINE_ENDPOINT_URL``.

        .. code-block:: bash

            pip install -U langchain-qualcomm-inference-suite
            export IMAGINE_API_KEY="your-api-key"
            export IMAGINE_ENDPOINT_URL="https://my-endpoint/api/v2"

    Key init args — completion params:
        model: str
            Name of Qualcomm Inference Suite model to use.
        temperature: float
            Sampling temperature.
        max_tokens: Optional[int]
            Max number of tokens to generate.
        top_k: Optional[int]
            Integer that controls the number of top tokens to consider. Set to -1 to
            consider all tokens.
        top_p: Optional[float]
            An alternative to sampling with temperature, called nucleus sampling, where
            the model considers the results of the tokens with top_p probability mass.
        streaming: bool
            Whether to stream the output of the model.
        frequency_penalty: Optional[float]
            Number between -2.0 and 2.0. Positive values penalize new tokens based on
            their existing frequency in the text so far.
        presence_penalty: Optional[float]
            Number between -2.0 and 2.0. Positive values penalize new tokens based on
            whether they appear in the text so far.
        repetition_penalty: Optional[float]
            Float that penalizes new tokens based on whether they appear in the prompt
            and the generated text so far.
        stop: Optional[List[str]]
            Sequences where the API will stop generating further tokens.
        max_seconds: Optional[int]
            Maximum number of seconds to allow for generation.
        ignore_eos: Optional[bool]
            Whether to ignore the EOS token and continue generating tokens after the EOS
            token is generated.
        skip_special_tokens: Optional[bool]
            Whether to skip special tokens in the output.
        stop_token_ids: Optional[List[List[int]]]
            List of tokens that stop the generation when they are generated.

    Key init args — client params:
        timeout: Optional[float]
            Timeout for requests.
        max_retries: int
            Max number of retries.
        api_key: Optional[str]
            Qualcomm Inference Suite API key. If not passed in will be read from env var
            IMAGINE_API_KEY.
        endpoint: Optional[str]
            Qualcomm Inference Suite API endpoint. If not passed in will be read from env
            var IMAGINE_API_ENDPOINT.

    See full list of supported init args and their descriptions in the params section.

    Instantiate:
        .. code-block:: python

            from langchain_qualcomm_inference_suite import ChatQIS

            llm = ChatQIS(
                model="Llama-3.1-8B",
                temperature=0,
                max_tokens=None,
                timeout=None,
                max_retries=2,
                # api_key="...",
                # other params...
            )

    Invoke:
        .. code-block:: python

            messages = [
                ("system", "You are a helpful translator. Translate the user sentence to
                French."),
                ("human", "I love programming."),
            ]
            llm.invoke(messages)

        .. code-block:: python

            AIMessage(content='The translation of "I love programming" to French
            is:\n\n"J\'adore le programmation."', additional_kwargs={},
            response_metadata={'token_usage': {'prompt_tokens': 33, 'total_tokens': 55,
            'completion_tokens': 22}, 'model_name': 'Llama-3.1-8B',
            'system_fingerprint': '', 'finish_reason': <FinishReason.stop: 'stop'>},
            id='run-c182fb49-12dc-4790-841c-d781418c4364-0',
            usage_metadata={'input_tokens': 33, 'output_tokens': 22,
            'total_tokens': 55})

    Stream:
        .. code-block:: python

            for chunk in llm.stream(messages):
                print(chunk.text(), end="")

        .. code-block:: python

            The translation of "I love programming" to French is:
            "J'
            adore le programmation."

        .. code-block:: python

            stream = llm.stream(messages)
            full = next(stream)
            for chunk in stream:
                full += chunk
            full

        .. code-block:: python

            AIMessageChunk(content='The translation of "I love programming" to French
            is:\n\n"J\'adore le programmation."', additional_kwargs={},
            response_metadata={'finish_reason': <FinishReason.stop: 'stop'>},
            id='run-eaf1a058-4aaf-48e0-bdcd-2a822eb8da46')

    Async:
        .. code-block:: python

            await llm.ainvoke(messages)

            # stream:
            # async for chunk in (await llm.astream(messages)):
            #     ...

            # batch:
            # await llm.abatch([messages])

    Token usage:
        .. code-block:: python

            ai_msg = llm.invoke(messages)
            ai_msg.usage_metadata

        .. code-block:: python

            {'input_tokens': 33, 'output_tokens': 22, 'total_tokens': 55}

    Response metadata
        .. code-block:: python

            ai_msg = llm.invoke(messages)
            ai_msg.response_metadata

        .. code-block:: python

             {'token_usage': {'prompt_tokens': 33, 'total_tokens': 55,
             'completion_tokens': 22}, 'model_name': 'Llama-3.1-8B',
             'system_fingerprint': '', 'finish_reason': <FinishReason.stop: 'stop'>}

    """  # noqa: E501

    model_name: str = Field(alias="model")
    """The name of the model"""
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """Holds any model parameters valid for `create` call not explicitly specified."""
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    timeout: Optional[int] = None
    stop: Optional[List[str]] = None
    streaming: bool = False
    max_retries: int = 2

    disabled_params: Optional[Dict[str, Any]] = Field(default=None)
    """Parameters of the Qualcomm Inference Suite client or chat.completions endpoint
    that should be disabled for the given model.

    Should be specified as ``{"param": None | ['val1', 'val2']}`` where the key is the 
    parameter and the value is either None, meaning that parameter should never be
    used, or it's a list of disabled values for the parameter.

    For example, older models may not support the 'parallel_tool_calls' parameter at 
    all, in which case ``disabled_params={"parallel_tool_calls": None}`` can be passed 
    in.

    If a parameter is disabled then it will not be used by default in any methods.
    However this does not prevent a user from directly passed in the parameter during
    invocation. 
    """

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "chat-qualcomm-inference-suite"

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling OpenAI API."""
        params = {
            "model": self.model_name,
            "stream": self.streaming,
            "temperature": self.temperature,
            **self.model_kwargs,
        }
        if self.max_tokens is not None:
            params["max_tokens"] = self.max_tokens
        return params

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Return a dictionary of identifying parameters.

        This information is used by the LangChain callback system, which
        is used for tracing purposes make it possible to monitor LLMs.
        """
        return {
            # The model name allows users to specify custom token counting
            # rules in LLM monitoring applications (e.g., in LangSmith users
            # can provide per token pricing for their model and monitor
            # costs for the given LLM.)
            "model_name": self.model_name,
        }

    def _create_message_dicts(
        self, messages: list[BaseMessage], stop: Optional[List[str]] = None
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        params = self._default_params
        if stop is not None:
            if "stop" in params:
                raise ValueError("`stop` found in both the input and default params.")
            params["stop"] = stop
        message_dicts = [_convert_message_to_dict(m) for m in messages]
        return message_dicts, params

    def _create_chat_result(self, response: Union[dict, BaseModel]) -> ChatResult:
        generations = []
        if not isinstance(response, dict):
            response = response.model_dump()

        # Sometimes the AI Model calling will get error, we should raise it.
        # Otherwise, the next code 'choices.extend(response["choices"])'
        # will throw a "TypeError: 'NoneType' object is not iterable" error
        # to mask the true error. Because 'response["choices"]' is None.
        if response.get("error"):
            raise ValueError(response.get("error"))

        token_usage = response.get("usage", {})
        for res in response["choices"]:
            message = _convert_dict_to_message(res["message"])
            if token_usage and isinstance(message, AIMessage):
                message.usage_metadata = {
                    "input_tokens": token_usage.get("prompt_tokens", 0),
                    "output_tokens": token_usage.get("completion_tokens", 0),
                    "total_tokens": token_usage.get("total_tokens", 0),
                }
            generation_info = dict(finish_reason=res.get("finish_reason"))
            if "logprobs" in res:
                generation_info["logprobs"] = res["logprobs"]
            gen = ChatGeneration(
                message=message,
                generation_info=generation_info,
            )
            generations.append(gen)
        llm_output = {
            "token_usage": token_usage,
            "model_name": self.model_name,
            "system_fingerprint": response.get("system_fingerprint", ""),
        }
        return ChatResult(generations=generations, llm_output=llm_output)

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """
        Args:
            messages: the prompt composed of a list of messages.
            stop: a list of strings on which the model should stop generating.
                  If generation stops due to a stop token, the stop token itself
                  SHOULD BE INCLUDED as part of the output. This is not enforced
                  across models right now, but it's a good practice to follow since
                  it makes it much easier to parse the output of the model
                  downstream and understand why generation stopped.
            run_manager: A run manager with callbacks for the LLM.
        """
        message_dicts, params = self._create_message_dicts(messages)
        params = {"stop": stop, **params, **kwargs}
        params.pop("stream", "")

        if self.streaming:
            stream_iter = self.client.chat_stream(
                messages=message_dicts,
                **params,  # type: ignore
            )
            return generate_from_stream(stream_iter)  # type: ignore

        response = self.client.chat(messages=message_dicts, **params)
        return self._create_chat_result(response)

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        *,
        stream_usage: Optional[bool] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """Stream the output of the model.

        Args:
            messages: the prompt composed of a list of messages.
            stop: a list of strings on which the model should stop generating.
                  If generation stops due to a stop token, the stop token itself
                  SHOULD BE INCLUDED as part of the output. This is not enforced
                  across models right now, but it's a good practice to follow since
                  it makes it much easier to parse the output of the model
                  downstream and understand why generation stopped.
            run_manager: A run manager with callbacks for the LLM.
        """
        message_dicts, params = self._create_message_dicts(messages, stop)
        params = {**params, **kwargs}
        # We are already calling a stream method, so no need to pass this:
        params.pop("stream")

        default_chunk_class = AIMessageChunk
        for chunk in self.client.chat_stream(messages=message_dicts, **params):
            if not isinstance(chunk, dict):
                chunk = chunk.model_dump()
            if len(chunk["choices"]) == 0:
                if token_usage := chunk.get("usage"):
                    usage_metadata = UsageMetadata(
                        input_tokens=token_usage.get("prompt_tokens", 0),
                        output_tokens=token_usage.get("completion_tokens", 0),
                        total_tokens=token_usage.get("total_tokens", 0),
                    )
                    chunk = ChatGenerationChunk(
                        message=default_chunk_class(
                            content="", usage_metadata=usage_metadata
                        )
                    )
                else:
                    continue
            else:
                choice = chunk["choices"][0]
                if choice["delta"] is None:
                    continue
                chunk = _convert_delta_to_message_chunk(
                    choice["delta"], default_chunk_class
                )
                generation_info = {}
                if finish_reason := choice.get("finish_reason"):
                    generation_info["finish_reason"] = finish_reason
                logprobs = choice.get("logprobs")
                if logprobs:
                    generation_info["logprobs"] = logprobs
                default_chunk_class = chunk.__class__
                chunk = ChatGenerationChunk(
                    message=chunk, generation_info=generation_info or None
                )
            if run_manager:
                run_manager.on_llm_new_token(chunk.text, chunk=chunk, logprobs=logprobs)
            yield chunk

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        message_dicts, params = self._create_message_dicts(messages, stop)
        params = {**params, **kwargs}
        # We are already calling a stream method, so no need to pass this:
        params.pop("stream")

        default_chunk_class = AIMessageChunk
        async for chunk in self.async_client.chat_stream(
            messages=message_dicts, **params
        ):
            if not isinstance(chunk, dict):
                chunk = chunk.model_dump()
            if len(chunk["choices"]) == 0:
                if token_usage := chunk.get("usage"):
                    usage_metadata = UsageMetadata(
                        input_tokens=token_usage.get("prompt_tokens", 0),
                        output_tokens=token_usage.get("completion_tokens", 0),
                        total_tokens=token_usage.get("total_tokens", 0),
                    )
                    chunk = ChatGenerationChunk(
                        message=default_chunk_class(
                            content="", usage_metadata=usage_metadata
                        )
                    )
                else:
                    continue
            else:
                choice = chunk["choices"][0]
                if choice["delta"] is None:
                    continue
                chunk = _convert_delta_to_message_chunk(
                    choice["delta"], default_chunk_class
                )
                generation_info = {}
                if finish_reason := choice.get("finish_reason"):
                    generation_info["finish_reason"] = finish_reason
                logprobs = choice.get("logprobs")
                if logprobs:
                    generation_info["logprobs"] = logprobs
                default_chunk_class = chunk.__class__
                chunk = ChatGenerationChunk(
                    message=chunk, generation_info=generation_info or None
                )
            if run_manager:
                await run_manager.on_llm_new_token(
                    token=chunk.text, chunk=chunk, logprobs=logprobs
                )
            yield chunk

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        message_dicts, params = self._create_message_dicts(messages)
        params = {**params, **kwargs}
        params.pop("stream", "")

        should_stream = (
            kwargs["stream"] if kwargs.get("stream") is not None else self.streaming
        )
        if should_stream:
            stream_iter = self._astream(
                messages=messages, stop=stop, run_manager=run_manager, **params
            )
            return await agenerate_from_stream(stream_iter)

        response = await self.async_client.chat(messages=message_dicts, **params)

        return self._create_chat_result(response)

    def _filter_disabled_params(self, **kwargs: Any) -> Dict[str, Any]:
        if not self.disabled_params:
            return kwargs
        filtered = {}
        for k, v in kwargs.items():
            # Skip param
            if k in self.disabled_params and (
                self.disabled_params[k] is None or v in self.disabled_params[k]
            ):
                continue
            # Keep param
            else:
                filtered[k] = v
        return filtered
