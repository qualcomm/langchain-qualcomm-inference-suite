"""
Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.

SPDX-License-Identifier: BSD-3-Clause
"""

from typing import Any

from imagine import ImagineAsyncClient, ImagineClient
from pydantic import ConfigDict, model_validator


class BaseLangChainMixin:
    """This mixin adds base functionality common to all the LangChain classes."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    client: ImagineClient
    async_client: ImagineAsyncClient
    verify: bool = False

    @model_validator(mode="before")
    @classmethod
    def pre_root(cls, values: dict[str, Any]) -> dict[str, Any]:
        client_params = {
            "endpoint": values.pop("endpoint", None),
            "api_key": values.pop("api_key", None),
        }

        optional_client_params = ["max_retries", "timeout", "verify", "proxy", "debug"]
        for param in optional_client_params:
            if param in values:
                client_params[param] = values.pop(param)

        values["client"] = ImagineClient(**client_params)
        values["async_client"] = ImagineAsyncClient(**client_params)

        return values
