"""
Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.

SPDX-License-Identifier: BSD-3-Clause
"""

from typing import List

from langchain_core.embeddings import Embeddings
from pydantic import BaseModel, Field

from langchain_qualcomm_inference_suite.mixins import BaseLangChainMixin


class QISEmbeddings(Embeddings, BaseModel, BaseLangChainMixin):
    """Qualcomm AI Inference Suite embedding model integration.

    Setup:
        Install ``langchain-qualcomm-inference-suite`` and set environment variables
        ``IMAGINE_API_KEY`` and ``IMAGINE_API_ENDPOINT``.

        .. code-block:: bash

            pip install -U langchain-qualcomm-inference-suite
            export IMAGINE_API_KEY="your-api-key"
            export IMAGINE_ENDPOINT_URL="https://my-endpoint/api/v2"

    Key init args — completion params:
        model: str
            Name of QIS model to use.

    Key init args — client params:
        timeout: Optional[float]
            Timeout for requests.
        max_retries: int
            Max number of retries.
        api_key: Optional[str]
            Qualcomm AI Inference Suite API key. If not passed in will be read from env var
            IMAGINE_API_KEY.
        endpoint: Optional[str]
            Qualcomm AI Inference Suite API endpoint. If not passed in will be read from
            env var IMAGINE_API_ENDPOINT.

    See full list of supported init args and their descriptions in the params section.

    Instantiate:
        .. code-block:: python

            from langchain_qualcomm_inference_suite import QISEmbeddings

            embed = QISEmbeddings(
                model="...",
                # api_key="...",
                # other params...
            )

    Embed single text:
        .. code-block:: python

            input_text = "The meaning of life is 42"
            embed.embed_query(input_text)

        .. code-block:: python

            [0.01593017578125, -0.029754638671875, ..., -0.028839111328125]

    Embed multiple text:
        .. code-block:: python

            input_texts = ["Document 1...", "Document 2..."]
            embed.embed_documents(input_texts)

        .. code-block:: python

            [[0.018341064453125, -0.019134521484375, ..., 0.006160736083984375]]

    Async:
        .. code-block:: python

            await embed.aembed_query(input_text)

            # multiple:
            # await embed.aembed_documents(input_texts)

        .. code-block:: python

            [0.01593017578125, -0.029754638671875, ..., -0.028839111328125]

    """

    model: str = Field(alias="model")
    """The name of the model"""

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search docs."""
        try:
            response = self.client.embeddings(text=texts, model=self.model)

            return [embedding_obj.embedding for embedding_obj in response.data]
        except Exception:
            raise

    def embed_query(self, text: str) -> List[float]:
        """Embed query text."""
        return self.embed_documents([text])[0]

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Asynchronous Embed search docs."""
        try:
            response = await self.async_client.embeddings(text=texts, model=self.model)

            return [embedding_obj.embedding for embedding_obj in response.data]
        except Exception:
            raise

    async def aembed_query(self, text: str) -> List[float]:
        """Asynchronous Embed query text."""
        return (await self.aembed_documents([text]))[0]
