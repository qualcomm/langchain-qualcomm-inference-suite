"""
Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.

SPDX-License-Identifier: BSD-3-Clause

Test QIS embeddings.
"""

from typing import Type

from langchain_tests.integration_tests import EmbeddingsIntegrationTests

from langchain_qualcomm_inference_suite.embeddings import QISEmbeddings


class TestQISEmbeddingsIntegration(EmbeddingsIntegrationTests):
    @property
    def embeddings_class(self) -> Type[QISEmbeddings]:
        return QISEmbeddings

    @property
    def embedding_model_params(self) -> dict:
        return {"model": "BAAI/bge-large-en-v1.5"}
