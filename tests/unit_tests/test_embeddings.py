"""Test embedding model integration."""

from typing import Type

from langchain_tests.unit_tests import EmbeddingsUnitTests

from langchain_qualcomm_inference_suite.embeddings import QISEmbeddings


class TestQISEmbeddingsUnit(EmbeddingsUnitTests):
    @property
    def embeddings_class(self) -> Type[QISEmbeddings]:
        return QISEmbeddings

    @property
    def embedding_model_params(self) -> dict:
        return {"model": "BAAI/bge-large-en-v1.5"}
