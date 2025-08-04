"""Test chat model integration."""

from typing import Type

from langchain_tests.unit_tests import ChatModelUnitTests

from langchain_qualcomm_inference_suite.chat_models import ChatQIS


class TestChatQISUnit(ChatModelUnitTests):
    @property
    def chat_model_class(self) -> Type[ChatQIS]:
        return ChatQIS

    @property
    def chat_model_params(self) -> dict:
        # These should be parameters used to initialize your integration for testing
        return {
            "model": "Llama-3.2-1B",
            "temperature": 0,
        }
