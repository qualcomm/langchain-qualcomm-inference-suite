"""Test ChatQIS chat model."""

from typing import Type

from langchain_tests.integration_tests import ChatModelIntegrationTests

from langchain_qualcomm_inference_suite.chat_models import ChatQIS


class TestChatQISIntegration(ChatModelIntegrationTests):
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

    @property
    def has_structured_output(self) -> bool:
        # Skip structured text output tests by returning False here
        return False
