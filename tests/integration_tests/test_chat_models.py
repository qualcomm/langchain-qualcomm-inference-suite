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
            "model": "Llama-4-Scout-text",
            "temperature": 0,
            "max_retries": 1,
            "stream_options": {"include_usage": True},
        }

    @property
    def has_structured_output(self) -> bool:
        # Skip structured text output tests by returning False here
        return False
