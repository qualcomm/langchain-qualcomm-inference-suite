"""
Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.

SPDX-License-Identifier: BSD-3-Clause
"""

from importlib import metadata

from langchain_qualcomm_inference_suite.chat_models import ChatQIS
from langchain_qualcomm_inference_suite.embeddings import QISEmbeddings
from langchain_qualcomm_inference_suite.llms import QISLLM

try:
    __version__ = metadata.version(__package__)
except metadata.PackageNotFoundError:
    # Case where package metadata is not available.
    __version__ = ""
del metadata  # optional, avoids polluting the results of dir(__package__)

__all__ = [
    "ChatQIS",
    "QISEmbeddings",
    "QISLLM",
    "__version__",
]
