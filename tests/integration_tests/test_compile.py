"""
Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.

SPDX-License-Identifier: BSD-3-Clause

Test compilation.
"""

import pytest


@pytest.mark.compile
def test_placeholder() -> None:
    """Used for compiling integration tests without running any real tests."""
    pass
