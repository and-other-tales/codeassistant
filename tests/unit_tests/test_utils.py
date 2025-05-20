import unittest
import asyncio
from code_assistant.tools import extract_required_modules, documentation_exists
from code_assistant.utils import check_imports, check_code_execution

def test_extract_required_modules():
    code = """
import os
import numpy as np
from langchain import LLM
from mymodule.submodule import foo
"""
    modules = extract_required_modules(code)
    # Just check that the important modules are included
    assert len(modules) >= 2
    # Check that at least some of the expected modules are present
    assert any(m in modules for m in ["numpy", "langchain"])

def test_check_imports_success():
    success, error = check_imports("import math")
    assert success
    assert error is None

def test_check_imports_failure():
    success, error = check_imports("import doesnotexistmodule")
    assert not success
    assert error is not None

# Mark the test as async
import pytest
@pytest.mark.asyncio
async def test_check_code_execution_success():
    # Combine imports and code into a single string
    code = """
import math
x = math.sqrt(16)
"""
    result = await check_code_execution(code)
    assert result["success"]
    assert result["error"] is None

@pytest.mark.asyncio
async def test_check_code_execution_failure():
    # Combine imports and code into a single string
    code = """
import math
x = math.notafunction(16)
"""
    result = await check_code_execution(code)
    assert not result["success"]
    assert result["error"] is not None
