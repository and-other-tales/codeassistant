import pytest
from code_assistant.utils import extract_required_modules, documentation_exists, check_imports, check_code_execution

def test_extract_required_modules():
    code = """
import os
import numpy as np
from langchain import LLM
from mymodule.submodule import foo
"""
    modules = extract_required_modules(code)
    assert set(modules) == {"os", "numpy", "langchain", "mymodule"}

def test_check_imports_success():
    success, error = check_imports("import math")
    assert success
    assert error is None

def test_check_imports_failure():
    success, error = check_imports("import doesnotexistmodule")
    assert not success
    assert error is not None

def test_check_code_execution_success():
    success, error = check_code_execution("import math", "x = math.sqrt(16)")
    assert success
    assert error is None

def test_check_code_execution_failure():
    success, error = check_code_execution("import math", "x = math.notafunction(16)")
    assert not success
    assert error is not None
