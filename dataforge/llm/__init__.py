"""LLM 客户端与工具契约。仅供 L3 编排层使用。"""

from .client import BaseLLM, LLMError, LLMUsage, OpenAICompatLLM, StubLLM, from_env
from .contracts import ContractError, EXTRACT_VARIABLES, check_variable_semantics, validate_against

__all__ = [
    "BaseLLM", "ContractError", "EXTRACT_VARIABLES", "LLMError", "LLMUsage",
    "OpenAICompatLLM", "StubLLM", "check_variable_semantics", "from_env",
    "validate_against",
]
