"""LLM 客户端。

**只有 L3 编排层允许构造它**（设计文档 §2.1）。节点通过 ``NodeContext.llm``
拿到实例，自己不知道模型是怎么配的——这样节点仍然可以脱离模型单测。

核心是 :meth:`LLMClient.structured`：给 JSON Schema，拿回**校验过**的对象。
校验失败会把错误喂回去重试，重试到上限仍失败就抛——不返回半成品让下游猜。
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable

from .contracts import ContractError, validate_against


class LLMError(RuntimeError):
    """模型调用失败，且重试已用尽。"""


@dataclass
class LLMUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    calls: int = 0

    @property
    def total(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def add(self, p: int, c: int) -> None:
        self.prompt_tokens += p
        self.completion_tokens += c
        self.calls += 1

    def to_dict(self) -> dict[str, int]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total": self.total,
            "calls": self.calls,
        }


def _extract_json(text: str) -> Any:
    """从模型回复里挖出 JSON。

    即便要求了「只输出 JSON」，模型仍可能裹上 ```json 围栏或加一句解释。
    与其指望提示词管用，不如在这里兜住。
    """
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.+?)\s*```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # 退一步：取第一个平衡的 { } 或 [ ]
    for opener, closer in (("{", "}"), ("[", "]")):
        i = text.find(opener)
        if i < 0:
            continue
        depth, in_str, esc = 0, False, False
        for j in range(i, len(text)):
            ch = text[j]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == opener:
                depth += 1
            elif ch == closer:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[i:j + 1])
                    except json.JSONDecodeError:
                        break
    raise LLMError(f"模型回复里找不到合法 JSON：{text[:220]}")


class BaseLLM:
    """所有客户端的接口。"""

    name = "base"

    def __init__(self) -> None:
        self.usage = LLMUsage()

    def complete(self, system: str, user: str, *, temperature: float = 0.0) -> str:
        raise NotImplementedError

    def structured(
        self,
        *,
        system: str,
        user: str,
        schema: dict[str, Any],
        max_retries: int = 3,
        on_attempt: Callable[[int, str], None] | None = None,
    ) -> Any:
        """要一个符合 schema 的对象回来。

        Args:
            max_retries: 校验失败后的重试次数。用尽仍不合规就抛 :class:`LLMError`。
            on_attempt: 每次尝试后的回调 ``(第几次, 错误或空串)``，供轨迹记录。
        """
        hint = ""
        last_err = ""
        for attempt in range(1, max_retries + 1):
            raw = self.complete(system, user + hint, temperature=0.0)
            try:
                obj = _extract_json(raw)
                validate_against(obj, schema)
                if on_attempt:
                    on_attempt(attempt, "")
                return obj
            except (ContractError, LLMError) as e:
                last_err = str(e)
                if on_attempt:
                    on_attempt(attempt, last_err)
                # 把校验错误原样喂回去——比重说一遍要求有效得多
                hint = (
                    f"\n\n上一次的输出不符合要求：{last_err}\n"
                    "请仅输出修正后的 JSON，不要任何解释文字。"
                )
        raise LLMError(f"重试 {max_retries} 次仍不符合契约。最后一次错误：{last_err}")


class OpenAICompatLLM(BaseLLM):
    """OpenAI 兼容协议。DeepSeek / Ollama / vLLM 都走这个。"""

    name = "openai-compat"

    def __init__(
        self,
        model: str | None = None,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 120.0,
    ) -> None:
        super().__init__()
        # 密钥只从环境变量取，绝不写进代码或配置文件
        self.api_key = api_key or os.getenv("DATAFORGE_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise LLMError(
                "没有配置模型密钥。请设置环境变量 DATAFORGE_API_KEY（或 OPENAI_API_KEY）。"
            )
        self.model = model or os.getenv("DATAFORGE_MODEL", "deepseek-chat")
        self.base_url = base_url or os.getenv("DATAFORGE_BASE_URL", "https://api.deepseek.com/v1")
        self.timeout = timeout
        self._client = None

    def _get(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(
                api_key=self.api_key, base_url=self.base_url, timeout=self.timeout
            )
        return self._client

    def complete(self, system: str, user: str, *, temperature: float = 0.0) -> str:
        resp = self._get().chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
        )
        u = getattr(resp, "usage", None)
        if u:
            self.usage.add(getattr(u, "prompt_tokens", 0) or 0,
                           getattr(u, "completion_tokens", 0) or 0)
        return resp.choices[0].message.content or ""


@dataclass
class StubLLM(BaseLLM):
    """确定性桩，给测试与离线演示用。

    按「提示词里包含某个关键词」匹配预设回复。测试不该依赖网络，
    也不该因为模型今天心情不同而变红。
    """

    responses: list[tuple[str, Any]] = field(default_factory=list)
    default: Any = None
    calls: list[dict[str, str]] = field(default_factory=list)
    name: str = "stub"

    def __post_init__(self) -> None:
        self.usage = LLMUsage()

    def complete(self, system: str, user: str, *, temperature: float = 0.0) -> str:
        self.calls.append({"system": system, "user": user})
        self.usage.add(len(system) // 4, 64)
        for needle, reply in self.responses:
            if needle in user or needle in system:
                return reply if isinstance(reply, str) else json.dumps(reply, ensure_ascii=False)
        if self.default is None:
            raise LLMError(f"StubLLM 没有匹配的预设回复。提示词片段：{user[:120]}")
        return (self.default if isinstance(self.default, str)
                else json.dumps(self.default, ensure_ascii=False))


def from_env() -> BaseLLM | None:
    """按环境变量构造客户端；没配密钥就返回 None。

    返回 None 时上层应当**明确告诉用户「没有配模型」**，
    而不是悄悄退化成规则方法还宣称用了模型。
    """
    try:
        return OpenAICompatLLM()
    except LLMError:
        return None
