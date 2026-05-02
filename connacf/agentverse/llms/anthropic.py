"""
anthropic.py — Anthropic API backend for ConnaCF.

Behaves identically to BedrockClaudeAdapter from the perspective of all other classes.
Set ANTHROPIC_API_KEY in the environment (or ~/.bashrc for tmux sessions).

Registered aliases:
    anthropic-claude          → claude-sonnet-4-5-20250929 (default)
    anthropic-sonnet-4-5      → claude-sonnet-4-5-20250929
    anthropic-sonnet-4-6      → claude-sonnet-4-6-20251101 (when available)
    anthropic-haiku-4-5       → claude-haiku-4-5-20251001
"""
import os
from typing import Dict, List, Optional, Union

from loguru import logger
from pydantic import Field

from . import llm_registry
from .base import BaseChatModel, BaseModelArgs, LLMResult

_MODEL_MAP = {
    "anthropic-claude":     "claude-sonnet-4-5-20250929",
    "anthropic-sonnet-4-5": "claude-sonnet-4-5-20250929",
    "anthropic-sonnet-4-6": "claude-sonnet-4-6",
    "anthropic-haiku-4-5":  "claude-haiku-4-5-20251001",
    "anthropic-opus-4-6":   "claude-opus-4-6",
    "anthropic-opus-4-7":   "claude-opus-4-7",
}


class AnthropicClaudeArgs(BaseModelArgs):
    model: str = Field(default="anthropic-sonnet-4-5")
    max_tokens: int = Field(default=2048)
    temperature: float = Field(default=0.2)
    api_key: Optional[str] = Field(default=None)


@llm_registry.register("anthropic-claude")
@llm_registry.register("anthropic-sonnet-4-5")
@llm_registry.register("anthropic-sonnet-4-6")
@llm_registry.register("anthropic-haiku-4-5")
@llm_registry.register("anthropic-opus-4-6")
@llm_registry.register("anthropic-opus-4-7")
class AnthropicClaudeAdapter(BaseChatModel):
    """Anthropic API adapter — drop-in replacement for BedrockClaudeAdapter."""

    args: AnthropicClaudeArgs = Field(default_factory=AnthropicClaudeArgs)

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, max_retry: int = 3, **kwargs):
        args = AnthropicClaudeArgs()
        args_dict = args.dict()
        for k, v in args_dict.items():
            args_dict[k] = kwargs.pop(k, v)

        super().__init__(args=args_dict, max_retry=max_retry)

        import anthropic as _anthropic
        api_key = args_dict.get("api_key") or os.environ.get("ANTHROPIC_API_KEY")
        self._client = _anthropic.Anthropic(api_key=api_key)
        self._model = _MODEL_MAP.get(args_dict["model"], args_dict["model"])

        print(f"  ┌─ Anthropic LLM ready ────────────────────────────")
        print(f"  │  model  : {self._model}")
        print(f"  └──────────────────────────────────────────────────")
        logger.info(f"[LLM Init] AnthropicClaudeAdapter initialized: model={self._model}")

    def _to_messages(self, prompt) -> List[Dict]:
        if isinstance(prompt, str):
            return [{"role": "user", "content": prompt}]
        if isinstance(prompt, list) and prompt and isinstance(prompt[0], dict):
            return prompt
        return [{"role": "user", "content": str(prompt)}]

    def _call(self, messages: List[Dict]) -> str:
        resp = self._client.messages.create(
            model=self._model,
            max_tokens=self.args.max_tokens,
            temperature=self.args.temperature,
            messages=messages,
        )
        return resp.content[0].text

    def _estimate_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)

    def generate_response(self, prompt) -> LLMResult:
        messages = self._to_messages(prompt)
        logger.info(f"[LLM Inference] generate_response via Anthropic: model={self._model}")
        response_text = self._call(messages)
        send_tokens = self._estimate_tokens(str(messages))
        recv_tokens = self._estimate_tokens(response_text)
        return LLMResult(
            content=response_text,
            send_tokens=send_tokens,
            recv_tokens=recv_tokens,
            total_tokens=send_tokens + recv_tokens,
        )

    async def agenerate_response(self, prompt) -> Union[LLMResult, List]:
        messages = self._to_messages(prompt)
        if isinstance(messages[0], list):
            logger.info(f"[LLM Inference] agenerate_response (batch={len(messages)}) via Anthropic: model={self._model}")
            return [self._call(m) for m in messages]
        logger.info(f"[LLM Inference] agenerate_response via Anthropic: model={self._model}")
        return [self._call(messages)]

    async def agenerate_response_without_construction(self, prompts) -> List:
        responses = []
        for prompt in prompts:
            messages = self._to_messages(prompt)
            messages = [m for m in messages if m.get("content", "").strip()]
            if not messages:
                responses.append({"choices": [{"message": {"content": ""}}]})
                continue
            text = self._call(messages)
            responses.append({"choices": [{"message": {"content": text}}]})
        return responses
