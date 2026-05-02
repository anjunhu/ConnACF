"""
Local HuggingFace inference backend — drop-in replacement for bedrock.py.
For reviewers with GPU access who don't want to pay for Bedrock.

Usage (same interface as BedrockClaudeLLM):
    from connacf.agentverse.llms.hf import HFLocalLLM
    llm = HFLocalLLM("Qwen/Qwen3-8B")   # or any HF model ID
    response = llm([{"role": "user", "content": "Hello"}])

Supported model families (auto-detected via chat template):
  - Qwen3:   Qwen/Qwen3-8B, Qwen/Qwen3-14B, ...
  - Llama3:  meta-llama/Llama-3.1-8B-Instruct, ...
  - Mistral: mistralai/Mistral-7B-Instruct-v0.3, ...
  - Any model with a HuggingFace chat template

Requirements:
    pip install transformers accelerate torch sentence-transformers
"""

from typing import Dict, List, Optional, Union
import logging
import torch
from pydantic import Field

from . import llm_registry
from .base import BaseChatModel, BaseModelArgs, LLMResult

logger = logging.getLogger("websocietysimulator")

# ---------------------------------------------------------------------------
# Singleton cache — one loaded model per (model_id, quantization) key
# ---------------------------------------------------------------------------
_HF_MODEL_CACHE: dict = {}

def _get_or_load_llm(model_name_or_path: str, device_map: str, load_in_4bit: bool, load_in_8bit: bool) -> "HFLocalLLM":
    key = (model_name_or_path, device_map, load_in_4bit, load_in_8bit)
    if key not in _HF_MODEL_CACHE:
        logger.info(f"[HF] Loading model weights: {model_name_or_path}")
        _HF_MODEL_CACHE[key] = HFLocalLLM(
            model_name_or_path=model_name_or_path,
            device_map=device_map,
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit,
        )
    else:
        logger.info(f"[HF] Reusing cached model: {model_name_or_path}")
    return _HF_MODEL_CACHE[key]

# ---------------------------------------------------------------------------
# Recommended cheap/small models for local GPU inference
# ---------------------------------------------------------------------------
RECOMMENDED_LOCAL_MODELS = {
    # ~8B models — fit on a single 24GB GPU
    "qwen3-8b":    "Qwen/Qwen3-8B",
    "qwen3-14b":   "Qwen/Qwen3-14B",
    "llama3-8b":   "meta-llama/Llama-3.1-8B-Instruct",
    "mistral-7b":  "mistralai/Mistral-7B-Instruct-v0.3",
    # ~32B models — fit on 2×24GB or 1×80GB
    "qwen3-32b":   "Qwen/Qwen3-32B",
    "llama3-70b":  "meta-llama/Llama-3.1-70B-Instruct",
}


class HFLocalEmbeddings:
    """Sentence-transformers embedding model, mirrors BedrockTitanEmbeddings API."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts, convert_to_numpy=True).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.model.encode([text], convert_to_numpy=True)[0].tolist()


class HFLocalLLM:
    """
    Local HuggingFace LLM with the same __call__ signature as BedrockClaudeLLM.

    Args:
        model_name_or_path: HuggingFace model ID or local path.
        device_map: "auto" (recommended), "cuda", "cpu", etc.
        load_in_4bit: Enable bitsandbytes 4-bit quantisation (saves ~75% VRAM).
        load_in_8bit: Enable bitsandbytes 8-bit quantisation (saves ~50% VRAM).
        embedding_model: HF sentence-transformers model for embeddings.
        max_new_tokens_default: Default max tokens when not specified per call.
    """

    def __init__(
        self,
        model_name_or_path: str = "Qwen/Qwen3-8B",
        device_map: str = "auto",
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        max_new_tokens_default: int = 500,
    ):
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

        self.model_name = model_name_or_path
        self.max_new_tokens_default = max_new_tokens_default

        quant_config = None
        if load_in_4bit:
            quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        elif load_in_8bit:
            quant_config = BitsAndBytesConfig(load_in_8bit=True)

        logger.info(f"Loading HF model: {model_name_or_path} (device_map={device_map})")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            device_map=device_map,
            quantization_config=quant_config,
            torch_dtype=torch.bfloat16 if not (load_in_4bit or load_in_8bit) else None,
            trust_remote_code=True,
        )
        self.model.eval()
        self._embedding_model = HFLocalEmbeddings(embedding_model)

    def __call__(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,          # ignored — kept for API compat
        temperature: float = 0.8,
        max_tokens: int = 500,
        stop_strs: Optional[List[str]] = None,
        n: int = 1,
    ) -> Union[str, List[str]]:
        # Apply chat template
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        responses = []
        for _ in range(n):
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature if temperature > 0 else None,
                    do_sample=temperature > 0,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            # Decode only the newly generated tokens
            new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
            response_text = self.tokenizer.decode(new_ids, skip_special_tokens=True)

            # Apply stop strings
            if stop_strs:
                for stop in stop_strs:
                    if stop in response_text:
                        response_text = response_text[:response_text.index(stop)]

            responses.append(response_text.strip())

        return responses[0] if n == 1 else responses

    def generate_response(self, prompt: str, temperature: float = 0.8, max_tokens: int = 500) -> LLMResult:
        """Compatible with BedrockClaudeLLM.generate_response."""
        messages = [{"role": "user", "content": prompt}]
        response_text = self(messages, temperature=temperature, max_tokens=max_tokens)
        send_tokens = max(1, len(prompt) // 4)
        recv_tokens = max(1, len(response_text) // 4)
        return LLMResult(
            content=response_text,
            send_tokens=send_tokens,
            recv_tokens=recv_tokens,
            total_tokens=send_tokens + recv_tokens,
        )

    def get_embedding_model(self) -> HFLocalEmbeddings:
        return self._embedding_model


# ---------------------------------------------------------------------------
# Registry adapters — same pattern as BedrockClaudeAdapter
# ---------------------------------------------------------------------------

class HFLocalArgs(BaseModelArgs):
    model: str = Field(default="Qwen/Qwen3-8B")
    device_map: str = Field(default="auto")
    load_in_4bit: bool = Field(default=False)
    load_in_8bit: bool = Field(default=False)
    max_tokens: int = Field(default=500)
    temperature: float = Field(default=0.8)


@llm_registry.register("hf-local")
@llm_registry.register("hf-qwen3-0.6b")
@llm_registry.register("hf-qwen3-1.7b")
@llm_registry.register("hf-qwen3-4b")
@llm_registry.register("hf-qwen3-8b")
@llm_registry.register("hf-qwen3-14b")
@llm_registry.register("hf-qwen3-32b")
@llm_registry.register("hf-llama3-8b")
@llm_registry.register("hf-llama3-70b")
@llm_registry.register("hf-mistral-7b")
class HFLocalAdapter(BaseChatModel):
    """Registry adapter wrapping HFLocalLLM for use anywhere BedrockClaudeAdapter is used."""

    args: HFLocalArgs = Field(default_factory=HFLocalArgs)

    class Config:
        arbitrary_types_allowed = True

    _ALIAS_MAP = {
        "hf-local":        "Qwen/Qwen3-8B",
        "hf-qwen3-0.6b":   "Qwen/Qwen3-0.6B",
        "hf-qwen3-1.7b":   "Qwen/Qwen3-1.7B",
        "hf-qwen3-4b":     "Qwen/Qwen3-4B",
        "hf-qwen3-8b":     "Qwen/Qwen3-8B",
        "hf-qwen3-14b":    "Qwen/Qwen3-14B",
        "hf-qwen3-32b":    "Qwen/Qwen3-32B",
        "hf-llama3-8b":    "meta-llama/Llama-3.1-8B-Instruct",
        "hf-llama3-70b":   "meta-llama/Llama-3.1-70B-Instruct",
        "hf-mistral-7b":   "mistralai/Mistral-7B-Instruct-v0.3",
    }

    def __init__(self, max_retry: int = 3, **kwargs):
        args = HFLocalArgs()
        args_dict = args.dict()
        for k, v in args_dict.items():
            args_dict[k] = kwargs.pop(k, v)
        super().__init__(args=args_dict, max_retry=max_retry)

        model_id = self._ALIAS_MAP.get(args_dict["model"], args_dict["model"])
        self._llm = _get_or_load_llm(
            model_name_or_path=model_id,
            device_map=args_dict["device_map"],
            load_in_4bit=args_dict["load_in_4bit"],
            load_in_8bit=args_dict["load_in_8bit"],
        )
        logger.info(f"[LLM Init] HFLocalAdapter initialized: model={model_id}")

    def _construct_messages(self, prompt):
        if isinstance(prompt, str):
            return [{"role": "user", "content": prompt}]
        if isinstance(prompt, list) and prompt and isinstance(prompt[0], dict):
            return prompt
        return [{"role": "user", "content": str(prompt)}]

    def generate_response(self, prompt: str) -> LLMResult:
        messages = self._construct_messages(prompt)
        logger.info(f"[LLM Inference] generate_response via HF local: model={self._llm.model_name}")
        text = self._llm(messages, temperature=self.args.temperature, max_tokens=self.args.max_tokens)
        send_tokens = max(1, len(str(messages)) // 4)
        recv_tokens = max(1, len(text) // 4)
        logger.debug(f"[LLM Inference] Response received: model={self._llm.model_name}, send_tokens~{send_tokens}, recv_tokens~{recv_tokens}")
        return LLMResult(
            content=text,
            send_tokens=send_tokens,
            recv_tokens=max(1, len(text) // 4),
            total_tokens=max(1, (len(str(messages)) + len(text)) // 4),
        )

    async def agenerate_response(self, prompt) -> list:
        # Batch of prompts: list of strings or list of message-lists
        if isinstance(prompt, list) and prompt and isinstance(prompt[0], (str, list)):
            results = []
            for p in prompt:
                messages = self._construct_messages(p)
                results.append(self._llm(messages, temperature=self.args.temperature, max_tokens=self.args.max_tokens))
            return results
        messages = self._construct_messages(prompt)
        logger.info(f"[LLM Inference] agenerate_response via HF local: model={self._llm.model_name}")
        return [self._llm(messages, temperature=self.args.temperature, max_tokens=self.args.max_tokens)]

    async def agenerate_response_without_construction(self, prompts) -> list:
        responses = []
        for prompt in prompts:
            if isinstance(prompt, list) and prompt and isinstance(prompt[0], dict):
                messages = prompt
            elif isinstance(prompt, str):
                messages = [{"role": "user", "content": prompt}]
            else:
                messages = [{"role": "user", "content": str(prompt)}]
            text = self._llm(messages, temperature=self.args.temperature, max_tokens=self.args.max_tokens)
            responses.append({"choices": [{"message": {"content": text}}]})
        return responses


# ---------------------------------------------------------------------------
# Anthropic Claude direct API backend (MIT license)
# pip install anthropic
# Set ANTHROPIC_API_KEY environment variable.
# Compatible with Apache 2.0: anthropic SDK is MIT licensed.
# ---------------------------------------------------------------------------

class AnthropicLLM:
    """
    Drop-in replacement for BedrockClaudeLLM using the Anthropic API directly.
    Useful for reviewers without AWS credentials.

    Usage:
        llm = AnthropicLLM("claude-3-5-haiku-20241022")
        response = llm([{"role": "user", "content": "Hello"}])

    Supported models: claude-3-5-haiku-20241022, claude-3-5-sonnet-20241022,
                      claude-3-opus-20240229, etc.
    """

    def __init__(self, model: str = "claude-3-5-haiku-20241022",
                 temperature: float = 0.2, max_tokens: int = 2000):
        try:
            import anthropic as _anthropic
        except ImportError:
            raise ImportError("pip install anthropic")
        self._client = _anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def _call(self, messages: list) -> str:
        # Separate system message if present
        system = ""
        user_messages = []
        for m in messages:
            if m.get("role") == "system":
                system = m["content"]
            else:
                user_messages.append(m)
        kwargs = dict(model=self.model, max_tokens=self.max_tokens,
                      temperature=self.temperature, messages=user_messages)
        if system:
            kwargs["system"] = system
        response = self._client.messages.create(**kwargs)
        return response.content[0].text

    def generate_response(self, prompt) -> LLMResult:
        messages = prompt if isinstance(prompt, list) else [{"role": "user", "content": str(prompt)}]
        text = self._call(messages)
        return LLMResult(content=text,
                         send_tokens=max(1, len(str(messages)) // 4),
                         recv_tokens=max(1, len(text) // 4),
                         total_tokens=max(1, (len(str(messages)) + len(text)) // 4))

    async def agenerate_response(self, prompt) -> list:
        messages = prompt if isinstance(prompt, list) else [{"role": "user", "content": str(prompt)}]
        if messages and isinstance(messages[0], list):
            return [self._call(m) for m in messages]
        return [self._call(messages)]

    async def agenerate_response_without_construction(self, prompts) -> list:
        results = []
        for p in prompts:
            msgs = p if isinstance(p, list) else [{"role": "user", "content": str(p)}]
            results.append({"choices": [{"message": {"content": self._call(msgs)}}]})
        return results
