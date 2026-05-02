from aws_config import AWS_REGION
from typing import Dict, List, Optional, Union, Tuple
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import os
import boto3
import json
import base64
import logging
import traceback
import asyncio
from pydantic import Field

# Import the registry and base classes for integration
from . import llm_registry
from .base import BaseChatModel, BaseCompletionModel, BaseModelArgs, LLMResult

logger = logging.getLogger("websocietysimulator")

# Disable boto3/botocore INFO logging
logging.getLogger('botocore').setLevel(logging.WARNING)
logging.getLogger('boto3').setLevel(logging.WARNING)
logging.getLogger('botocore.tokens').setLevel(logging.WARNING)

def create_bedrock_client(region: str, aws_access_key_id: Optional[str] = None, 
                         aws_secret_access_key: Optional[str] = None, 
                         aws_session_token: Optional[str] = None):
    """Create a Bedrock client with optional credentials"""
    client_kwargs = {"region_name": region}
    if aws_access_key_id and aws_secret_access_key:
        client_kwargs["aws_access_key_id"] = aws_access_key_id
        client_kwargs["aws_secret_access_key"] = aws_secret_access_key
        if aws_session_token:
            client_kwargs["aws_session_token"] = aws_session_token
    
    return boto3.client("bedrock-runtime", **client_kwargs)

class LLMBase:
    """Base class for Bedrock LLM implementations"""
    def __init__(self, model: str):
        self.model = model
        
    def __call__(self, messages: List[Dict[str, str]], model: Optional[str] = None, temperature: float = 0.0, max_tokens: int = 500, stop_strs: Optional[List[str]] = None, n: int = 1) -> Union[str, List[str]]:
        """Call LLM to get response"""
        raise NotImplementedError("Subclasses need to implement this method")
    
    def get_embedding_model(self):
        """Get the embedding model for text embeddings"""
        raise NotImplementedError("Subclasses need to implement this method") 


class BedrockTitanEmbeddings:
    """Bedrock Titan Embeddings wrapper compatible with LangChain interface."""
    
    # Titan Embed v2 has 8192 token limit. With ~4 chars/token average,
    # use 30000 chars to stay safely under limit (leaves room for special tokens)
    MAX_CHARS = 30000
    
    def __init__(self, client, model: str = "amazon.titan-embed-text-v2:0"):
        self.client = client
        self.model = model
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents."""
        embeddings = []
        for text in texts:
            if len(text) > self.MAX_CHARS:
                logger.warning(f"Truncating embedding text from {len(text)} to {self.MAX_CHARS} chars")
                text = text[:self.MAX_CHARS]
            response = self.client.invoke_model(
                modelId=self.model,
                body=json.dumps({
                    "inputText": text,
                    "dimensions": 1024,
                    "normalize": True
                })
            )
            result = json.loads(response['body'].read())
            embeddings.append(result['embedding'])
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query text."""
        if len(text) > self.MAX_CHARS:
            logger.warning(f"Truncating embedding text from {len(text)} to {self.MAX_CHARS} chars")
            text = text[:self.MAX_CHARS]
        response = self.client.invoke_model(
            modelId=self.model,
            body=json.dumps({
                "inputText": text,
                "dimensions": 1024,
                "normalize": True
            })
        )
        result = json.loads(response['body'].read())
        return result['embedding']


class BedrockClaudeLLM(LLMBase):
    """
    Amazon Bedrock Claude implementation compatible with LLMBase interface.
    Supports both text-only and multimodal (image) inputs.
    """
    
    def __init__(
        self, 
        model: str = "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        embedding_model_id: str = "amazon.titan-embed-text-v2:0",
        region: str = AWS_REGION,
        return_text_only: bool = True,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        aws_session_token: Optional[str] = None,
    ):
        """
        Initialize Bedrock Claude LLM
        
        Args:
            model: Claude model ID on Bedrock
            region: AWS region for Bedrock
            aws_access_key_id: Optional AWS access key (uses default credentials if not provided)
            aws_secret_access_key: Optional AWS secret key
            aws_session_token: Optional AWS session token
            embedding_model_id: Bedrock embedding model ID
        """
        super().__init__(model)
        
        # Initialize Bedrock client
        client_kwargs = {"region_name": region}
        if aws_access_key_id and aws_secret_access_key:
            client_kwargs["aws_access_key_id"] = aws_access_key_id
            client_kwargs["aws_secret_access_key"] = aws_secret_access_key
            if aws_session_token:
                client_kwargs["aws_session_token"] = aws_session_token
        
        self.client = boto3.client("bedrock-runtime", **client_kwargs)
        self.region = region
        self.embedding_model = BedrockTitanEmbeddings(self.client, embedding_model_id)
        self.return_text_only = return_text_only
        # logger.info(f"Initialized Bedrock Claude: {model} in {region}")
    
    def _convert_messages_to_bedrock_format(
        self, 
        messages: List[Dict[str, str]]
    ) -> List[Dict]:
        """
        Convert OpenAI-style messages to Bedrock Converse API format.
        
        Args:
            messages: OpenAI-style messages [{"role": "user", "content": "text"}]
            
        Returns:
            Bedrock-formatted messages
        """
        bedrock_messages = []
        
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            
            # Map roles (assistant maps to assistant, user maps to user)
            if role == "system":
                # System messages need special handling in Bedrock
                # We'll prepend them to the first user message
                continue
            
            # Handle content - can be string or list of content blocks
            if isinstance(content, str):
                # Skip empty content - Bedrock rejects empty content blocks
                if not content or not content.strip():
                    logger.warning(f"Skipping message with empty content for role '{role}'")
                    continue
                bedrock_content = [{"text": content}]
            elif isinstance(content, list):
                # Already in multi-modal format
                bedrock_content = self._convert_content_blocks(content)
                # Filter out empty text blocks
                bedrock_content = [b for b in bedrock_content if not (isinstance(b, dict) and b.get("text", "").strip() == "")]
                if not bedrock_content:
                    logger.warning(f"Skipping message with empty content blocks for role '{role}'")
                    continue
            else:
                content_str = str(content) if content else ""
                if not content_str.strip():
                    logger.warning(f"Skipping message with empty content for role '{role}'")
                    continue
                bedrock_content = [{"text": content_str}]
            
            bedrock_messages.append({
                "role": role,
                "content": bedrock_content
            })
        
        # Handle system messages by prepending to first user message
        system_messages = [m for m in messages if m["role"] == "system"]
        if system_messages and bedrock_messages:
            system_text = "\n\n".join([m.get("content", "") for m in system_messages if m.get("content", "").strip()])
            # Prepend to first user message if we have system text
            if system_text.strip():
                for msg in bedrock_messages:
                    if msg["role"] == "user":
                        msg["content"].insert(0, {"text": f"System: {system_text}\n\n"})
                        break
        
        return bedrock_messages
    
    def _convert_content_blocks(self, content_blocks: List) -> List[Dict]:
        """Convert content blocks to Bedrock format."""
        bedrock_blocks = []
        
        for block in content_blocks:
            if isinstance(block, dict):
                if "text" in block:
                    bedrock_blocks.append({"text": block["text"]})
                elif "image_url" in block:
                    # Handle image URLs or base64 data
                    image_data = block["image_url"].get("url", "")
                    if image_data.startswith("data:image"):
                        # Extract base64 data
                        _, base64_data = image_data.split(",", 1)
                        image_bytes = base64.b64decode(base64_data)
                        # Determine format from data URL
                        format_str = image_data.split(";")[0].split("/")[1]
                        bedrock_blocks.append({
                            "image": {
                                "format": format_str,
                                "source": {"bytes": image_bytes}
                            }
                        })
                elif "image" in block:
                    # Already in Bedrock format
                    bedrock_blocks.append(block)
            elif isinstance(block, str):
                bedrock_blocks.append({"text": block})
        
        return bedrock_blocks
    
    # Bedrock Claude 3/3.5 hard limit is 200k tokens.
    # Reserve headroom for the response and overhead; cap input at 190k tokens.
    # Using 4 chars-per-token as a conservative estimate.
    _MAX_INPUT_TOKENS = 190_000
    _CHARS_PER_TOKEN = 4

    def _truncate_messages_to_token_budget(
        self,
        bedrock_messages: List[Dict],
        max_response_tokens: int = 500,
    ) -> List[Dict]:
        """
        Trim the largest text block(s) so the total estimated token count
        stays under the model's context limit.

        Strategy: estimate total chars, then proportionally trim the single
        largest text block until we're within budget.  This preserves all
        messages and only shortens the bloated content (typically the system
        role that embeds a full agent memory).
        """
        budget_chars = (self._MAX_INPUT_TOKENS - max_response_tokens) * self._CHARS_PER_TOKEN

        def _total_chars(msgs):
            total = 0
            for m in msgs:
                for block in m.get("content", []):
                    if isinstance(block, dict):
                        total += len(block.get("text", ""))
            return total

        total = _total_chars(bedrock_messages)
        if total <= budget_chars:
            return bedrock_messages

        overage = total - budget_chars
        logger.warning(
            f"[Bedrock] Prompt too long (~{total // self._CHARS_PER_TOKEN} tokens). "
            f"Trimming {overage} chars to fit within {self._MAX_INPUT_TOKENS} token budget."
        )

        # Find the single largest text block and trim it
        largest_msg_idx = None
        largest_block_idx = None
        largest_len = 0
        for mi, msg in enumerate(bedrock_messages):
            for bi, block in enumerate(msg.get("content", [])):
                if isinstance(block, dict):
                    blen = len(block.get("text", ""))
                    if blen > largest_len:
                        largest_len = blen
                        largest_msg_idx = mi
                        largest_block_idx = bi

        if largest_msg_idx is not None and largest_block_idx is not None:
            block = bedrock_messages[largest_msg_idx]["content"][largest_block_idx]
            original_text = block["text"]
            keep_chars = max(200, largest_len - overage)
            trimmed = original_text[:keep_chars] + "\n...[truncated to fit context limit]..."
            bedrock_messages[largest_msg_idx]["content"][largest_block_idx] = {"text": trimmed}
            logger.warning(
                f"[Bedrock] Trimmed message[{largest_msg_idx}] block[{largest_block_idx}] "
                f"from {largest_len} to {len(trimmed)} chars."
            )

        return bedrock_messages

    @retry(
        retry=retry_if_exception_type(Exception),
        wait=wait_exponential(multiplier=1, min=10, max=300),
        stop=stop_after_attempt(10)
    )
    def __call__(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.80,
        max_tokens: int = 500,
        stop_strs: Optional[List[str]] = None,
        n: int = 1
    ) -> Union[str, List[str]]:
        """
        Call Bedrock Claude to get response.

        Returns:
            Union[str, List[str]]: Response text from LLM, either a single string or list of strings
        """
        model = model or self.model
        bedrock_messages = self._convert_messages_to_bedrock_format(messages)
        bedrock_messages = self._truncate_messages_to_token_budget(bedrock_messages, max_tokens)
        
        # Validate that we have at least one message with content
        if not bedrock_messages:
            logger.error("No valid messages to send to Bedrock - all messages had empty content")
            raise ValueError("Cannot call Bedrock API with empty messages. All provided messages had empty content.")
        
        # Ensure the first message is from user (Bedrock requirement)
        if bedrock_messages[0]["role"] != "user":
            logger.warning("First message is not from user, prepending placeholder user message")
            bedrock_messages.insert(0, {
                "role": "user",
                "content": [{"text": "Please respond to the following:"}]
            })
        
        # Bedrock doesn't support n>1 natively, so we make multiple calls
        responses = []
        
        for _ in range(n):
            try:
                inference_config = {
                    "temperature": temperature,
                    "maxTokens": max_tokens,
                }

                if stop_strs and (cleaned := [s for s in stop_strs if isinstance(s, str) and s.strip()]):
                    inference_config["stopSequences"] = cleaned

                # Call Bedrock Converse API
                response = self.client.converse(
                    modelId=model,
                    messages=bedrock_messages,
                    inferenceConfig=inference_config
                )

                responses.append(response)

            except Exception as e:
                if "ThrottlingException" in str(e) or "429" in str(e):
                    logger.warning(f"Rate limit exceeded for Bedrock Claude: {e}")
                else:
                    logger.error(f"Bedrock Claude Error: {e}")
                traceback.format_exc()
                raise e

        if self.return_text_only:
            # return text only
            texts = [r["output"]["message"]["content"][0]["text"] for r in responses]
            return texts[0] if n == 1 else texts
        else:
            # return full metadata
            return responses[0] if n == 1 else responses
        
    
    def call_with_images(
        self,
        text_prompt: str,
        image_paths: Optional[List[str]] = None,
        image_bytes_list: Optional[List[Tuple[bytes, str]]] = None,
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 2048
    ) -> str:
        """
        Extended method for multimodal calls with images.
        
        Args:
            text_prompt: Text prompt
            image_paths: Optional list of image file paths
            image_bytes_list: Optional list of (image_bytes, format) tuples
            model: Optional model override
            temperature: Temperature for sampling
            max_tokens: Maximum tokens in response
            
        Returns:
            str: Response text from Claude
        """
        content = [{"text": text_prompt}]
        
        # Add images from paths
        if image_paths:
            for image_path in image_paths:
                img_bytes, img_format = self._load_image_bytes(image_path)
                content.append({
                    "image": {
                        "format": img_format,
                        "source": {"bytes": img_bytes}
                    }
                })
        
        # Add images from bytes
        if image_bytes_list:
            for img_bytes, img_format in image_bytes_list:
                content.append({
                    "image": {
                        "format": img_format,
                        "source": {"bytes": img_bytes}
                    }
                })
        
        # Create message in OpenAI format, but with multimodal content
        messages = [{"role": "user", "content": content}]
        
        return self(messages, model=model, temperature=temperature, max_tokens=max_tokens)
    
    def _load_image_bytes(self, image_path: str) -> Tuple[bytes, str]:
        """Load image and determine format."""
        with open(image_path, "rb") as f:
            image_data = f.read()
        
        # Determine format from extension
        ext = os.path.splitext(image_path)[1].lower()
        format_map = {
            ".jpg": "jpeg",
            ".jpeg": "jpeg",
            ".png": "png",
            ".gif": "gif",
            ".webp": "webp"
        }
        image_format = format_map.get(ext, "png")
        
        return image_data, image_format
    
    def generate_response(self, prompt: str, temperature: float = 0.8, max_tokens: int = 500) -> 'LLMResult':
        """
        Generate response compatible with BaseLLM interface.
        
        This method wraps the __call__ method to provide compatibility with
        code that expects the BaseLLM.generate_response interface (like MACF agents).
        
        Args:
            prompt: Text prompt to send to the model
            temperature: Temperature for sampling (0.0-1.0)
            max_tokens: Maximum tokens in response
            
        Returns:
            LLMResult: Response object with content and token counts
        """
        # Convert prompt to messages format
        messages = [{"role": "user", "content": prompt}]
        
        # Call the underlying __call__ method
        response_text = self(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        # Estimate token counts (Bedrock doesn't provide exact counts)
        send_tokens = max(1, len(prompt) // 4)
        recv_tokens = max(1, len(response_text) // 4)
        
        return LLMResult(
            content=response_text,
            send_tokens=send_tokens,
            recv_tokens=recv_tokens,
            total_tokens=send_tokens + recv_tokens
        )
    
    def get_embedding_model(self):
        """
        Get the Bedrock Titan embedding model
        
        Returns:
            BedrockTitanEmbeddings: Bedrock Titan embeddings instance
        """
        return self.embedding_model
    
    def extract_json(self, text: str) -> Dict:
        """
        Extract and parse JSON from response.
        Useful for structured outputs.
        """
        # Remove markdown formatting
        text = text.replace("```json", "").replace("```", "").strip()
        
        # Find JSON object
        start_idx = text.find("{")
        end_idx = text.rfind("}")
        
        if start_idx == -1 or end_idx == -1:
            raise ValueError(f"No JSON found in response: {text}")
        
        json_text = text[start_idx:end_idx+1]
        return json.loads(json_text)


# ============================================================================
# AMAZON TITAN
# ============================================================================

class BedrockTitanLLM(LLMBase):
    """Amazon Titan Text models"""
    
    def __init__(
        self,
        model: str = "amazon.titan-text-premier-v1:0",
        embedding_model_id: str = "amazon.titan-embed-text-v2:0",
        region: str = AWS_REGION,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        aws_session_token: Optional[str] = None,
    ):
        super().__init__(model)
        self.client = create_bedrock_client(region, aws_access_key_id, aws_secret_access_key, aws_session_token)
        self.region = region
        self.embedding_model = BedrockTitanEmbeddings(self.client, embedding_model_id)
        logger.info(f"Initialized Bedrock Titan: {model} in {region}")
    
    def _messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Convert messages to Titan prompt format"""
        prompt_parts = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
        return "\n\n".join(prompt_parts) + "\n\nAssistant:"
    
    @retry(
        retry=retry_if_exception_type(Exception),
        wait=wait_exponential(multiplier=1, min=10, max=300),
        stop=stop_after_attempt(10)
    )
    def __call__(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 500,
        stop_strs: Optional[List[str]] = None,
        n: int = 1
    ) -> Union[str, List[str]]:
        model = model or self.model
        prompt = self._messages_to_prompt(messages)
        
        responses = []
        
        for _ in range(n):
            try:
                request_body = {
                    "inputText": prompt,
                    "textGenerationConfig": {
                        "maxTokenCount": max_tokens,
                        "temperature": temperature,
                        "topP": 0.9
                    }
                }
                
                if stop_strs:
                    request_body["textGenerationConfig"]["stopSequences"] = stop_strs
                
                response = self.client.invoke_model(
                    modelId=model,
                    body=json.dumps(request_body)
                )
                
                response_body = json.loads(response['body'].read())
                response_text = response_body['results'][0]['outputText']
                responses.append(response_text)
                
            except Exception as e:
                if "ThrottlingException" in str(e) or "429" in str(e):
                    logger.warning(f"Rate limit exceeded for Bedrock Titan: {e}")
                else:
                    logger.error(f"Bedrock Titan Error: {e}")
                raise e
        
        return responses[0] if n == 1 else responses
    
    def get_embedding_model(self):
        return self.embedding_model

# ============================================================================
# DEEPSEEK
# ============================================================================

class BedrockDeepSeekLLM(LLMBase):
    """
    DeepSeek models (R1, V3.1) on Amazon Bedrock
    "us.deepseek.r1-v1:0"
    """
    
    def __init__(
        self,
        model: str = "us.deepseek.r1-v1:0",
        embedding_model_id: str = "amazon.titan-embed-text-v2:0",
        region: str = AWS_REGION,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        aws_session_token: Optional[str] = None,
    ):
        """
        Initialize Bedrock DeepSeek LLM
        
        Args:
            model: DeepSeek model ID on Bedrock (default: deepseek-r1)
            embedding_model_id: Bedrock embedding model ID
            region: AWS region for Bedrock
            aws_access_key_id: Optional AWS access key
            aws_secret_access_key: Optional AWS secret key
            aws_session_token: Optional AWS session token
        """
        super().__init__(model)
        
        # Initialize Bedrock client
        client_kwargs = {"region_name": region}
        if aws_access_key_id and aws_secret_access_key:
            client_kwargs["aws_access_key_id"] = aws_access_key_id
            client_kwargs["aws_secret_access_key"] = aws_secret_access_key
            if aws_session_token:
                client_kwargs["aws_session_token"] = aws_session_token
        
        self.client = boto3.client("bedrock-runtime", **client_kwargs)
        self.region = region
        self.embedding_model = BedrockTitanEmbeddings(self.client, embedding_model_id)
        logger.info(f"Initialized Bedrock DeepSeek: {model} in {region}")
    
    def _convert_messages_to_bedrock_format(
        self, 
        messages: List[Dict[str, str]]
    ) -> Tuple[Optional[str], List[Dict]]:
        """Convert OpenAI-style messages to Bedrock Converse API format"""
        bedrock_messages = []
        system_message = None
        
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            
            if role == "system":
                system_message = content if isinstance(content, str) else str(content)
                continue
            
            if isinstance(content, str):
                bedrock_content = [{"text": content}]
            elif isinstance(content, list):
                bedrock_content = []
                for block in content:
                    if isinstance(block, dict) and "text" in block:
                        bedrock_content.append({"text": block["text"]})
                    elif isinstance(block, str):
                        bedrock_content.append({"text": block})
            else:
                bedrock_content = [{"text": str(content)}]
            
            bedrock_messages.append({
                "role": role,
                "content": bedrock_content
            })
        
        return system_message, bedrock_messages
    
    @retry(
        retry=retry_if_exception_type(Exception),
        wait=wait_exponential(multiplier=1, min=10, max=300),
        stop=stop_after_attempt(10)
    )
    def __call__(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 500,
        stop_strs: Optional[List[str]] = None,
        n: int = 1
    ) -> Union[str, List[str]]:
        """
        Call Bedrock DeepSeek to get response
        
        Args:
            messages: List of input messages
            model: Optional model override
            temperature: Temperature for sampling (0.0-1.0)
            max_tokens: Maximum tokens in response
            stop_strs: Optional list of stop strings
            n: Number of responses to generate
            
        Returns:
            Union[str, List[str]]: Response text from LLM
        """
        model = model or self.model
        system_message, bedrock_messages = self._convert_messages_to_bedrock_format(messages)
        
        responses = []
        
        for _ in range(n):
            try:
                inference_config = {
                    "temperature": temperature,
                    "maxTokens": max_tokens,
                }
                
                if stop_strs:
                    inference_config["stopSequences"] = stop_strs
                
                request_params = {
                    "modelId": model,
                    "messages": bedrock_messages,
                    "inferenceConfig": inference_config
                }
                
                if system_message:
                    request_params["system"] = [{"text": system_message}]
                
                response = self.client.converse(**request_params)
                response_text = response["output"]["message"]["content"][0]["text"]
                responses.append(response_text)
                
            except Exception as e:
                if "ThrottlingException" in str(e) or "429" in str(e):
                    logger.warning(f"Rate limit exceeded for Bedrock DeepSeek: {e}")
                else:
                    logger.error(f"Bedrock DeepSeek Error: {e}")
                raise e
        
        return responses[0] if n == 1 else responses
    
    def get_embedding_model(self):
        return self.embedding_model


# ============================================================================
# QWEN
# ============================================================================

class BedrockQwenLLM(LLMBase):
    """
    Qwen models on Amazon Bedrock (fully managed, serverless as of Sep 2025)
    qwen.qwen3-32b-v1:0                   (Dense, cheapest, ~$0.20/$0.60 per 1M tokens)
    qwen.qwen3-235b-a22b-2507-v1:0        (MoE flagship)
    qwen.qwen3-coder-30b-a3b-v1:0         (Coder MoE, small)
    qwen.qwen3-coder-480b-a35b-v1:0       (Coder MoE, large)
    """
    
    def __init__(
        self,
        model: str = "qwen.qwen3-32b-v1:0",
        embedding_model_id: str = "amazon.titan-embed-text-v2:0",
        region: str = AWS_REGION,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        aws_session_token: Optional[str] = None,
    ):
        """
        Initialize Bedrock Qwen LLM
        
        Args:
            model: Qwen model ID on Bedrock (default: Qwen3-235B)
            embedding_model_id: Bedrock embedding model ID
            region: AWS region for Bedrock
            aws_access_key_id: Optional AWS access key
            aws_secret_access_key: Optional AWS secret key
            aws_session_token: Optional AWS session token
        """
        super().__init__(model)
        
        # Initialize Bedrock client
        client_kwargs = {"region_name": region}
        if aws_access_key_id and aws_secret_access_key:
            client_kwargs["aws_access_key_id"] = aws_access_key_id
            client_kwargs["aws_secret_access_key"] = aws_secret_access_key
            if aws_session_token:
                client_kwargs["aws_session_token"] = aws_session_token
        
        self.client = boto3.client("bedrock-runtime", **client_kwargs)
        self.region = region
        self.embedding_model = BedrockTitanEmbeddings(self.client, embedding_model_id)
        logger.info(f"Initialized Bedrock Qwen: {model} in {region}")
    
    def _convert_messages_to_bedrock_format(
        self, 
        messages: List[Dict[str, str]]
    ) -> Tuple[Optional[str], List[Dict]]:
        """Convert OpenAI-style messages to Bedrock Converse API format"""
        bedrock_messages = []
        system_message = None
        
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            
            if role == "system":
                system_message = content if isinstance(content, str) else str(content)
                continue
            
            if isinstance(content, str):
                bedrock_content = [{"text": content}]
            elif isinstance(content, list):
                bedrock_content = []
                for block in content:
                    if isinstance(block, dict) and "text" in block:
                        bedrock_content.append({"text": block["text"]})
                    elif isinstance(block, str):
                        bedrock_content.append({"text": block})
            else:
                bedrock_content = [{"text": str(content)}]
            
            bedrock_messages.append({
                "role": role,
                "content": bedrock_content
            })
        
        return system_message, bedrock_messages
    
    @retry(
        retry=retry_if_exception_type(Exception),
        wait=wait_exponential(multiplier=1, min=10, max=300),
        stop=stop_after_attempt(10)
    )
    def __call__(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 500,
        stop_strs: Optional[List[str]] = None,
        n: int = 1
    ) -> Union[str, List[str]]:
        """
        Call Bedrock Qwen to get response
        
        Args:
            messages: List of input messages
            model: Optional model override
            temperature: Temperature for sampling (0.0-1.0)
            max_tokens: Maximum tokens in response
            stop_strs: Optional list of stop strings
            n: Number of responses to generate
            
        Returns:
            Union[str, List[str]]: Response text from LLM
        """
        model = model or self.model
        system_message, bedrock_messages = self._convert_messages_to_bedrock_format(messages)
        
        responses = []
        
        for _ in range(n):
            try:
                inference_config = {
                    "temperature": temperature,
                    "maxTokens": max_tokens,
                }
                
                if stop_strs and (cleaned := [s for s in stop_strs if isinstance(s, str) and s.strip()]):
                    inference_config["stopSequences"] = cleaned
                
                request_params = {
                    "modelId": model,
                    "messages": bedrock_messages,
                    "inferenceConfig": inference_config
                }
                
                if system_message:
                    request_params["system"] = [{"text": system_message}]
                
                response = self.client.converse(**request_params)
                response_text = response["output"]["message"]["content"][0]["text"]
                responses.append(response_text)
                
            except Exception as e:
                if "ThrottlingException" in str(e) or "429" in str(e):
                    logger.warning(f"Rate limit exceeded for Bedrock Qwen: {e}")
                else:
                    logger.error(f"Bedrock Qwen Error: {e}")
                raise e
        
        return responses[0] if n == 1 else responses
    
    def get_embedding_model(self):
        return self.embedding_model

# ============================================================================
# META LLAMA
# ============================================================================

class BedrockLlamaLLM(LLMBase):
    """Meta Llama models (Llama 2, Llama 3, etc.)"""
    
    def __init__(
        self,
        model: str = "meta.llama3-70b-instruct-v1:0",
        embedding_model_id: str = "amazon.titan-embed-text-v2:0",
        region: str = AWS_REGION,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        aws_session_token: Optional[str] = None,
    ):
        super().__init__(model)
        self.client = create_bedrock_client(region, aws_access_key_id, aws_secret_access_key, aws_session_token)
        self.region = region
        self.embedding_model = BedrockTitanEmbeddings(self.client, embedding_model_id)
        logger.info(f"Initialized Bedrock Llama: {model} in {region}")
    
    def _format_llama_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Format messages in Llama chat format"""
        prompt = "<|begin_of_text|>"
        
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            
            if role == "system":
                prompt += f"<|start_header_id|>system<|end_header_id|>\n\n{content}<|eot_id|>"
            elif role == "user":
                prompt += f"<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|>"
            elif role == "assistant":
                prompt += f"<|start_header_id|>assistant<|end_header_id|>\n\n{content}<|eot_id|>"
        
        prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n"
        return prompt
    
    @retry(
        retry=retry_if_exception_type(Exception),
        wait=wait_exponential(multiplier=1, min=10, max=300),
        stop=stop_after_attempt(10)
    )
    def __call__(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 500,
        stop_strs: Optional[List[str]] = None,
        n: int = 1
    ) -> Union[str, List[str]]:
        model = model or self.model
        prompt = self._format_llama_prompt(messages)
        
        responses = []
        
        for _ in range(n):
            try:
                request_body = {
                    "prompt": prompt,
                    "max_gen_len": max_tokens,
                    "temperature": temperature,
                    "top_p": 0.9
                }
                
                response = self.client.invoke_model(
                    modelId=model,
                    body=json.dumps(request_body)
                )
                
                response_body = json.loads(response['body'].read())
                response_text = response_body['generation']
                responses.append(response_text)
                
            except Exception as e:
                if "ThrottlingException" in str(e) or "429" in str(e):
                    logger.warning(f"Rate limit exceeded for Bedrock Llama: {e}")
                else:
                    logger.error(f"Bedrock Llama Error: {e}")
                raise e
        
        return responses[0] if n == 1 else responses
    
    def get_embedding_model(self):
        return self.embedding_model


# ============================================================================
# MISTRAL AI
# ============================================================================

class BedrockMistralLLM(LLMBase):
    """Mistral AI models"""
    
    def __init__(
        self,
        model: str = "mistral.mistral-large-2402-v1:0",
        embedding_model_id: str = "amazon.titan-embed-text-v2:0",
        region: str = AWS_REGION,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        aws_session_token: Optional[str] = None,
    ):
        super().__init__(model)
        self.client = create_bedrock_client(region, aws_access_key_id, aws_secret_access_key, aws_session_token)
        self.region = region
        self.embedding_model = BedrockTitanEmbeddings(self.client, embedding_model_id)
        logger.info(f"Initialized Bedrock Mistral: {model} in {region}")
    
    def _format_mistral_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Format messages in Mistral instruction format"""
        prompt = "<s>"
        
        for i, msg in enumerate(messages):
            role = msg["role"]
            content = msg["content"]
            
            if role == "system":
                prompt += f"[INST] {content} [/INST]"
            elif role == "user":
                if i > 0 and messages[i-1]["role"] == "assistant":
                    prompt += f"[INST] {content} [/INST]"
                else:
                    prompt += f"[INST] {content} [/INST]"
            elif role == "assistant":
                prompt += f" {content}</s>"
        
        return prompt
    
    @retry(
        retry=retry_if_exception_type(Exception),
        wait=wait_exponential(multiplier=1, min=10, max=300),
        stop=stop_after_attempt(10)
    )
    def __call__(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 500,
        stop_strs: Optional[List[str]] = None,
        n: int = 1
    ) -> Union[str, List[str]]:
        model = model or self.model
        prompt = self._format_mistral_prompt(messages)
        
        responses = []
        
        for _ in range(n):
            try:
                request_body = {
                    "prompt": prompt,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "top_p": 0.9
                }
                
                response = self.client.invoke_model(
                    modelId=model,
                    body=json.dumps(request_body)
                )
                
                response_body = json.loads(response['body'].read())
                response_text = response_body['outputs'][0]['text']
                responses.append(response_text)
                
            except Exception as e:
                if "ThrottlingException" in str(e) or "429" in str(e):
                    logger.warning(f"Rate limit exceeded for Bedrock Mistral: {e}")
                else:
                    logger.error(f"Bedrock Mistral Error: {e}")
                raise e
        
        return responses[0] if n == 1 else responses
    
    def get_embedding_model(self):
        return self.embedding_model


# ============================================================================
# COHERE COMMAND
# ============================================================================

class BedrockCohereLLM(LLMBase):
    """Cohere Command models"""
    
    def __init__(
        self,
        model: str = "cohere.command-text-v14",
        embedding_model_id: str = "amazon.titan-embed-text-v2:0",
        region: str = AWS_REGION,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        aws_session_token: Optional[str] = None,
    ):
        super().__init__(model)
        self.client = create_bedrock_client(region, aws_access_key_id, aws_secret_access_key, aws_session_token)
        self.region = region
        self.embedding_model = BedrockTitanEmbeddings(self.client, embedding_model_id)
        logger.info(f"Initialized Bedrock Cohere: {model} in {region}")
    
    def _messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Convert messages to Cohere prompt format"""
        prompt_parts = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Chatbot: {content}")
        return "\n\n".join(prompt_parts) + "\n\nChatbot:"
    
    @retry(
        retry=retry_if_exception_type(Exception),
        wait=wait_exponential(multiplier=1, min=10, max=300),
        stop=stop_after_attempt(10)
    )
    def __call__(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 500,
        stop_strs: Optional[List[str]] = None,
        n: int = 1
    ) -> Union[str, List[str]]:
        model = model or self.model
        prompt = self._messages_to_prompt(messages)
        
        responses = []
        
        for _ in range(n):
            try:
                request_body = {
                    "prompt": prompt,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "p": 0.75,
                    "k": 0
                }
                
                if stop_strs:
                    request_body["stop_sequences"] = stop_strs
                
                response = self.client.invoke_model(
                    modelId=model,
                    body=json.dumps(request_body)
                )
                
                response_body = json.loads(response['body'].read())
                response_text = response_body['generations'][0]['text']
                responses.append(response_text)
                
            except Exception as e:
                if "ThrottlingException" in str(e) or "429" in str(e):
                    logger.warning(f"Rate limit exceeded for Bedrock Cohere: {e}")
                else:
                    logger.error(f"Bedrock Cohere Error: {e}")
                raise e
        
        return responses[0] if n == 1 else responses
    
    def get_embedding_model(self):
        return self.embedding_model


# ============================================================================
# AI21 LABS JURASSIC
# ============================================================================

class BedrockAI21LLM(LLMBase):
    """AI21 Labs Jurassic models"""
    
    def __init__(
        self,
        model: str = "ai21.j2-ultra-v1",
        embedding_model_id: str = "amazon.titan-embed-text-v2:0",
        region: str = AWS_REGION,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        aws_session_token: Optional[str] = None,
    ):
        super().__init__(model)
        self.client = create_bedrock_client(region, aws_access_key_id, aws_secret_access_key, aws_session_token)
        self.region = region
        self.embedding_model = BedrockTitanEmbeddings(self.client, embedding_model_id)
        logger.info(f"Initialized Bedrock AI21: {model} in {region}")
    
    def _messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Convert messages to AI21 prompt format"""
        prompt_parts = []
        for msg in messages:
            role = msg["role"].capitalize()
            content = msg["content"]
            prompt_parts.append(f"{role}: {content}")
        return "\n\n".join(prompt_parts) + "\n\nAssistant:"
    
    @retry(
        retry=retry_if_exception_type(Exception),
        wait=wait_exponential(multiplier=1, min=10, max=300),
        stop=stop_after_attempt(10)
    )
    def __call__(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 500,
        stop_strs: Optional[List[str]] = None,
        n: int = 1
    ) -> Union[str, List[str]]:
        model = model or self.model
        prompt = self._messages_to_prompt(messages)
        
        responses = []
        
        for _ in range(n):
            try:
                request_body = {
                    "prompt": prompt,
                    "maxTokens": max_tokens,
                    "temperature": temperature,
                    "topP": 1.0
                }
                
                if stop_strs:
                    request_body["stopSequences"] = stop_strs
                
                response = self.client.invoke_model(
                    modelId=model,
                    body=json.dumps(request_body)
                )
                
                response_body = json.loads(response['body'].read())
                response_text = response_body['completions'][0]['data']['text']
                responses.append(response_text)
                
            except Exception as e:
                if "ThrottlingException" in str(e) or "429" in str(e):
                    logger.warning(f"Rate limit exceeded for Bedrock AI21: {e}")
                else:
                    logger.error(f"Bedrock AI21 Error: {e}")
                raise e
        
        return responses[0] if n == 1 else responses
    
    def get_embedding_model(self):
        return self.embedding_model


# ============================================================================
# AMAZON NOVA
# ============================================================================

class BedrockNovaLLM(LLMBase):
    """Amazon Nova models (Pro, Lite, Micro)"""
    
    def __init__(
        self,
        model: str = "amazon.nova-pro-v1:0",
        embedding_model_id: str = "amazon.titan-embed-text-v2:0",
        region: str = AWS_REGION,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        aws_session_token: Optional[str] = None,
    ):
        super().__init__(model)
        self.client = create_bedrock_client(region, aws_access_key_id, aws_secret_access_key, aws_session_token)
        self.region = region
        self.embedding_model = BedrockTitanEmbeddings(self.client, embedding_model_id)
        logger.info(f"Initialized Bedrock Nova: {model} in {region}")
    
    def _convert_messages_to_bedrock_format(
        self, 
        messages: List[Dict[str, str]]
    ) -> Tuple[Optional[str], List[Dict]]:
        """Convert OpenAI-style messages to Bedrock Converse API format for Nova"""
        bedrock_messages = []
        system_message = None
        
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            
            if role == "system":
                system_message = content if isinstance(content, str) else str(content)
                continue
            
            if isinstance(content, str):
                bedrock_content = [{"text": content}]
            elif isinstance(content, list):
                bedrock_content = []
                for block in content:
                    if isinstance(block, dict) and "text" in block:
                        bedrock_content.append({"text": block["text"]})
                    elif isinstance(block, str):
                        bedrock_content.append({"text": block})
            else:
                bedrock_content = [{"text": str(content)}]
            
            bedrock_messages.append({
                "role": role,
                "content": bedrock_content
            })
        
        return system_message, bedrock_messages
    
    @retry(
        retry=retry_if_exception_type(Exception),
        wait=wait_exponential(multiplier=1, min=10, max=300),
        stop=stop_after_attempt(10)
    )
    def __call__(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 500,
        stop_strs: Optional[List[str]] = None,
        n: int = 1
    ) -> Union[str, List[str]]:
        model = model or self.model
        system_message, bedrock_messages = self._convert_messages_to_bedrock_format(messages)
        
        responses = []
        
        for _ in range(n):
            try:
                inference_config = {
                    "temperature": temperature,
                    "maxTokens": max_tokens,
                }
                
                if stop_strs:
                    inference_config["stopSequences"] = stop_strs
                
                request_params = {
                    "modelId": model,
                    "messages": bedrock_messages,
                    "inferenceConfig": inference_config
                }
                
                if system_message:
                    request_params["system"] = [{"text": system_message}]
                
                response = self.client.converse(**request_params)
                response_text = response["output"]["message"]["content"][0]["text"]
                responses.append(response_text)
                
            except Exception as e:
                if "ThrottlingException" in str(e) or "429" in str(e):
                    logger.warning(f"Rate limit exceeded for Bedrock Nova: {e}")
                else:
                    logger.error(f"Bedrock Nova Error: {e}")
                raise e
        
        return responses[0] if n == 1 else responses
    
    def get_embedding_model(self):
        return self.embedding_model

# ============================================================================
# ADAPTER CLASSES FOR CONNACF INTEGRATION
# ============================================================================

class BedrockClaudeArgs(BaseModelArgs):
    """Arguments for Bedrock Claude models"""
    model: str = Field(default="us.anthropic.claude-sonnet-4-5-20250929-v1:0")
    region: str = Field(default=AWS_REGION)
    embedding_model_id: str = Field(default="amazon.titan-embed-text-v2:0")
    max_tokens: int = Field(default=2048)
    temperature: float = Field(default=0.2)
    aws_access_key_id: Optional[str] = Field(default=None)
    aws_secret_access_key: Optional[str] = Field(default=None)
    aws_session_token: Optional[str] = Field(default=None)


@llm_registry.register("bedrock-claude")
@llm_registry.register("bedrock-nova")
@llm_registry.register("amazon.nova-2-sonic-v1:0")
# Nova (cheap)
@llm_registry.register("amazon.nova-micro-v1:0")
@llm_registry.register("amazon.nova-lite-v1:0")
@llm_registry.register("amazon.nova-pro-v1:0")
@llm_registry.register("us.amazon.nova-micro-v1:0")
@llm_registry.register("us.amazon.nova-lite-v1:0")
@llm_registry.register("us.amazon.nova-pro-v1:0")
# Qwen3 (fully managed on Bedrock since Sep 2025)
@llm_registry.register("bedrock-qwen")
@llm_registry.register("qwen.qwen3-32b-v1:0")
@llm_registry.register("qwen.qwen3-235b-a22b-2507-v1:0")
@llm_registry.register("qwen.qwen3-coder-30b-a3b-v1:0")
@llm_registry.register("qwen.qwen3-coder-480b-a35b-v1:0")
# Meta Llama (cheap)
@llm_registry.register("meta.llama3-1-8b-instruct-v1:0")
@llm_registry.register("meta.llama3-1-70b-instruct-v1:0")
@llm_registry.register("us.meta.llama4-maverick-17b-instruct-v1:0")
@llm_registry.register("bedrock-llama4")
# Mistral (cheap)
@llm_registry.register("mistral.mistral-small-2402-v1:0")
@llm_registry.register("mistral.mixtral-8x7b-instruct-v0:1")
# DeepSeek
@llm_registry.register("us.deepseek.r1-v1:0")
@llm_registry.register("deepseek.v3-v1:0")
# Claude 4 Series
@llm_registry.register("us.anthropic.claude-sonnet-4-20250514-v1:0")
@llm_registry.register("us.anthropic.claude-haiku-4-5-20251001-v1:0")
@llm_registry.register("us.anthropic.claude-sonnet-4-6")
@llm_registry.register("us.anthropic.claude-opus-4-6-v1")
@llm_registry.register("us.anthropic.claude-sonnet-4-5-20250929-v1:0")
@llm_registry.register("global.us.anthropic.claude-sonnet-4-5-20250929-v1:0")
@llm_registry.register("us.anthropic.claude-opus-4-1-20250805-v1:0")
@llm_registry.register("us.anthropic.claude-opus-4-5-20251101-v1:0")
# Claude 3 Series
@llm_registry.register("us.anthropic.claude-3-5-sonnet-20240620-v1:0")
@llm_registry.register("us.anthropic.claude-3-sonnet-20240229-v1:0")
@llm_registry.register("us.anthropic.claude-3-haiku-20240307-v1:0")
@llm_registry.register("us.anthropic.claude-3-5-haiku-20241022-v1:0")
class BedrockClaudeAdapter(BaseChatModel):
    """Adapter to integrate BedrockClaudeLLM with ConnaCF's LLM interface"""
    
    args: BedrockClaudeArgs = Field(default_factory=BedrockClaudeArgs)
    
    class Config:
        arbitrary_types_allowed = True
    
    def _map_model_name(self, model_name: str) -> str:
        """Map generic model names to specific Bedrock model identifiers"""
        model_mapping = {
            "bedrock-claude": "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
            "bedrock-claude-sonnet": "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
            "bedrock-nova": "us.amazon.nova-pro-v1:0",
            # Qwen aliases
            "bedrock-qwen": "qwen.qwen3-32b-v1:0",
            # Llama 4
            "bedrock-llama4": "us.meta.llama4-maverick-17b-instruct-v1:0",
        }
        return model_mapping.get(model_name, model_name)
    
    def __init__(self, max_retry: int = 3, **kwargs):
        # Parse arguments similar to OpenAI implementation
        args = BedrockClaudeArgs()
        args_dict = args.dict()
        
        for k, v in args_dict.items():
            args_dict[k] = kwargs.pop(k, v)
        
        super().__init__(args=args_dict, max_retry=max_retry)
        
        # Map the model name to actual Bedrock identifier
        actual_model = self._map_model_name(args_dict['model'])
        
        # Initialize the underlying Bedrock Claude LLM
        self._bedrock_llm = BedrockClaudeLLM(
            model=actual_model,
            region=args_dict['region'],
            embedding_model_id=args_dict['embedding_model_id'],
            aws_access_key_id=args_dict['aws_access_key_id'],
            aws_secret_access_key=args_dict['aws_secret_access_key'],
            aws_session_token=args_dict['aws_session_token']
        )
        print(f"  ┌─ Bedrock LLM ready ──────────────────────────────")
        print(f"  │  model  : {actual_model}")
        print(f"  │  region : {args_dict['region']}")
        print(f"  └──────────────────────────────────────────────────")
        logger.info(f"[LLM Init] BedrockClaudeAdapter initialized: model={actual_model}, region={args_dict['region']}")
    
    def _construct_messages(self, prompt):
        """Convert prompt to messages format"""
        if isinstance(prompt, str):
            return [{"role": "user", "content": prompt}]
        elif isinstance(prompt, list):
            # Already in messages format or list of prompts
            if len(prompt) > 0 and isinstance(prompt[0], dict):
                return prompt  # Already messages
            else:
                # List of prompts
                return [[{"role": "user", "content": p}] for p in prompt]
        return [{"role": "user", "content": str(prompt)}]
    
    def generate_response(self, prompt: str) -> LLMResult:
        """Synchronous response generation"""
        messages = self._construct_messages(prompt)
        model = self._bedrock_llm.model
        logger.info(f"[LLM Inference] generate_response via Bedrock: model={model}")
        
        try:
            response_text = self._bedrock_llm(
                messages=messages,
                temperature=getattr(self.args, 'temperature', 0.2),
                max_tokens=getattr(self.args, 'max_tokens', 2048)
            )
            
            # Estimate token counts (Bedrock doesn't provide exact counts)
            send_tokens = self._estimate_tokens(str(messages))
            recv_tokens = self._estimate_tokens(response_text)
            
            logger.debug(f"[LLM Inference] Response received: model={model}, send_tokens~{send_tokens}, recv_tokens~{recv_tokens}")
            return LLMResult(
                content=response_text,
                send_tokens=send_tokens,
                recv_tokens=recv_tokens,
                total_tokens=send_tokens + recv_tokens
            )
        except Exception as e:
            logger.error(f"Bedrock Claude error: {e}")
            raise
    
    async def agenerate_response(self, prompt) -> Union[LLMResult, List]:
        """Asynchronous response generation"""
        messages = self._construct_messages(prompt)
        model = self._bedrock_llm.model
        
        # Handle batch processing like OpenAI implementation
        if isinstance(messages[0], list):
            # Batch of message lists
            logger.info(f"[LLM Inference] agenerate_response (batch={len(messages)}) via Bedrock: model={model}")
            responses = []
            for msg_list in messages:
                try:
                    response_text = self._bedrock_llm(
                        messages=msg_list,
                        temperature=getattr(self.args, 'temperature', 0.2),
                        max_tokens=getattr(self.args, 'max_tokens', 2048)
                    )
                    responses.append(response_text)
                except Exception as e:
                    logger.error(f"Bedrock Claude batch error: {e}")
                    raise
            return responses
        else:
            # Single message list
            logger.info(f"[LLM Inference] agenerate_response via Bedrock: model={model}")
            try:
                response_text = self._bedrock_llm(
                    messages=messages,
                    temperature=getattr(self.args, 'temperature', 0.2),
                    max_tokens=getattr(self.args, 'max_tokens', 2048)
                )
                return [response_text]
            except Exception as e:
                logger.error(f"Bedrock Claude async error: {e}")
                raise
    
    async def agenerate_response_without_construction(self, prompts) -> List:
        """Asynchronous response generation without message construction for batch processing"""
        responses = []
        
        for prompt in prompts:
            try:
                # Handle different prompt formats
                if isinstance(prompt, list) and len(prompt) > 0:
                    # Already in messages format
                    if isinstance(prompt[0], dict) and 'role' in prompt[0]:
                        messages = prompt
                    else:
                        # List of strings - filter out empty ones
                        messages = [{"role": "user", "content": str(p)} for p in prompt if str(p).strip()]
                elif isinstance(prompt, dict) and 'role' in prompt:
                    # Single message dict
                    messages = [prompt]
                elif isinstance(prompt, str):
                    # Plain string - check if empty
                    if not prompt.strip():
                        logger.warning("Skipping empty prompt in batch processing")
                        responses.append({"choices": [{"message": {"content": ""}}]})
                        continue
                    messages = [{"role": "user", "content": prompt}]
                else:
                    # Fallback - convert to string
                    content = str(prompt) if prompt else ""
                    if not content.strip():
                        logger.warning("Skipping empty prompt in batch processing")
                        responses.append({"choices": [{"message": {"content": ""}}]})
                        continue
                    messages = [{"role": "user", "content": content}]
                
                # Filter out messages with empty content
                messages = [m for m in messages if m.get("content", "").strip()]
                if not messages:
                    logger.warning("All messages in prompt were empty, skipping")
                    responses.append({"choices": [{"message": {"content": ""}}]})
                    continue
                
                response_text = self._bedrock_llm(
                    messages=messages,
                    temperature=getattr(self.args, 'temperature', 0.2),
                    max_tokens=getattr(self.args, 'max_tokens', 2048)
                )
                
                # Format response to match OpenAI-style structure expected by the parser
                formatted_response = {
                    "choices": [{
                        "message": {
                            "content": response_text
                        }
                    }]
                }
                responses.append(formatted_response)
            except TypeError as te:
                logger.error(f"Bedrock Claude TypeError in batch processing: {te}")
                logger.error(f"Prompt type: {type(prompt)}, Prompt value: {prompt}")
                raise
            except Exception as e:
                logger.error(f"Bedrock Claude batch error: {e}")
                raise
        
        return responses
    
    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimation (4 chars ≈ 1 token for English)"""
        return max(1, len(text) // 4)


class BedrockEmbeddingArgs(BaseModelArgs):
    """Arguments for Bedrock embedding models"""
    model: str = Field(default="amazon.titan-embed-text-v2:0")
    region: str = Field(default=AWS_REGION)
    aws_access_key_id: Optional[str] = Field(default=None)
    aws_secret_access_key: Optional[str] = Field(default=None)
    aws_session_token: Optional[str] = Field(default=None)


@llm_registry.register("bedrock-embedding")
@llm_registry.register("amazon.titan-embed-text-v2:0")
class BedrockEmbeddingAdapter(BaseCompletionModel):
    """Adapter for Bedrock embedding models"""
    
    args: BedrockEmbeddingArgs = Field(default_factory=BedrockEmbeddingArgs)
    
    class Config:
        arbitrary_types_allowed = True
    
    def _map_model_name(self, model_name: str) -> str:
        """Map generic model names to specific Bedrock model identifiers"""
        model_mapping = {
            "bedrock-embedding": "amazon.titan-embed-text-v2:0",
            "bedrock-titan-embedding": "amazon.titan-embed-text-v2:0",
        }
        return model_mapping.get(model_name, model_name)
    
    def __init__(self, max_retry: int = 3, **kwargs):
        args = BedrockEmbeddingArgs()
        args_dict = args.dict()
        
        for k, v in args_dict.items():
            args_dict[k] = kwargs.pop(k, v)
        
        super().__init__(args=args_dict, max_retry=max_retry)
        
        # Map the model name to actual Bedrock identifier
        actual_model = self._map_model_name(args_dict['model'])
        
        # Initialize Bedrock client and embedding model
        client_kwargs = {"region_name": args_dict['region']}
        if args_dict['aws_access_key_id'] and args_dict['aws_secret_access_key']:
            client_kwargs["aws_access_key_id"] = args_dict['aws_access_key_id']
            client_kwargs["aws_secret_access_key"] = args_dict['aws_secret_access_key']
            if args_dict['aws_session_token']:
                client_kwargs["aws_session_token"] = args_dict['aws_session_token']
        
        self._client = boto3.client("bedrock-runtime", **client_kwargs)
        self._embedding_model = BedrockTitanEmbeddings(self._client, actual_model)
    
    def generate_response(self, prompt: str) -> LLMResult:
        """Generate embedding for a single text"""
        try:
            embedding = self._embedding_model.embed_query(prompt)
            return LLMResult(
                content=embedding,
                send_tokens=self._estimate_tokens(prompt),
                recv_tokens=0,  # Embeddings don't generate text
                total_tokens=self._estimate_tokens(prompt)
            )
        except Exception as e:
            logger.error(f"Bedrock embedding error: {e}")
            raise
    
    async def agenerate_response(self, sentences) -> List:
        """Generate embeddings for multiple texts"""
        try:
            if isinstance(sentences, str):
                sentences = [sentences]
            
            # Process embeddings (Bedrock doesn't have async, so we simulate)
            embeddings = []
            for sentence in sentences:
                embedding = self._embedding_model.embed_query(sentence)
                embeddings.append({
                    'data': [{'embedding': embedding}],
                    'usage': {'prompt_tokens': self._estimate_tokens(sentence)}
                })
            
            return embeddings
        except Exception as e:
            logger.error(f"Bedrock embedding batch error: {e}")
            raise
    
    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimation"""
        return max(1, len(text) // 4)


# Additional model adapters for other Bedrock models
class BedrockTitanArgs(BedrockClaudeArgs):
    model: str = Field(default="amazon.titan-text-premier-v1:0")


@llm_registry.register("bedrock-titan")
@llm_registry.register("amazon.titan-text-premier-v1:0")
class BedrockTitanAdapter(BaseChatModel):
    """Adapter for Bedrock Titan models"""
    
    args: BedrockTitanArgs = Field(default_factory=BedrockTitanArgs)
    
    class Config:
        arbitrary_types_allowed = True
    
    def __init__(self, max_retry: int = 3, **kwargs):
        args = BedrockTitanArgs()
        args_dict = args.dict()
        
        for k, v in args_dict.items():
            args_dict[k] = kwargs.pop(k, v)
        
        super().__init__(args=args_dict, max_retry=max_retry)
        
        self._bedrock_llm = BedrockTitanLLM(
            model=args_dict['model'],
            region=args_dict['region'],
            embedding_model_id=args_dict['embedding_model_id'],
            aws_access_key_id=args_dict['aws_access_key_id'],
            aws_secret_access_key=args_dict['aws_secret_access_key'],
            aws_session_token=args_dict['aws_session_token']
        )
    
    def _construct_messages(self, prompt):
        if isinstance(prompt, str):
            return [{"role": "user", "content": prompt}]
        elif isinstance(prompt, list):
            if len(prompt) > 0 and isinstance(prompt[0], dict):
                return prompt
            else:
                return [[{"role": "user", "content": p}] for p in prompt]
        return [{"role": "user", "content": str(prompt)}]
    
    def generate_response(self, prompt: str) -> LLMResult:
        messages = self._construct_messages(prompt)
        
        try:
            response_text = self._bedrock_llm(
                messages=messages,
                temperature=getattr(self.args, 'temperature', 0.2),
                max_tokens=getattr(self.args, 'max_tokens', 2048)
            )
            
            send_tokens = self._estimate_tokens(str(messages))
            recv_tokens = self._estimate_tokens(response_text)
            
            return LLMResult(
                content=response_text,
                send_tokens=send_tokens,
                recv_tokens=recv_tokens,
                total_tokens=send_tokens + recv_tokens
            )
        except Exception as e:
            logger.error(f"Bedrock Titan error: {e}")
            raise
    
    async def agenerate_response(self, prompt) -> Union[LLMResult, List]:
        messages = self._construct_messages(prompt)
        
        if isinstance(messages[0], list):
            responses = []
            for msg_list in messages:
                try:
                    response_text = self._bedrock_llm(
                        messages=msg_list,
                        temperature=getattr(self.args, 'temperature', 0.2),
                        max_tokens=getattr(self.args, 'max_tokens', 2048)
                    )
                    responses.append(response_text)
                except Exception as e:
                    logger.error(f"Bedrock Titan batch error: {e}")
                    raise
            return responses
        else:
            try:
                response_text = self._bedrock_llm(
                    messages=messages,
                    temperature=getattr(self.args, 'temperature', 0.2),
                    max_tokens=getattr(self.args, 'max_tokens', 2048)
                )
                return [response_text]
            except Exception as e:
                logger.error(f"Bedrock Titan async error: {e}")
                raise
    
    async def agenerate_response_without_construction(self, prompts) -> List:
        """Asynchronous response generation without message construction for batch processing"""
        responses = []
        
        for prompt in prompts:
            try:
                # Assume prompt is already in messages format
                response_text = self._bedrock_llm(
                    messages=prompt,
                    temperature=getattr(self.args, 'temperature', 0.2),
                    max_tokens=getattr(self.args, 'max_tokens', 2048)
                )
                
                # Format response to match OpenAI-style structure expected by the parser
                formatted_response = {
                    "choices": [{
                        "message": {
                            "content": response_text
                        }
                    }]
                }
                responses.append(formatted_response)
            except Exception as e:
                logger.error(f"Bedrock Titan batch error: {e}")
                raise
        
        return responses
    
    def _estimate_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)