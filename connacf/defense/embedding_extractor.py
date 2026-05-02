"""Embedding extractor for G-safeguard / BlindGuard defense.

Extracts sentence embeddings from agent interaction text using a pre-trained
Sentence Transformer model and maintains a temporal buffer of embeddings
per agent across training rounds.
"""

import logging
from typing import Dict, List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class EmbeddingExtractor:
    """Extracts and buffers sentence embeddings from agent interactions.

    Uses sentence-transformers (all-MiniLM-L6-v2) to encode agent text into
    384-dimensional vectors. Maintains a per-agent temporal buffer that stores
    up to ``max_temporal_windows`` embeddings in FIFO order.

    Args:
        model_name: HuggingFace model identifier for the sentence transformer.
        max_temporal_windows: Maximum number of temporal embedding windows
            to retain per agent. Oldest embeddings are dropped when full.
        embedding_dim: Dimensionality of the embedding vectors.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        max_temporal_windows: int = 3,
        embedding_dim: int = 384,
    ):
        self.model_name = model_name
        self.max_temporal_windows = max_temporal_windows
        self.embedding_dim = embedding_dim

        try:
            self.model = SentenceTransformer(model_name)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load SentenceTransformer model '{model_name}'. "
                f"Install with: pip install sentence-transformers\n"
                f"Original error: {e}"
            ) from e

        # Per-agent temporal buffer:
        # {agent_id: {"type": str, "embeddings": np.ndarray,
        #             "round_indices": List[int], "text_history": List[str]}}
        self._buffer: Dict[int, dict] = {}

    def extract_embedding(self, text: str) -> np.ndarray:
        """Encode text into a fixed-dimension embedding vector.

        Args:
            text: Input text string. If empty or None, returns a zero vector.

        Returns:
            Numpy array of shape ``(embedding_dim,)`` with finite values.
        """
        if not text:
            logger.debug("Empty or None text received, returning zero vector.")
            return np.zeros(self.embedding_dim, dtype=np.float32)

        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.astype(np.float32).flatten()

    def record_agent_embedding(
        self, agent_id: int, agent_type: str, text: str, round_idx: int
    ) -> None:
        """Store an embedding in the temporal buffer for the given agent.

        Computes the embedding for ``text`` and appends it to the agent's
        temporal buffer. If the buffer already contains ``max_temporal_windows``
        entries, the oldest entry is dropped (FIFO).

        Args:
            agent_id: Unique identifier for the agent.
            agent_type: Either ``"user"`` or ``"item"``.
            text: The agent's profile/description text to embed.
            round_idx: The training round index for this embedding.
        """
        embedding = self.extract_embedding(text)

        if agent_id not in self._buffer:
            self._buffer[agent_id] = {
                "type": agent_type,
                "embeddings": np.zeros(
                    (self.max_temporal_windows, self.embedding_dim),
                    dtype=np.float32,
                ),
                "round_indices": [],
                "text_history": [],
            }

        buf = self._buffer[agent_id]
        n_existing = len(buf["round_indices"])

        if n_existing < self.max_temporal_windows:
            # Still have room — place at the next open slot
            buf["embeddings"][n_existing] = embedding
        else:
            # Buffer full — shift left (drop oldest) and place at end
            buf["embeddings"][:-1] = buf["embeddings"][1:]
            buf["embeddings"][-1] = embedding
            # Trim metadata lists to keep max_temporal_windows - 1 before append
            buf["round_indices"] = buf["round_indices"][-(self.max_temporal_windows - 1):]
            buf["text_history"] = buf["text_history"][-(self.max_temporal_windows - 1):]

        buf["round_indices"].append(round_idx)
        buf["text_history"].append(text if text else "")

    def get_temporal_embeddings(self, agent_id: int) -> np.ndarray:
        """Return the temporal embedding matrix for an agent.

        If the agent has fewer than ``max_temporal_windows`` recorded
        embeddings, the remaining slots are zero-padded. If the agent
        has not been recorded at all, returns an all-zeros matrix.

        Args:
            agent_id: Unique identifier for the agent.

        Returns:
            Numpy array of shape ``(max_temporal_windows, embedding_dim)``.
        """
        if agent_id not in self._buffer:
            return np.zeros(
                (self.max_temporal_windows, self.embedding_dim),
                dtype=np.float32,
            )
        return self._buffer[agent_id]["embeddings"].copy()

    def reset(self) -> None:
        """Clear all buffered embeddings."""
        self._buffer.clear()
