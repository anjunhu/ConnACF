"""
MACFToolkit - Retrieval tools for MACF agents.

Provides the 4 retrieval utilities specified in the MACF paper:
- GetSimilarUsers: Find users similar to target
- GetRelevantItems: Find history items relevant to query
- RetrieveByQuery: Find candidates by query embedding
- RetrieveByItem: Find candidates similar to an item

All methods include parameter validation and return empty lists
with logged warnings for invalid inputs.
"""

import logging
from typing import List, Any, Optional

import torch
import numpy as np

from connacf.macf.index_manager import GlobalIndexManager

logger = logging.getLogger(__name__)


class MACFToolkit:
    """
    Retrieval tools for MACF agents.
    
    Provides the 4 retrieval utilities specified in the MACF paper:
    - GetSimilarUsers: Find users similar to target
    - GetRelevantItems: Find history items relevant to query
    - RetrieveByQuery: Find candidates by query embedding
    - RetrieveByItem: Find candidates similar to an item
    
    Attributes:
        index_manager: GlobalIndexManager for accessing indices
        embedding_agent: Agent for generating text embeddings
    """
    
    def __init__(self, index_manager: GlobalIndexManager, embedding_agent: Any):
        """
        Initialize MACFToolkit with index manager and embedding agent.
        
        Args:
            index_manager: GlobalIndexManager instance with built indices
            embedding_agent: Embedding agent with llm.agenerate_response method
                            for generating text embeddings
        """
        self.index_manager = index_manager
        self.embedding_agent = embedding_agent
        
        logger.info("MACFToolkit initialized")
    
    def get_similar_users(self, target_user_id: int, n: int = 5) -> List[int]:
        """
        Get n users most similar to target user.
        
        Uses interaction history overlap or embedding similarity via
        the GlobalIndexManager's user similarity matrix.
        
        Args:
            target_user_id: The ID of the target user
            n: Number of similar users to return (default: 5)
            
        Returns:
            List[int]: List of n user IDs most similar to target user,
                      sorted by similarity (most similar first).
                      Returns empty list if invalid parameters.
        
        Validates: Requirements 2.4
        """
        # Parameter validation
        if not self._validate_user_id(target_user_id, "get_similar_users"):
            return []
        
        if not self._validate_count(n, "get_similar_users"):
            return []
        
        # Check if indices are built
        if not self.index_manager.indices_built:
            logger.warning("get_similar_users: Indices not built, returning empty list")
            return []
        
        try:
            # Get similarity scores for target user
            similarity_scores = self.index_manager.get_user_similarity(target_user_id)
            
            # Set self-similarity to -inf to exclude target user from results
            similarity_scores[target_user_id] = float('-inf')
            
            # Get indices of top-n most similar users
            # Use argpartition for efficiency, then sort the top-n
            n_users = len(similarity_scores)
            n = min(n, n_users - 1)  # Exclude target user
            
            if n <= 0:
                return []
            
            # Get top-n indices
            top_n_indices = np.argpartition(similarity_scores, -n)[-n:]
            
            # Sort by similarity (descending)
            top_n_indices = top_n_indices[np.argsort(similarity_scores[top_n_indices])[::-1]]
            
            # Filter out users with zero similarity (no overlap)
            result = [
                int(idx) for idx in top_n_indices 
                if similarity_scores[idx] > 0
            ]
            
            logger.debug(
                f"get_similar_users: Found {len(result)} similar users for user {target_user_id}"
            )
            
            return result
            
        except Exception as e:
            logger.warning(f"get_similar_users: Error retrieving similar users: {e}")
            return []
    
    def get_relevant_items(self, target_user_id: int, query: str, n: int = 5) -> List[int]:
        """
        Get n items from target user's history most relevant to query.
        
        Performs History_Retrieval: filters user's history, ranks by
        semantic similarity to query.
        
        Args:
            target_user_id: The ID of the target user
            query: Natural language query string
            n: Number of relevant items to return (default: 5)
            
        Returns:
            List[int]: List of n item IDs from user's history most relevant
                      to the query, sorted by relevance (most relevant first).
                      Returns empty list if invalid parameters.
        
        Validates: Requirements 2.5
        """
        # Parameter validation
        if not self._validate_user_id(target_user_id, "get_relevant_items"):
            return []
        
        if not self._validate_query(query, "get_relevant_items"):
            return []
        
        if not self._validate_count(n, "get_relevant_items"):
            return []
        
        # Check if indices are built
        if not self.index_manager.indices_built:
            logger.warning("get_relevant_items: Indices not built, returning empty list")
            return []
        
        try:
            # Get user's interaction history
            user_history = self.index_manager.get_user_history(target_user_id)
            
            if not user_history:
                logger.warning(
                    f"get_relevant_items: User {target_user_id} has no interaction history"
                )
                return []
            
            # Use CF-based scoring: rank history items by their CF embedding similarity
            # to items the user is likely to want (based on user embedding)
            user_embedding = self.index_manager.get_user_embedding(target_user_id)
            
            if user_embedding is None or torch.all(user_embedding == 0):
                logger.warning("get_relevant_items: No user embedding, returning recent history")
                return user_history[:n]
            
            # Score each history item by how well it represents the user's preferences
            # Items with high user-item affinity are more "relevant"
            item_scores = []
            for item_id in user_history:
                item_embedding = self.index_manager.get_item_embedding(item_id)
                
                # CF-based score: user_emb · item_emb (both are normalized)
                if item_embedding is not None and not torch.all(item_embedding == 0):
                    score = float(torch.dot(user_embedding, item_embedding).item())
                else:
                    score = 0.0
                item_scores.append((item_id, score))
            
            # Sort by CF score (descending) and return top-n
            item_scores.sort(key=lambda x: x[1], reverse=True)
            result = [item_id for item_id, _ in item_scores[:n]]
            
            logger.debug(
                f"get_relevant_items: Found {len(result)} relevant items for user {target_user_id} (CF-based)"
            )
            
            return result
            
        except Exception as e:
            logger.warning(f"get_relevant_items: Error retrieving relevant items: {e}")
            return []
    
    def retrieve_by_query(self, query: str, k: int = 15, user_id: int = None) -> List[int]:
        """
        Get k candidate items using CF embeddings.
        
        If user_id is provided, uses CF-based retrieval (user embedding -> item scores).
        Otherwise falls back to text-based semantic search.
        
        CF-based retrieval is preferred as it captures collaborative filtering signals.
        
        Args:
            query: Natural language query string (used for fallback)
            k: Number of candidate items to return (default: 15)
            user_id: Optional user ID for CF-based retrieval
            
        Returns:
            List[int]: List of k item IDs most relevant to the user/query,
                      sorted by relevance (most relevant first).
                      Returns empty list if invalid parameters.
        
        Validates: Requirements 2.6
        """
        # Parameter validation
        if not self._validate_count(k, "retrieve_by_query"):
            return []
        
        # Check if indices are built
        if not self.index_manager.indices_built:
            logger.warning("retrieve_by_query: Indices not built, returning empty list")
            return []
        
        try:
            # PREFER CF-based retrieval if user_id is available
            if user_id is not None:
                result = self.index_manager.get_items_for_user(user_id, k)
                if result:
                    logger.debug(
                        f"retrieve_by_query: CF-based retrieval found {len(result)} candidates for user {user_id}"
                    )
                    return result
            
            # Fallback to text-based semantic search
            if not self._validate_query(query, "retrieve_by_query"):
                return []
            
            # Generate query embedding
            query_embedding = self._get_query_embedding(query)
            
            if query_embedding is None:
                logger.warning("retrieve_by_query: Failed to generate query embedding")
                return []
            
            # Perform semantic search
            result = self.index_manager.semantic_search_items(query_embedding, k)
            
            logger.debug(
                f"retrieve_by_query: Text-based retrieval found {len(result)} candidates for query '{query[:50]}...'"
            )
            
            return result
            
        except Exception as e:
            logger.warning(f"retrieve_by_query: Error retrieving candidates: {e}")
            return []
    
    def retrieve_by_item(self, item_id: int, k: int = 15) -> List[int]:
        """
        Get k candidate items similar to given item using CF embeddings.
        
        Uses the item's CF embedding to find similar items based on
        collaborative filtering signals (items that co-occur with similar users).
        
        Args:
            item_id: The ID of the source item
            k: Number of candidate items to return (default: 15)
            
        Returns:
            List[int]: List of k item IDs most similar to the source item,
                      sorted by similarity (most similar first).
                      Returns empty list if invalid parameters.
        
        Validates: Requirements 2.7
        """
        # Parameter validation
        if not self._validate_item_id(item_id, "retrieve_by_item"):
            return []
        
        if not self._validate_count(k, "retrieve_by_item"):
            return []
        
        # Check if indices are built
        if not self.index_manager.indices_built:
            logger.warning("retrieve_by_item: Indices not built, returning empty list")
            return []
        
        try:
            # Get item CF embedding
            item_embedding = self.index_manager.get_item_embedding(item_id)
            
            # Check if embedding is valid (not all zeros)
            if torch.all(item_embedding == 0):
                logger.warning(
                    f"retrieve_by_item: Item {item_id} has zero embedding, returning empty list"
                )
                return []
            
            # Perform CF-based search (request k+1 to exclude source item)
            candidates = self.index_manager.semantic_search_items(item_embedding, k + 1)
            
            # Remove source item from results if present
            result = [cid for cid in candidates if cid != item_id][:k]
            
            logger.debug(
                f"retrieve_by_item: Found {len(result)} similar items for item {item_id} (CF-based)"
            )
            
            return result
            
        except Exception as e:
            logger.warning(f"retrieve_by_item: Error retrieving similar items: {e}")
            return []
    
    # ==================== Private Helper Methods ====================
    
    def _validate_user_id(self, user_id: int, method_name: str) -> bool:
        """
        Validate user_id parameter.
        
        Args:
            user_id: User ID to validate
            method_name: Name of calling method for logging
            
        Returns:
            bool: True if valid, False otherwise
        """
        if not isinstance(user_id, (int, np.integer)):
            logger.warning(
                f"{method_name}: Invalid user_id type {type(user_id)}, expected int"
            )
            return False
        
        if user_id < 0:
            logger.warning(
                f"{method_name}: Invalid user_id {user_id}, must be non-negative"
            )
            return False
        
        # Check against index manager bounds if available
        if self.index_manager.indices_built and user_id >= self.index_manager.n_users:
            logger.warning(
                f"{method_name}: user_id {user_id} out of range "
                f"(max: {self.index_manager.n_users - 1})"
            )
            return False
        
        return True
    
    def _validate_item_id(self, item_id: int, method_name: str) -> bool:
        """
        Validate item_id parameter.
        
        Args:
            item_id: Item ID to validate
            method_name: Name of calling method for logging
            
        Returns:
            bool: True if valid, False otherwise
        """
        if not isinstance(item_id, (int, np.integer)):
            logger.warning(
                f"{method_name}: Invalid item_id type {type(item_id)}, expected int"
            )
            return False
        
        if item_id < 0:
            logger.warning(
                f"{method_name}: Invalid item_id {item_id}, must be non-negative"
            )
            return False
        
        # Check against index manager bounds if available
        if self.index_manager.indices_built and item_id >= self.index_manager.n_items:
            logger.warning(
                f"{method_name}: item_id {item_id} out of range "
                f"(max: {self.index_manager.n_items - 1})"
            )
            return False
        
        return True
    
    def _validate_query(self, query: str, method_name: str) -> bool:
        """
        Validate query parameter.
        
        Args:
            query: Query string to validate
            method_name: Name of calling method for logging
            
        Returns:
            bool: True if valid, False otherwise
        """
        if not isinstance(query, str):
            logger.warning(
                f"{method_name}: Invalid query type {type(query)}, expected str"
            )
            return False
        
        if not query or not query.strip():
            logger.warning(
                f"{method_name}: Empty query string provided"
            )
            return False
        
        return True
    
    def _validate_count(self, count: int, method_name: str) -> bool:
        """
        Validate count parameter (n or k).
        
        Args:
            count: Count value to validate
            method_name: Name of calling method for logging
            
        Returns:
            bool: True if valid, False otherwise
        """
        if not isinstance(count, (int, np.integer)):
            logger.warning(
                f"{method_name}: Invalid count type {type(count)}, expected int"
            )
            return False
        
        if count <= 0:
            logger.warning(
                f"{method_name}: Invalid count {count}, must be positive"
            )
            return False
        
        return True
    
    def _get_query_embedding(self, query: str) -> Optional[torch.Tensor]:
        """
        Generate embedding for a query string.
        
        Uses the embedding_agent to generate text embeddings.
        
        Args:
            query: Query string to embed
            
        Returns:
            torch.Tensor: Query embedding (on CPU), or None if generation fails
        """
        try:
            if self.embedding_agent is None:
                logger.warning("No embedding agent available")
                return None
            
            embedding = None
            
            # Try different embedding agent interfaces
            if hasattr(self.embedding_agent, 'get_embedding'):
                # Direct embedding method
                embedding = self.embedding_agent.get_embedding(query)
            elif hasattr(self.embedding_agent, 'generate_embedding'):
                # ConnaCF model's embedding method (takes a list)
                embeddings = self.embedding_agent.generate_embedding([query])
                if embeddings is not None and len(embeddings) > 0:
                    if isinstance(embeddings, torch.Tensor):
                        embedding = embeddings[0, :] if embeddings.dim() > 1 else embeddings
                    elif isinstance(embeddings, list):
                        embedding = embeddings[0]
                    else:
                        embedding = embeddings
            elif hasattr(self.embedding_agent, 'llm') and hasattr(self.embedding_agent.llm, 'get_embedding'):
                # Embedding via LLM wrapper
                embedding = self.embedding_agent.llm.get_embedding(query)
            elif hasattr(self.embedding_agent, 'embed'):
                # Alternative embedding method
                embedding = self.embedding_agent.embed(query)
            else:
                logger.warning(
                    "Embedding agent does not have a recognized embedding method "
                    f"(available: {[m for m in dir(self.embedding_agent) if not m.startswith('_')]})"
                )
                return None
            
            if embedding is None:
                return None
            
            # Convert to tensor if needed and ensure on CPU
            if isinstance(embedding, np.ndarray):
                embedding = torch.from_numpy(embedding).float()
            elif isinstance(embedding, list):
                embedding = torch.tensor(embedding, dtype=torch.float32)
            elif isinstance(embedding, torch.Tensor):
                embedding = embedding.detach().cpu().float()
            else:
                embedding = torch.tensor(embedding, dtype=torch.float32)
            
            # Ensure CPU for consistency with index_manager
            if embedding.is_cuda:
                embedding = embedding.cpu()
            
            return embedding
            
        except Exception as e:
            logger.warning(f"Error generating query embedding: {e}")
            return None
    
    def _cosine_similarity(
        self, 
        embedding1: torch.Tensor, 
        embedding2: torch.Tensor
    ) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            float: Cosine similarity score in range [-1, 1]
        """
        try:
            # Ensure 1D tensors
            if embedding1.dim() > 1:
                embedding1 = embedding1.squeeze()
            if embedding2.dim() > 1:
                embedding2 = embedding2.squeeze()
            
            # Compute norms
            norm1 = embedding1.norm(p=2)
            norm2 = embedding2.norm(p=2)
            
            # Avoid division by zero
            if norm1 < 1e-8 or norm2 < 1e-8:
                return 0.0
            
            # Compute cosine similarity
            similarity = torch.dot(embedding1, embedding2) / (norm1 * norm2)
            
            return float(similarity.item())
            
        except Exception as e:
            logger.warning(f"Error computing cosine similarity: {e}")
            return 0.0
    
    def __repr__(self) -> str:
        """String representation of MACFToolkit."""
        return (
            f"MACFToolkit("
            f"index_manager={self.index_manager}, "
            f"embedding_agent={'set' if self.embedding_agent else 'None'})"
        )
