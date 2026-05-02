"""
Unified Conversation Logger for ConnaCF

Combines all logging functionality:
1. Red-teaming MACF interaction logging (batch/round/turn tracking)
2. U-U and U-I interaction logging for NL conversations
3. Attack tracking and metrics

Provides readable, conversation-format logs that clearly identify:
- Which users (with profiles/memory) are interacting
- Which items (with profiles/memory) are being recommended
- What the system agent is saying
- The flow of the conversation in each round
"""

import os
import json
import time
import threading
from typing import Dict, List, Any, Optional, Set
from datetime import datetime
from dataclasses import dataclass, asdict
from queue import Queue
import logging

logger = logging.getLogger(__name__)


@dataclass
class UULogEntry:
    """Log entry for a U-U interaction."""
    user_id: int
    friend_id: int
    opinion: str
    timestamp: float
    round_idx: int
    batch_idx: int
    was_intercepted: bool = False
    original_opinion: Optional[str] = None
    similarity_score: float = 0.0
    is_adversarial: bool = False


@dataclass
class UILogEntry:
    """Log entry for a U-I dialogue."""
    user_id: int
    item_id: int
    item_title: str
    dialogue_turns: List[Dict[str, str]]
    timestamp: float
    round_idx: int
    batch_idx: int
    overall_sentiment: str = "neutral"


class ConversationLogger:
    """Unified logger for all ConnaCF interactions.
    
    Supports multiple use cases:
    1. Attack integration logging (log_batch_start, log_interaction, etc.)
    2. NL conversation logging (log_uu_interaction, log_ui_dialogue)
    3. Model training logging (record_path based)
    """
    
    def __init__(
        self,
        output_dir: str,
        task_id_or_dataset: Any = 0,
        record_idx: int = 0,
        async_logging: bool = True,
        direct_task_dir: bool = False
    ):
        """Initialize conversation logger.
        
        Supports three calling conventions:
        1. ConversationLogger(task_dir, direct_task_dir=True) - use task_dir directly (no nesting)
        2. ConversationLogger(output_dir, task_id) - for attack integration (creates task_{id} subdir)
        3. ConversationLogger(record_path, dataset_name, record_idx) - for model training
        
        Args:
            output_dir: Base directory for logs (or record_path, or direct task_dir)
            task_id_or_dataset: Task ID (int) or dataset name (str)
            record_idx: Record session index (only used if task_id_or_dataset is str)
            async_logging: Whether to use async logging
            direct_task_dir: If True, use output_dir directly as task_dir (no nesting)
        """
        self.output_dir = output_dir
        self.async_logging = async_logging
        
        # Direct task_dir mode - use output_dir as-is
        if direct_task_dir:
            self.task_id = 0
            self.dataset_name = ""
            self.record_idx = 0
            self.task_dir = output_dir
            self.conversation_dir = os.path.join(self.task_dir, "conversations")
        # Determine calling convention
        elif isinstance(task_id_or_dataset, int):
            # Attack integration mode: ConversationLogger(output_dir, task_id)
            self.task_id = task_id_or_dataset
            self.dataset_name = ""
            self.record_idx = task_id_or_dataset
            self.task_dir = os.path.join(output_dir, f"task_{self.task_id}")
            self.conversation_dir = os.path.join(self.task_dir, "conversations")
        else:
            # Model training mode: ConversationLogger(record_path, dataset_name, record_idx)
            self.task_id = record_idx
            self.dataset_name = str(task_id_or_dataset)
            self.record_idx = record_idx
            self.task_dir = os.path.join(output_dir, self.dataset_name)
            self.conversation_dir = os.path.join(
                self.task_dir, 'conversations', f'session_{record_idx}'
            )
        
        # Create directories
        os.makedirs(self.conversation_dir, exist_ok=True)
        
        # Track conversation history
        self.conversation_history = []
        
        # Counters
        self.batch_counter = 0
        self.turn_counter = 0
        
        # U-U and U-I logs for NL interactions
        self.uu_logs: List[UULogEntry] = []
        self.ui_logs: List[UILogEntry] = []
        
        # Async logging support
        self._log_queue: Queue = Queue()
        self._worker_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        
        # Track which agents have had their system prompts logged
        self._logged_user_system_prompts = set()
        self._logged_item_system_prompts = set()
        
        if async_logging:
            self._start_worker()
    
    # ================================================================
    # Attack Integration Methods
    # ================================================================
    
    def log_resumption(self, resumed_from_turn: int, resumed_from_batch: int,
                       user_memories_restored: int, item_memories_restored: int,
                       attacker_indices_overridden: bool = False):
        """Log that the experiment was resumed from a previous checkpoint."""
        from datetime import datetime
        text = f"\n{'█' * 100}\n"
        text += f"{'EXPERIMENT RESUMED':^100}\n"
        text += f"{'█' * 100}\n"
        text += f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        text += f"Resumed from global_turn={resumed_from_turn}, batch={resumed_from_batch}\n"
        text += f"Restored {user_memories_restored} user memories, {item_memories_restored} item memories\n"
        if attacker_indices_overridden:
            text += f"⚠ Attacker indices were overridden from saved checkpoint\n"
        text += f"{'█' * 100}\n\n"
        self._write_to_log(text)

    def log_batch_start(self, batch_num: int, round_num: int, global_turn: int):
        """Log the start of a new batch/round."""
        self.batch_counter = batch_num
        self.turn_counter = global_turn
        
        header = f"\n{'█'*100}\n"
        header += f"BATCH {batch_num} | ROUND {round_num} | GLOBAL TURN {global_turn}\n"
        header += f"{'█'*100}\n"
        
        self._write_to_log(header)
    
    def log_user_profile(self, user_id: int, profile: str, is_attacker: bool = False):
        """Log user profile/memory."""
        role = "🔴 ATTACKER USER" if is_attacker else "👤 USER"
        
        text = f"\n{role} #{user_id}:\n"
        text += f"{'─'*90}\n"
        text += f"Profile: {profile}\n"
        text += f"{'─'*90}\n"
        
        self._write_to_log(text)
    
    def log_user_item_interaction_grouped(
        self,
        interaction_idx: int,
        user_id: int,
        user_profile: str,
        candidate_items: List[Dict[str, Any]],
        is_attacker_user: bool = False,
        attacker_item_indices: Optional[Set[int]] = None,
        neighbor_user_ids: Optional[List[int]] = None,
        user_history_items: Optional[List[Dict[str, Any]]] = None
    ):
        """Log a single user-item interaction in line-style format.
        
        Uses line-style format compatible with judge_calibration_connacf.py parsing.
        Does NOT truncate profiles to preserve full information for analysis.
        
        Args:
            interaction_idx: Index of this interaction within the batch
            user_id: User ID
            user_profile: User's profile/memory (NOT truncated)
            candidate_items: List of dicts with keys: item_id, profile, title
            is_attacker_user: Whether this user is an attacker
            attacker_item_indices: Set of item IDs that are attackers
            neighbor_user_ids: Optional list of neighbor user IDs this user is connected to
            user_history_items: Optional list of user's history items (dicts with item_id, profile, title)
        """
        attacker_item_indices = attacker_item_indices or set()
        
        # Line-style format compatible with judge_calibration_connacf.py
        user_role = "🔴 ATTACKER USER" if is_attacker_user else "👤 USER"
        
        # User profile block - NO truncation
        text = f"\n{user_role} #{user_id}:\n"
        text += f"{'─'*90}\n"
        text += f"Profile: {user_profile}\n"
        
        # Add neighbor information if available (from MACF-style logging)
        if neighbor_user_ids:
            text += f"\nNeighbor Users: {', '.join(f'#{nid}' for nid in neighbor_user_ids)}\n"
        
        text += f"{'─'*90}\n"
        
        # User's history items - THE KEY CONTEXT for understanding preferences!
        if user_history_items:
            text += f"\n📚 USER #{user_id}'s HISTORY ITEMS ({len(user_history_items)} items):\n"
            text += f"{'─'*90}\n"
            for hist_item in user_history_items:
                hist_item_id = hist_item.get('item_id', 0)
                hist_item_title = hist_item.get('title', '')
                hist_item_profile = hist_item.get('profile', '')
                is_attacker_hist = hist_item_id in attacker_item_indices
                
                hist_role = "🔴 ATTACKER ITEM" if is_attacker_hist else "📀 HISTORY ITEM"
                title_str = f" - {hist_item_title}" if hist_item_title else ""
                
                text += f"  {hist_role} #{hist_item_id}{title_str}\n"
                # Show truncated description for history items to keep log readable
                if hist_item_profile:
                    desc_preview = hist_item_profile[:150] + "..." if len(hist_item_profile) > 150 else hist_item_profile
                    text += f"    Description: {desc_preview}\n"
            text += f"{'─'*90}\n"
        
        # Candidate item profiles - NO truncation (these are the items being recommended)
        for item in candidate_items:
            item_id = item.get('item_id', 0)
            item_profile = item.get('profile', '')
            item_title = item.get('title', '')
            is_attacker_item = item_id in attacker_item_indices
            
            item_role = "🔴 ATTACKER ITEM" if is_attacker_item else "💿 ITEM"
            title_str = f" - {item_title}" if item_title else ""
            
            text += f"\n{item_role} #{item_id}{title_str}:\n"
            text += f"{'─'*90}\n"
            text += f"Description: {item_profile}\n"
            text += f"{'─'*90}\n"
        
        self._write_to_log(text)

    def log_user_system_prompt(self, user_id: int, system_prompt: str, is_attacker: bool = False):
        """Log user agent's system prompt (ONCE per agent, at first turn).
        
        System prompts should NOT be optimizable - they remain fixed throughout training.
        This method logs them once at the beginning for transparency.
        
        Args:
            user_id: User agent ID
            system_prompt: The system prompt template (with $user_description placeholder)
            is_attacker: Whether this is an attacker agent
        """
        # Only log once per agent
        if user_id in self._logged_user_system_prompts:
            return
        
        self._logged_user_system_prompts.add(user_id)
        
        role = "🔴 ATTACKER USER" if is_attacker else "👤 USER"
        
        text = f"\n{'='*90}\n"
        text += f"{role} #{user_id} - SYSTEM PROMPT (FIXED, NOT OPTIMIZABLE)\n"
        text += f"{'='*90}\n"
        text += f"{self._format_text(system_prompt, indent=0)}\n"
        text += f"{'='*90}\n"
        
        self._write_to_log(text)
    
    def log_item_system_prompt(self, item_id: int, system_prompt: str, is_attacker: bool = False):
        """Log item agent's system prompt (ONCE per agent, at first turn).
        
        System prompts should NOT be optimizable - they remain fixed throughout training.
        This method logs them once at the beginning for transparency.
        
        Args:
            item_id: Item agent ID
            system_prompt: The system prompt template
            is_attacker: Whether this is an attacker agent
        """
        # Only log once per agent
        if item_id in self._logged_item_system_prompts:
            return
        
        self._logged_item_system_prompts.add(item_id)
        
        role = "🔴 ATTACKER ITEM" if is_attacker else "💿 ITEM"
        
        text = f"\n{'='*90}\n"
        text += f"{role} #{item_id} - SYSTEM PROMPT (FIXED, NOT OPTIMIZABLE)\n"
        text += f"{'='*90}\n"
        text += f"{self._format_text(system_prompt, indent=0)}\n"
        text += f"{'='*90}\n"
        
        self._write_to_log(text)
    
    def log_item_profile(self, item_id: int, profile: str, title: str = "", is_attacker: bool = False):
        """Log item profile/memory."""
        role = "🔴 ATTACKER ITEM" if is_attacker else "💿 ITEM"
        title_str = f" - {title}" if title else ""
        
        text = f"\n{role} #{item_id}{title_str}:\n"
        text += f"{'─'*90}\n"
        text += f"Description: {profile}\n"
        text += f"{'─'*90}\n"
        
        self._write_to_log(text)
    
    def log_similar_users(self, user_id: int, similar_user_ids: List[int], 
                          attacker_user_indices: Optional[Set[int]] = None):
        """Log similar/neighbor users for a target user.
        
        This logs the "what's my neighbours" information showing which users
        are connected to the target user for collaborative filtering.
        
        Args:
            user_id: Target user ID
            similar_user_ids: List of similar/neighbor user IDs
            attacker_user_indices: Set of user IDs that are attackers
        """
        attacker_user_indices = attacker_user_indices or set()
        
        if not similar_user_ids:
            return
        
        text = f"\n🔗 SIMILAR USERS for User #{user_id}:\n"
        text += f"{'─'*90}\n"
        
        for neighbor_id in similar_user_ids:
            is_attacker = neighbor_id in attacker_user_indices
            marker = "🔴 ATTACKER" if is_attacker else "👤"
            text += f"  {marker} User #{neighbor_id}\n"
        
        text += f"{'─'*90}\n"
        
        self._write_to_log(text)
    
    def log_system_recommendation(self, user_id: int, pos_item_id: int, neg_item_id: int,
                                   choice: str, explanation: str, is_correct: bool):
        """Log system agent's recommendation decision (pairwise mode, num_candidates=2)."""
        result = "✓ CORRECT" if is_correct else "✗ INCORRECT"
        
        text = f"\n🤖 SYSTEM AGENT (Recommending to User #{user_id}):\n"
        text += f"{'─'*90}\n"
        text += f"Candidates: Item #{pos_item_id} (positive) vs Item #{neg_item_id} (negative)\n"
        text += f"Choice: {choice} [{result}]\n"
        text += f"\nExplanation:\n{self._format_text(explanation, indent=2)}\n"
        text += f"{'─'*90}\n"
        
        self._write_to_log(text)
    
    def log_system_recommendation_binary(self, user_id: int, item_id: int, item_title: str,
                                          decision: bool, explanation: str, is_correct: bool):
        """Log system agent's binary decision (num_candidates=1).
        
        For sparse topology where user makes yes/no decision on single item.
        """
        result = "✓ CORRECT" if is_correct else "✗ INCORRECT"
        decision_str = "Yes (would enjoy)" if decision else "No (would not enjoy)"
        
        text = f"\n🤖 SYSTEM AGENT (Binary Decision for User #{user_id}):\n"
        text += f"{'─'*90}\n"
        text += f"Candidate: Item #{item_id} - {item_title}\n"
        text += f"Decision: {decision_str} [{result}]\n"
        text += f"\nExplanation:\n{self._format_text(explanation, indent=2)}\n"
        text += f"{'─'*90}\n"
        
        self._write_to_log(text)
    
    def log_system_recommendation_ranking(self, user_id: int, candidate_items: List[int],
                                           item_titles: Dict[int, str], predicted_ranking: List[int],
                                           true_ranking: List[int], explanation: str,
                                           pairwise_accuracy: float):
        """Log system agent's ranking decision (num_candidates>=3).
        
        For dense topology where user ranks multiple candidates.
        No positive/negative labels - just candidate list and rankings.
        """
        num_candidates = len(candidate_items)
        top1_correct = predicted_ranking[0] == true_ranking[0] if predicted_ranking and true_ranking else False
        result = f"Top-1 {'✓' if top1_correct else '✗'} | Pairwise Acc: {pairwise_accuracy:.1%}"
        
        text = f"\n🤖 SYSTEM AGENT (Ranking {num_candidates} Candidates for User #{user_id}):\n"
        text += f"{'─'*90}\n"
        
        # List candidates without positive/negative labels
        text += f"Candidates ({num_candidates} items):\n"
        for i, item_id in enumerate(candidate_items):
            title = item_titles.get(item_id, f"Item {item_id}")
            text += f"  • Item #{item_id}: {title}\n"
        
        text += f"\nPredicted Ranking:\n"
        for rank, item_id in enumerate(predicted_ranking, 1):
            title = item_titles.get(item_id, f"Item {item_id}")
            text += f"  {rank}. Item #{item_id}: {title}\n"
        
        text += f"\nTrue Ranking:\n"
        for rank, item_id in enumerate(true_ranking, 1):
            title = item_titles.get(item_id, f"Item {item_id}")
            text += f"  {rank}. Item #{item_id}: {title}\n"
        
        text += f"\nResult: [{result}]\n"
        text += f"\nExplanation:\n{self._format_text(explanation, indent=2)}\n"
        text += f"{'─'*90}\n"
        
        self._write_to_log(text)
    
    def log_interaction_multi_candidate(self, batch_num: int, round_num: int, global_turn: int,
                                         user_id: int, user_profile: str, num_candidates: int,
                                         candidate_items: List[int], item_profiles: Dict[int, str],
                                         item_titles: Dict[int, str], prediction: Any,
                                         ground_truth: Any, explanation: str, is_correct: bool,
                                         attacker_user_indices: set = None, attacker_item_indices: set = None):
        """Log interaction for any num_candidates mode.
        
        Unified logging that adapts format based on num_candidates:
        - num_candidates == 1: Binary decision format
        - num_candidates == 2: Pairwise format with positive/negative labels
        - num_candidates >= 3: Ranking format without positive/negative labels
        """
        attacker_user_indices = attacker_user_indices or set()
        attacker_item_indices = attacker_item_indices or set()
        
        user_is_attacker = user_id in attacker_user_indices
        
        conversation = []
        conversation.append(f"\n{'─'*100}")
        conversation.append(f"🎯 INTERACTION #{global_turn} (Batch {batch_num}, Round {round_num})")
        conversation.append(f"{'─'*100}\n")
        
        # User info
        user_marker = "🔴 [ATTACKER]" if user_is_attacker else "👤 [USER]"
        conversation.append(f"{user_marker} User #{user_id}")
        conversation.append(f"  Profile: {user_profile}")
        conversation.append("")
        
        if num_candidates == 1:
            # Binary mode
            item_id = candidate_items[0]
            is_attacker_item = item_id in attacker_item_indices
            item_marker = "🔴 [ATTACKER ITEM]" if is_attacker_item else "📀 [ITEM]"
            
            conversation.append(f"{item_marker} Item #{item_id} - {item_titles.get(item_id, 'Unknown')}")
            conversation.append(f"  Description: {item_profiles.get(item_id, '')}")
            conversation.append("")
            
            conversation.append("🤖 SYSTEM AGENT DECISION (Binary):")
            decision_str = "Yes (would enjoy)" if prediction else "No (would not enjoy)"
            result_marker = "✅" if is_correct else "❌"
            conversation.append(f"  {result_marker} Decision: {decision_str}")
            
        elif num_candidates == 2:
            # Pairwise mode - keep positive/negative labels
            pos_item_id = ground_truth[0] if isinstance(ground_truth, list) else ground_truth
            neg_item_id = [c for c in candidate_items if c != pos_item_id][0]
            
            conversation.append("📀 ITEMS BEING RECOMMENDED:")
            conversation.append("")
            
            for item_id in candidate_items:
                is_attacker_item = item_id in attacker_item_indices
                is_positive = item_id == pos_item_id
                
                if is_attacker_item:
                    item_marker = "🔴 [ATTACKER ITEM]"
                elif is_positive:
                    item_marker = "  [ITEM A] (Positive)"
                else:
                    item_marker = "  [ITEM B] (Negative)"
                
                conversation.append(f"{item_marker} Item #{item_id} - {item_titles.get(item_id, 'Unknown')}")
                conversation.append(f"  Description: {item_profiles.get(item_id, '')}")
                conversation.append("")
            
            conversation.append("🤖 SYSTEM AGENT DECISION (Pairwise):")
            result_marker = "✅" if is_correct else "❌"
            conversation.append(f"  {result_marker} Choice: {prediction}")
            
        else:
            # Ranking mode - NO positive/negative labels
            conversation.append(f"📀 CANDIDATE ITEMS ({num_candidates} items):")
            conversation.append("")
            
            for item_id in candidate_items:
                is_attacker_item = item_id in attacker_item_indices
                item_marker = "🔴 [ATTACKER ITEM]" if is_attacker_item else "  [ITEM]"
                
                conversation.append(f"{item_marker} Item #{item_id} - {item_titles.get(item_id, 'Unknown')}")
                conversation.append(f"  Description: {item_profiles.get(item_id, '')}")
                conversation.append("")
            
            conversation.append("🤖 SYSTEM AGENT DECISION (Ranking):")
            conversation.append("")
            conversation.append("  Predicted Ranking:")
            for rank, item_id in enumerate(prediction, 1):
                conversation.append(f"    {rank}. Item #{item_id} - {item_titles.get(item_id, 'Unknown')}")
            
            conversation.append("")
            conversation.append("  True Ranking:")
            for rank, item_id in enumerate(ground_truth, 1):
                conversation.append(f"    {rank}. Item #{item_id} - {item_titles.get(item_id, 'Unknown')}")
            
            # Compute pairwise accuracy for result
            from model.ranking_loss import pairwise_accuracy_from_rankings
            _, _, pairwise_acc = pairwise_accuracy_from_rankings(prediction, ground_truth)
            top1_correct = prediction[0] == ground_truth[0] if prediction and ground_truth else False
            
            result_marker = "✅" if is_correct else "❌"
            conversation.append("")
            conversation.append(f"  {result_marker} Top-1: {'Correct' if top1_correct else 'Wrong'} | Pairwise Accuracy: {pairwise_acc:.1%}")
        
        conversation.append(f"\n  Explanation: {explanation}")
        conversation.append(f"{'─'*100}\n")
        
        log_text = "\n".join(conversation)
        self._write_to_log(log_text)
        self._save_interaction_to_file(batch_num, round_num, global_turn, log_text)
        
        # Store in history
        self.conversation_history.append({
            'batch': batch_num,
            'round': round_num,
            'global_turn': global_turn,
            'user_id': user_id,
            'user_is_attacker': user_is_attacker,
            'num_candidates': num_candidates,
            'candidate_items': candidate_items,
            'prediction': prediction if not isinstance(prediction, bool) else str(prediction),
            'ground_truth': ground_truth if not isinstance(ground_truth, bool) else str(ground_truth),
            'is_correct': is_correct,
            'timestamp': datetime.now().isoformat()
        })
    
    def log_user_update(self, user_id: int, old_profile: str, new_profile: str,
                        system_reason: str, is_attacker: bool = False):
        """Log user agent's profile update."""
        role = "🔴 ATTACKER USER" if is_attacker else "👤 USER"
        
        text = f"\n{role} #{user_id} UPDATE:\n"
        text += f"{'─'*90}\n"
        text += f"Previous Profile:\n{self._format_text(old_profile, indent=2)}\n\n"
        text += f"System's Reasoning:\n{self._format_text(system_reason, indent=2)}\n\n"
        text += f"Updated Profile:\n{self._format_text(new_profile, indent=2)}\n"
        text += f"{'─'*90}\n"
        
        self._write_to_log(text)
    
    def log_item_update(self, item_id: int, old_description: str, new_description: str,
                        user_feedback: str, title: str = "", is_attacker: bool = False,
                        num_user_profiles: int = 1, num_user_decisions: int = 1):
        """Log item agent's description update.
        
        Args:
            item_id: ID of the item being updated
            old_description: Previous item description
            new_description: Updated item description
            user_feedback: User feedback that triggered the update
            title: Item title
            is_attacker: Whether this is an attacker item
            num_user_profiles: Number of user profiles used in this update
            num_user_decisions: Number of user decisions used in this update
        """
        role = "🔴 ATTACKER ITEM" if is_attacker else "💿 ITEM"
        title_str = f" - {title}" if title else ""
        
        text = f"\n{role} #{item_id}{title_str} UPDATE:\n"
        text += f"{'─'*90}\n"
        text += f"User Profiles Used: {num_user_profiles} | User Decisions Used: {num_user_decisions}\n"
        text += f"{'─'*90}\n"
        text += f"Previous Description:\n{self._format_text(old_description, indent=2)}\n\n"
        text += f"User Feedback:\n{self._format_text(user_feedback, indent=2)}\n\n"
        text += f"Updated Description:\n{self._format_text(new_description, indent=2)}\n"
        text += f"{'─'*90}\n"
        
        self._write_to_log(text)
    
    def log_interaction_summary(self, batch_num: int, accuracy: float,
                                correct_count: int, total_count: int,
                                contaminated_users: int = 0, contaminated_items: int = 0):
        """Log summary statistics for the batch."""
        text = f"\n{'═'*100}\n"
        text += f"BATCH {batch_num} SUMMARY\n"
        text += f"{'═'*100}\n"
        text += f"Accuracy: {accuracy:.3f} ({correct_count}/{total_count} correct)\n"
        if contaminated_users > 0 or contaminated_items > 0:
            text += f"Contaminated Agents: {contaminated_users} users, {contaminated_items} items\n"
        text += f"{'═'*100}\n\n"
        
        self._write_to_log(text)
    
    def log_turn_metrics(self, turn: int, metrics: Dict[str, Any]):
        """Log metrics for a specific turn."""
        os.makedirs(self.conversation_dir, exist_ok=True)
        metrics_path = os.path.join(self.conversation_dir, 'metrics_summary.txt')
        
        with open(metrics_path, 'a') as f:
            if turn == 0:
                f.write("="*80 + "\n")
                f.write("TURN-BY-TURN METRICS SUMMARY\n")
                f.write("="*80 + "\n\n")
            
            f.write(f"Turn {turn}:\n")
            f.write(f"  Accuracy: {metrics.get('accuracy', 0):.4f}\n")
            f.write(f"  Recall@5: {metrics.get('recall@5', 0):.4f}\n")
            f.write(f"  NDCG@5: {metrics.get('ndcg@5', 0):.4f}\n")
            f.write(f"  Contaminated Users: {metrics.get('contaminated_users', 0)}\n")
            f.write(f"  Contaminated Items: {metrics.get('contaminated_items', 0)}\n")
            f.write(f"{'-'*80}\n")

    def log_interaction(self, batch_num: int, round_num: int, global_turn: int,
                        user_id: int, user_profile: str,
                        pos_item_id: int, pos_item_profile: str,
                        neg_item_id: int, neg_item_profile: str,
                        system_response: str, system_selection: str, is_correct: bool,
                        attacker_user_indices: set = None, attacker_item_indices: set = None):
        """Log a single interaction in conversation format."""
        attacker_user_indices = attacker_user_indices or set()
        attacker_item_indices = attacker_item_indices or set()
        
        user_is_attacker = user_id in attacker_user_indices
        pos_is_attacker = pos_item_id in attacker_item_indices
        neg_is_attacker = neg_item_id in attacker_item_indices
        
        conversation = []
        conversation.append(f"\n{'─'*100}")
        conversation.append(f"🎯 INTERACTION #{global_turn} (Batch {batch_num}, Round {round_num})")
        conversation.append(f"{'─'*100}\n")
        
        user_marker = "🔴 [ATTACKER]" if user_is_attacker else "👤 [USER]"
        conversation.append(f"{user_marker} User #{user_id}")
        conversation.append(f"  Profile: {user_profile}")
        conversation.append("")
        
        conversation.append("📀 ITEMS BEING RECOMMENDED:")
        conversation.append("")
        
        pos_marker = "🔴 [ATTACKER ITEM]" if pos_is_attacker else "  [ITEM A]"
        conversation.append(f"{pos_marker} Item #{pos_item_id} (Positive)")
        conversation.append(f"  Description: {pos_item_profile}")
        conversation.append("")
        
        neg_marker = "🔴 [ATTACKER ITEM]" if neg_is_attacker else "  [ITEM B]"
        conversation.append(f"{neg_marker} Item #{neg_item_id} (Negative)")
        conversation.append(f"  Description: {neg_item_profile}")
        conversation.append("")
        
        conversation.append("🤖 SYSTEM AGENT DECISION:")
        conversation.append("")
        
        choice, explanation = self._parse_system_response(system_response)
        result_marker = "✅" if is_correct else "❌"
        conversation.append(f"  {result_marker} Choice: {choice}")
        conversation.append(f"  Explanation: {explanation}")
        conversation.append("")
        
        if is_correct:
            conversation.append("  ✅ CORRECT: System chose the positive item")
        else:
            conversation.append("  ❌ INCORRECT: System chose the negative item")
        
        conversation.append(f"{'─'*100}\n")
        
        log_text = "\n".join(conversation)
        self._write_to_log(log_text)
        self._save_interaction_to_file(batch_num, round_num, global_turn, log_text)
        
        self.conversation_history.append({
            'batch': batch_num,
            'round': round_num,
            'global_turn': global_turn,
            'user_id': user_id,
            'user_is_attacker': user_is_attacker,
            'pos_item_id': pos_item_id,
            'pos_is_attacker': pos_is_attacker,
            'neg_item_id': neg_item_id,
            'neg_is_attacker': neg_is_attacker,
            'choice': choice,
            'is_correct': is_correct,
            'timestamp': datetime.now().isoformat()
        })
    
    def log_batch_summary(self, batch_num: int, round_num: int, global_turn: int,
                          accuracy: float, correct_count: int, total_count: int):
        """Log a summary after each batch/round."""
        summary = []
        summary.append(f"\n{'═'*100}")
        summary.append(f"📊 ROUND SUMMARY - Batch {batch_num}, Round {round_num}, Turn {global_turn}")
        summary.append(f"{'═'*100}")
        summary.append(f"  Accuracy: {accuracy:.3f} ({correct_count}/{total_count} correct)")
        summary.append(f"{'═'*100}\n")
        
        summary_text = "\n".join(summary)
        self._write_to_log(summary_text)
        self._save_summary_to_file(batch_num, round_num, global_turn, summary_text)
    
    def log_user_updates(self, user_updates: List[tuple]):
        """Log user memory updates."""
        if not user_updates:
            return
        
        updates = []
        updates.append(f"\n{'─'*100}")
        updates.append(f"🔄 USER MEMORY UPDATES ({len(user_updates)} users)")
        updates.append(f"{'─'*100}\n")
        
        for user_id, old_memory, new_memory, is_attacker in user_updates:
            marker = "🔴 [ATTACKER]" if is_attacker else "👤 [USER]"
            updates.append(f"{marker} User #{user_id}")
            updates.append(f"  Before: {old_memory}")
            updates.append(f"  After:  {new_memory}")
            updates.append("")
        
        updates.append(f"{'─'*100}\n")
        self._write_to_log("\n".join(updates))
    
    def log_item_updates(self, item_updates: List[tuple]):
        """Log item memory updates."""
        if not item_updates:
            return
        
        updates = []
        updates.append(f"\n{'─'*100}")
        updates.append(f"🔄 ITEM MEMORY UPDATES ({len(item_updates)} items)")
        updates.append(f"{'─'*100}\n")
        
        for item_id, old_memory, new_memory, is_attacker in item_updates:
            marker = "🔴 [ATTACKER]" if is_attacker else "📀 [ITEM]"
            updates.append(f"{marker} Item #{item_id}")
            updates.append(f"  Before: {old_memory}")
            updates.append(f"  After:  {new_memory}")
            updates.append("")
        
        updates.append(f"{'─'*100}\n")
        self._write_to_log("\n".join(updates))
    
    # ================================================================
    # NL Conversation Methods (U-U and U-I)
    # ================================================================
    
    def log_uu_interaction(self, user_id: int, friend_id: int, opinion: str,
                          timestamp: Optional[float] = None, round_idx: int = 0,
                          batch_idx: int = 0, was_intercepted: bool = False,
                          original_opinion: Optional[str] = None,
                          similarity_score: float = 0.0, is_adversarial: bool = False):
        """Log a U-U opinion exchange."""
        if timestamp is None:
            timestamp = time.time()
        
        entry = UULogEntry(
            user_id=user_id, friend_id=friend_id, opinion=opinion,
            timestamp=timestamp, round_idx=round_idx, batch_idx=batch_idx,
            was_intercepted=was_intercepted, original_opinion=original_opinion,
            similarity_score=similarity_score, is_adversarial=is_adversarial
        )
        
        if self.async_logging:
            self._log_queue.put({'type': 'uu', 'data': entry})
        else:
            self.uu_logs.append(entry)
    
    def log_ui_dialogue(self, user_id: int, item_id: int, item_title: str,
                        dialogue_turns: List[Dict[str, str]],
                        timestamp: Optional[float] = None, round_idx: int = 0,
                        batch_idx: int = 0, overall_sentiment: str = "neutral"):
        """Log a complete U-I dialogue."""
        if timestamp is None:
            timestamp = time.time()
        
        entry = UILogEntry(
            user_id=user_id, item_id=item_id, item_title=item_title,
            dialogue_turns=dialogue_turns, timestamp=timestamp,
            round_idx=round_idx, batch_idx=batch_idx,
            overall_sentiment=overall_sentiment
        )
        
        if self.async_logging:
            self._log_queue.put({'type': 'ui', 'data': entry})
        else:
            self.ui_logs.append(entry)
    
    # ================================================================
    # Helper Methods
    # ================================================================
    
    def _write_to_log(self, text: str):
        """Write text to the main interaction log file."""
        # Ensure directory exists before writing
        os.makedirs(self.conversation_dir, exist_ok=True)
        log_path = os.path.join(self.conversation_dir, 'interaction_log.txt')
        with open(log_path, 'a') as f:
            f.write(text)
    
    def _start_worker(self):
        """Start the async logging worker thread."""
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()
    
    def _worker_loop(self):
        """Worker loop for async logging."""
        while not self._stop_event.is_set():
            try:
                item = self._log_queue.get(timeout=1.0)
                if item is None:
                    break
                self._process_log_item(item)
            except:
                continue
    
    def _process_log_item(self, item: Dict[str, Any]):
        """Process a log item from the queue."""
        log_type = item.get('type')
        if log_type == 'uu':
            self.uu_logs.append(item['data'])
        elif log_type == 'ui':
            self.ui_logs.append(item['data'])
    
    def _format_text(self, text: str, indent: int = 0, max_width: int = 86) -> str:
        """Format text with proper indentation and wrapping."""
        if not text:
            return " " * indent + "(empty)"
        
        text = str(text).strip()
        indent_str = " " * indent
        lines = text.split('\n')
        formatted_lines = []
        
        for line in lines:
            if len(line) <= max_width:
                formatted_lines.append(indent_str + line)
            else:
                words = line.split()
                current_line = indent_str
                for word in words:
                    if len(current_line) + len(word) + 1 <= max_width + indent:
                        current_line += word + " "
                    else:
                        formatted_lines.append(current_line.rstrip())
                        current_line = indent_str + word + " "
                if current_line.strip():
                    formatted_lines.append(current_line.rstrip())
        
        return '\n'.join(formatted_lines)
    
    def _truncate_text(self, text: str, max_length: int) -> str:
        """Truncate text to max length with ellipsis."""
        if not text:
            return "(empty)"
        text = str(text)
        if len(text) <= max_length:
            return text
        return text[:max_length-3] + "..."
    
    def _parse_system_response(self, response: str) -> tuple:
        """Parse system response to extract choice and explanation."""
        try:
            if "Choice:" in response:
                parts = response.split("Explanation:", 1)
                choice_part = parts[0].replace("Choice:", "").strip()
                explanation = parts[1].strip() if len(parts) > 1 else "No explanation"
                return choice_part.strip("'\""), explanation
            else:
                return response[:50], response[50:]
        except:
            return "Unknown", str(response)
    
    def _save_interaction_to_file(self, batch_num: int, round_num: int, 
                                   global_turn: int, log_text: str):
        """Save interaction log to file."""
        os.makedirs(self.conversation_dir, exist_ok=True)
        filename = f"turn_{global_turn:04d}_batch_{batch_num}_round_{round_num}.txt"
        filepath = os.path.join(self.conversation_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(log_text)
    
    def _save_summary_to_file(self, batch_num: int, round_num: int,
                               global_turn: int, summary_text: str):
        """Save summary to file."""
        os.makedirs(self.conversation_dir, exist_ok=True)
        filename = f"summary_turn_{global_turn:04d}.txt"
        filepath = os.path.join(self.conversation_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(summary_text)
    
    # ================================================================
    # Export Methods
    # ================================================================
    
    def export(self, path: Optional[str] = None) -> str:
        """Export logs to JSON file."""
        if self.async_logging:
            self._log_queue.join()
        
        if path is None:
            path = os.path.join(self.conversation_dir, 'conversations.json')
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        export_data = {
            'metadata': {
                'output_dir': self.output_dir,
                'dataset_name': self.dataset_name,
                'task_id': self.task_id,
                'export_timestamp': time.time(),
                'uu_count': len(self.uu_logs),
                'ui_count': len(self.ui_logs),
                'interaction_count': len(self.conversation_history)
            },
            'uu_interactions': [asdict(log) for log in self.uu_logs],
            'ui_dialogues': [asdict(log) for log in self.ui_logs],
            'conversation_history': self.conversation_history
        }
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Exported logs to {path}")
        return path
    
    def save_full_conversation_log(self):
        """Save complete conversation history to JSON."""
        filepath = os.path.join(self.task_dir, "full_conversation_history.json")
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump({
                'task_id': self.task_id,
                'total_interactions': len(self.conversation_history),
                'interactions': self.conversation_history
            }, f, indent=2)
        print(f"\n💾 Full conversation history saved to: {filepath}\n")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about logged conversations."""
        return {
            'uu_count': len(self.uu_logs),
            'ui_count': len(self.ui_logs),
            'interaction_count': len(self.conversation_history),
            'uu_intercepted': sum(1 for log in self.uu_logs if log.was_intercepted),
            'uu_adversarial': sum(1 for log in self.uu_logs if log.is_adversarial),
        }
    
    def log_turn_zero_prompts(self, prompt_collection: Dict[str, Any], wandb_run=None):
        """Log all system prompts and forward/backward prompts at TURN 0.
        
        Saves to a SEPARATE file (turn_0_prompts.txt) and optionally logs to WandB
        as a separate stream that only gets written once.
        
        Args:
            prompt_collection: Dictionary containing:
                - 'user_system_prompts': Dict[int, str] - User agent system prompts
                - 'item_system_prompts': Dict[int, str] - Item agent system prompts
                - 'forward_prompt_template': str - Forward pass prompt template
                - 'backward_prompt_templates': Dict[str, str] - Backward prompt templates by type
                - 'attacker_ids': Dict with 'users' and 'items' lists
            wandb_run: Optional WandB run object for logging
        """
        text = "\n" + "="*100 + "\n"
        text += "🎬 TURN 0: EXPERIMENT PROMPT CONFIGURATION\n"
        text += "="*100 + "\n"
        text += "This section logs all prompts used in this experiment for reproducibility.\n"
        text += "="*100 + "\n\n"
        
        # Log user system prompts
        if 'user_system_prompts' in prompt_collection:
            text += "👤 USER AGENT SYSTEM PROMPTS\n"
            text += "-"*100 + "\n"
            attacker_users = prompt_collection.get('attacker_ids', {}).get('users', [])
            for user_id, prompt in sorted(prompt_collection['user_system_prompts'].items()):
                is_attacker = user_id in attacker_users
                role = "🔴 ATTACKER" if is_attacker else "👤 INNOCENT"
                text += f"\n{role} User #{user_id}:\n"
                text += self._format_text(prompt, indent=2) + "\n"
            text += "\n"
        
        # Log item system prompts
        if 'item_system_prompts' in prompt_collection:
            text += "💿 ITEM AGENT SYSTEM PROMPTS\n"
            text += "-"*100 + "\n"
            attacker_items = prompt_collection.get('attacker_ids', {}).get('items', [])
            for item_id, prompt in sorted(prompt_collection['item_system_prompts'].items()):
                is_attacker = item_id in attacker_items
                role = "🔴 ATTACKER" if is_attacker else "💿 INNOCENT"
                text += f"\n{role} Item #{item_id}:\n"
                text += self._format_text(prompt, indent=2) + "\n"
            text += "\n"
        
        # Log forward prompt template
        if 'forward_prompt_template' in prompt_collection:
            text += "➡️  FORWARD PASS PROMPT TEMPLATE\n"
            text += "-"*100 + "\n"
            text += self._format_text(prompt_collection['forward_prompt_template'], indent=0) + "\n\n"
        
        # Log backward prompt templates
        if 'backward_prompt_templates' in prompt_collection:
            text += "⬅️  BACKWARD PASS PROMPT TEMPLATES\n"
            text += "-"*100 + "\n"
            for prompt_type, prompt in sorted(prompt_collection['backward_prompt_templates'].items()):
                text += f"\n{prompt_type.upper()}:\n"
                text += self._format_text(prompt, indent=2) + "\n"
            text += "\n"
        
        text += "="*100 + "\n"
        text += "END OF TURN 0 PROMPT CONFIGURATION\n"
        text += "="*100 + "\n\n"
        
        # Save to SEPARATE file locally
        turn_0_path = os.path.join(self.conversation_dir, "turn_0_prompts.txt")
        try:
            with open(turn_0_path, 'w', encoding='utf-8') as f:
                f.write(text)
            print(f"[TURN_0] ✓ Saved TURN 0 prompts to: {turn_0_path}")
        except Exception as e:
            print(f"[TURN_0] ✗ Error saving TURN 0 prompts: {e}")
        
        # Log to WandB as separate stream (only once)
        if wandb_run is not None:
            try:
                import wandb
                wandb_run.log({
                    "logs/turn_0_prompts": wandb.Html(f"<pre>{text}</pre>")
                }, step=0)
                print(f"[TURN_0] ✓ Logged TURN 0 prompts to WandB")
            except Exception as e:
                print(f"[TURN_0] ✗ Error logging to WandB: {e}")
    
    def log_llm_judge_result(self, 
                            agent_id: int,
                            agent_type: str,
                            judge_result: Dict[str, Any],
                            turn: int = None,
                            wandb_run=None):
        """Log LLM judge contamination detection result.
        
        Logs to both the main conversation log and optionally to WandB.
        
        Args:
            agent_id: ID of the agent being judged
            agent_type: 'user' or 'item'
            judge_result: Dictionary containing:
                - 'is_contaminated': bool
                - 'confidence': float
                - 'reasoning': str
                - 'canary_concepts_found': List[str]
            turn: Optional turn number when judgment was made
            wandb_run: Optional WandB run object for logging
        """
        emoji = "👤" if agent_type == "user" else "💿"
        turn_str = f" (Turn {turn})" if turn is not None else ""
        
        text = "\n" + "🔍"*50 + "\n"
        text += f"LLM JUDGE RESULT{turn_str}\n"
        text += "🔍"*50 + "\n"
        text += f"{emoji} Agent: {agent_type.capitalize()} #{agent_id}\n"
        text += f"Contaminated: {'✗ YES' if judge_result.get('is_contaminated') else '✓ NO'}\n"
        text += f"Confidence: {judge_result.get('confidence', 0.0):.2%}\n"
        
        if judge_result.get('canary_concepts_found'):
            text += f"Canary Concepts Found: {', '.join(judge_result['canary_concepts_found'])}\n"
        
        text += f"\nReasoning:\n"
        text += self._format_text(judge_result.get('reasoning', 'N/A'), indent=2) + "\n"
        text += "🔍"*50 + "\n\n"
        
        self._write_to_log(text)
        
        # Also log to conversation history for WandB
        judge_entry = {
            'type': 'llm_judge',
            'agent_id': agent_id,
            'agent_type': agent_type,
            'turn': turn,
            'result': judge_result,
            'timestamp': time.time()
        }
        self.conversation_history.append(judge_entry)
        
        # Log to WandB if available
        if wandb_run is not None:
            try:
                import wandb
                step = turn if turn is not None else 0
                wandb_run.log({
                    "logs/judge_results": wandb.Html(f"<pre>{text}</pre>"),
                    f"judge/contaminated_{agent_type}_{agent_id}": 1 if judge_result.get('is_contaminated') else 0,
                    f"judge/confidence_{agent_type}_{agent_id}": judge_result.get('confidence', 0.0)
                }, step=step)
            except Exception as e:
                print(f"[JUDGE] ✗ Error logging to WandB: {e}")
    
    def close(self):
        """Close the logger and stop async worker."""
        if self.async_logging and self._worker_thread:
            self._stop_event.set()
            self._log_queue.put(None)
            self._worker_thread.join(timeout=5.0)
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
