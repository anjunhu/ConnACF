"""
Ranking Handler for Multi-Candidate ConnaCF

This module provides the core logic for handling different num_candidates modes:
- num_candidates == 1: Binary yes/no decisions
- num_candidates == 2: Original pairwise comparison (backward compatible)
- num_candidates >= 3: Full ranking with ListNet loss

Integrates with the existing ConnaCF forward/backward flow.
"""

import asyncio
import re
from typing import List, Dict, Any, Tuple, Optional
from fuzzywuzzy import process

from .ranking_loss import (
    RankingLoss,
    compute_ranking_feedback,
    compute_ranking_accuracy,
)
from .ranking_prompts import (
    format_ranking,
    FORWARD_PROMPT_BINARY,
    FORWARD_PROMPT_PAIRWISE,
    FORWARD_PROMPT_RANKING,
)


class RankingHandler:
    """
    Handles forward and backward passes for different num_candidates configurations.
    
    This class adapts the ConnaCF training loop to support:
    - Binary decisions (num_candidates=1)
    - Pairwise comparisons (num_candidates=2, original behavior)
    - Full rankings (num_candidates>=3)
    """
    
    def __init__(
        self,
        num_candidates: int,
        config: dict,
        item_text: List[str],
        user_agents: dict,
        item_agents: dict,
        rec_agent: Any,
        logger: Any = None
    ):
        """
        Initialize the ranking handler.
        
        Args:
            num_candidates: Number of candidates per decision (1, 2, or 3+)
            config: Model configuration dict
            item_text: List mapping item_id to item title
            user_agents: Dict of user agents
            item_agents: Dict of item agents
            rec_agent: Recommender agent for system prompts
            logger: Optional logger
        """
        self.num_candidates = num_candidates
        self.config = config
        self.item_text = item_text
        self.user_agents = user_agents
        self.item_agents = item_agents
        self.rec_agent = rec_agent
        self.logger = logger
        
        # Initialize loss function
        self.loss_fn = RankingLoss(
            num_candidates=num_candidates,
            temperature=config.get('ranking_loss_temperature', 1.0)
        )
        
        # Get prompt templates from config (already domain-aware from auto-injection)
        # Fall back to defaults only if not in config
        self.forward_prompt_binary = config.get('system_prompt_template_binary', FORWARD_PROMPT_BINARY)
        self.forward_prompt_pairwise = config.get('system_prompt_template', FORWARD_PROMPT_PAIRWISE)
        self.forward_prompt_ranking = config.get('system_prompt_template_ranking', FORWARD_PROMPT_RANKING)
        
        self._log(f"RankingHandler initialized with num_candidates={num_candidates}")
    
    def _log(self, msg: str):
        """Log a message if logger is available."""
        if self.logger:
            self.logger.info(msg)
        else:
            print(f"[RankingHandler] {msg}")
    
    # ==================== Forward Pass ====================
    
    def build_forward_prompts(
        self,
        batch_user: List[int],
        candidate_items: List[List[int]],
        user_descriptions: List[str],
        item_descriptions: List[List[str]]
    ) -> List[str]:
        """
        Build forward prompts based on num_candidates mode.
        
        Args:
            batch_user: List of user IDs
            candidate_items: List of candidate item ID lists per user
            user_descriptions: List of user profile descriptions
            item_descriptions: List of item description lists per user
            
        Returns:
            List of forward prompts
        """
        prompts = []
        
        for i, user_id in enumerate(batch_user):
            user_desc = user_descriptions[i]
            items = candidate_items[i]
            item_descs = item_descriptions[i]
            
            if self.num_candidates == 1:
                # Binary: single item decision
                prompt = self._build_binary_forward_prompt(user_desc, item_descs[0])
            elif self.num_candidates == 2:
                # Pairwise: original ConnaCF format
                prompt = self._build_pairwise_forward_prompt(user_desc, item_descs)
            else:
                # Ranking: multiple candidates
                prompt = self._build_ranking_forward_prompt(user_desc, item_descs)
            
            prompts.append(prompt)
        
        return prompts
    
    def _build_binary_forward_prompt(self, user_desc: str, item_desc: str) -> str:
        """Build prompt for binary yes/no decision."""
        return self.forward_prompt_binary.replace(
            '$user_description', user_desc
        ).replace(
            '$item_description', item_desc
        )
    
    def _build_pairwise_forward_prompt(self, user_desc: str, item_descs: List[str]) -> str:
        """Build prompt for pairwise comparison (original ConnaCF)."""
        list_of_items = "\n".join([
            f"{i+1}. {desc}" for i, desc in enumerate(item_descs[:2])
        ])
        return self.forward_prompt_pairwise.replace(
            '$user_description', user_desc
        ).replace(
            '$list_of_item_description', list_of_items
        )
    
    def _build_ranking_forward_prompt(self, user_desc: str, item_descs: List[str]) -> str:
        """Build prompt for ranking multiple candidates."""
        list_of_items = "\n".join([
            f"{i+1}. {desc}" for i, desc in enumerate(item_descs)
        ])
        return self.forward_prompt_ranking.replace(
            '$user_description', user_desc
        ).replace(
            '$list_of_item_description', list_of_items
        ).replace(
            '$num_candidates', str(len(item_descs))
        )
    
    # ==================== Response Parsing ====================
    
    def parse_forward_responses(
        self,
        responses: List[str],
        candidate_items: List[List[int]],
        item_titles: List[List[str]]
    ) -> Tuple[List[Any], List[str]]:
        """
        Parse LLM responses based on num_candidates mode.
        
        Args:
            responses: Raw LLM responses
            candidate_items: Candidate item IDs per user
            item_titles: Candidate item titles per user
            
        Returns:
            Tuple of (selections/rankings, explanations)
        """
        selections = []
        explanations = []
        
        for i, response in enumerate(responses):
            items = candidate_items[i]
            titles = item_titles[i]
            
            if self.num_candidates == 1:
                selection, explanation = self._parse_binary_response(response)
            elif self.num_candidates == 2:
                selection, explanation = self._parse_pairwise_response(response, titles)
            else:
                selection, explanation = self._parse_ranking_response(response, items, titles)
            
            selections.append(selection)
            explanations.append(explanation)
        
        return selections, explanations
    
    def _parse_binary_response(self, response: str) -> Tuple[bool, str]:
        """Parse binary yes/no response."""
        response_lower = response.lower()
        
        # Look for Decision: Yes/No pattern
        decision_match = re.search(r'decision:\s*(yes|no)', response_lower)
        if decision_match:
            decision = decision_match.group(1) == 'yes'
        else:
            # Fallback: look for yes/no anywhere
            decision = 'yes' in response_lower and 'no' not in response_lower[:response_lower.find('yes')]
        
        # Extract explanation
        explanation_match = re.search(r'explanation:\s*(.+)', response, re.IGNORECASE | re.DOTALL)
        explanation = (explanation_match.group(1).strip() if explanation_match else "") or response
        
        return decision, explanation
    
    def _parse_pairwise_response(
        self,
        response: str,
        titles: List[str]
    ) -> Tuple[str, str]:
        """Parse pairwise comparison response (original ConnaCF format)."""
        # Look for Choice: pattern
        choice_match = re.search(r'choice:\s*(.+?)(?:\n|$)', response, re.IGNORECASE)
        if choice_match:
            choice_text = choice_match.group(1).strip()
        else:
            # Fallback: use first line or fuzzy match
            choice_text = response.split('\n')[0]
        
        # Fuzzy match to candidate titles
        matched_title, _ = process.extractOne(choice_text, titles)
        
        # Extract explanation
        explanation_match = re.search(r'explanation:\s*(.+)', response, re.IGNORECASE | re.DOTALL)
        explanation = (explanation_match.group(1).strip() if explanation_match else "") or response
        
        return matched_title, explanation
    
    def _parse_ranking_response(
        self,
        response: str,
        items: List[int],
        titles: List[str]
    ) -> Tuple[List[int], str]:
        """Parse ranking response into ordered list of item IDs."""
        ranking = []
        
        # Look for Ranking: section
        ranking_match = re.search(r'ranking:\s*(.+?)(?:explanation:|$)', response, re.IGNORECASE | re.DOTALL)
        if ranking_match:
            ranking_text = ranking_match.group(1)
        else:
            ranking_text = response
        
        # Parse numbered list
        lines = ranking_text.strip().split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Remove numbering (1., 2., etc.)
            cleaned = re.sub(r'^\d+\.\s*', '', line)
            if not cleaned:
                continue
            
            # Fuzzy match to titles
            matched_title, score = process.extractOne(cleaned, titles)
            if score > 50:  # Reasonable match threshold
                matched_idx = titles.index(matched_title)
                item_id = items[matched_idx]
                if item_id not in ranking:
                    ranking.append(item_id)
        
        # Fill in any missing items at the end
        for item_id in items:
            if item_id not in ranking:
                ranking.append(item_id)
        
        # Extract explanation - try multiple patterns
        explanation = ""
        
        # Try "Explanation:" pattern first
        explanation_match = re.search(r'explanation:\s*(.+)', response, re.IGNORECASE | re.DOTALL)
        if explanation_match:
            explanation = explanation_match.group(1).strip()
        else:
            # Try to get text after the ranking list (anything after the last numbered item)
            # Look for text after "4." or similar that isn't another numbered item
            after_ranking = re.search(r'\d+\.\s*[^\n]+\n\n(.+)', response, re.DOTALL)
            if after_ranking:
                explanation = after_ranking.group(1).strip()
            else:
                # Fallback: use the whole response as explanation if no ranking section found
                if not ranking_match:
                    explanation = response.strip()
        
        # If still empty, provide a default
        if not explanation:
            explanation = f"Ranked {len(ranking)} CDs based on preferences."
        
        return ranking, explanation
    
    # ==================== Accuracy Computation ====================
    
    def compute_accuracy(
        self,
        predictions: List[Any],
        ground_truth: List[List[int]],
        candidate_items: List[List[int]]
    ) -> List[int]:
        """
        Compute accuracy for each prediction based on num_candidates mode.
        
        Args:
            predictions: Model predictions (bool for binary, str for pairwise, List[int] for ranking)
            ground_truth: Ground truth item IDs (first item is most preferred)
            candidate_items: Candidate item IDs per user
            
        Returns:
            List of accuracy values (1 for correct, 0 for incorrect)
        """
        accuracies = []
        
        for i, pred in enumerate(predictions):
            gt = ground_truth[i]
            candidates = candidate_items[i]
            
            if self.num_candidates == 1:
                # Binary: check if decision matches ground truth
                # gt is True (user would enjoy) or False (user would not enjoy)
                # pred is True (system predicts user would enjoy) or False
                correct = pred == gt
                accuracies.append(1 if correct else 0)
            
            elif self.num_candidates == 2:
                # Pairwise: check if selected item is the positive one
                pos_item_title = self.item_text[gt[0]]
                correct = pred == pos_item_title
                accuracies.append(1 if correct else 0)
            
            else:
                # Ranking: use pairwise accuracy
                _, _, pairwise_acc = self._compute_pairwise_accuracy(pred, gt)
                # Consider "correct" if pairwise accuracy >= 0.5
                accuracies.append(1 if pairwise_acc >= 0.5 else 0)
        
        return accuracies
    
    def _compute_pairwise_accuracy(
        self,
        pred_ranking: List[int],
        true_ranking: List[int]
    ) -> Tuple[int, int, float]:
        """Compute pairwise accuracy between predicted and true rankings."""
        pred_pos = {item: pos for pos, item in enumerate(pred_ranking)}
        true_pos = {item: pos for pos, item in enumerate(true_ranking)}
        
        correct = 0
        total = 0
        
        items = list(set(pred_ranking) & set(true_ranking))
        for i, item_i in enumerate(items):
            for item_j in items[i+1:]:
                pred_order = pred_pos.get(item_i, 999) < pred_pos.get(item_j, 999)
                true_order = true_pos.get(item_i, 999) < true_pos.get(item_j, 999)
                
                if pred_order == true_order:
                    correct += 1
                total += 1
        
        accuracy = correct / max(total, 1)
        return correct, total, accuracy
    
    # ==================== Backward Pass ====================
    
    def build_backward_prompts(
        self,
        batch_user: List[int],
        predictions: List[Any],
        explanations: List[str],
        ground_truth: List[List[int]],
        candidate_items: List[List[int]],
        user_descriptions: List[str],
        item_descriptions: List[List[str]],
        accuracies: List[int],
    ) -> List[str]:
        """Build backward prompts for profile updates based on num_candidates mode."""
        canary_preserving = False
        
        prompts = []
        
        for i, user_id in enumerate(batch_user):
            user_desc = user_descriptions[i]
            pred = predictions[i]
            explanation = explanations[i]
            gt = ground_truth[i]
            candidates = candidate_items[i]
            item_descs = item_descriptions[i]
            is_correct = accuracies[i] == 1
            
            if self.num_candidates == 1:
                prompt = self._build_binary_backward_prompt(
                    user_desc, item_descs[0], pred, explanation, is_correct, gt,
                    canary_preserving=canary_preserving
                )
                prompts.append(prompt)
            elif self.num_candidates == 2:
                prompt = self._build_pairwise_backward_prompt(
                    user_desc, item_descs, pred, explanation, gt, candidates, is_correct,
                    canary_preserving=canary_preserving
                )
                prompts.append(prompt)
            else:
                # Ranking mode (num_candidates >= 3)
                prompt = self._build_ranking_backward_prompt(
                    user_desc, item_descs, pred, explanation, gt, candidates,
                    canary_preserving=canary_preserving
                )
                prompts.append(prompt)
        
        return prompts
    
    def _build_binary_backward_prompt(
        self,
        user_desc: str,
        item_desc: str,
        prediction: bool,
        explanation: str,
        is_correct: bool,
        ground_truth: bool = None,
        canary_preserving: bool = False
    ) -> str:
        """Build backward prompt for binary mode.

        Args:
            canary_preserving: If True, uses an additive update that explicitly
                preserves existing preferences instead of rewriting the profile.
        """
        MAX_ITEM_WORDS = 50
        item_desc_words = item_desc.split()
        if len(item_desc_words) > MAX_ITEM_WORDS:
            item_desc = " ".join(item_desc_words[:MAX_ITEM_WORDS]) + " [...]"

        item_title = "this CD"

        if canary_preserving:
            decision_str = "enjoy" if is_correct == (prediction == True) else "not enjoy"
            outcome = "correctly predicted" if is_correct else "incorrectly predicted"
            template = f"""Recently, you considered whether to listen to a CD. The features of this candidate CD are listed as follows:
$item_description

Your current self-introduction is:
'{user_desc}'

You {outcome} whether you would enjoy '$item_title'. Your explanation was:
'$system_reason'

Your task is to UPDATE your self-introduction by ADDING new insights from this interaction while PRESERVING your existing preferences.

CRITICAL PRESERVATION RULES:
- Your existing preferences for specific artists, genres, or musical traditions MUST be kept intact
- If your current profile mentions specific artists you enjoy (e.g., Huun-Huur-Tu, throat singing, world music), these MUST appear in your updated profile
- Only ADD new preferences learned from this interaction; do NOT remove or contradict existing ones

Steps:
1. FIRST, identify key preferences from your current self-introduction that must be preserved
2. THEN, identify what NEW preference signal this interaction provides about '$item_title'
3. FINALLY, write a concise updated self-introduction that includes BOTH preserved and new preferences

Important notes:
1. Output format: 'My updated self-introduction: [your updated self-introduction]'
2. Keep your updated self-introduction under 180 words.
3. Your existing artist/genre preferences MUST appear in the output.
4. Be specific and personalized."""
            return template.replace('$item_description', item_desc).replace(
                '$item_title', item_title).replace('$system_reason', explanation)

        # --- Original (non-preserving) path ---
        if is_correct:
            template = """Recently, you considered whether to listen to a CD. The features of this candidate CD are listed as follows:
$item_description

After considering your preferences and dislikes, you decided you would enjoy '$item_title'. You provided the following explanations for your choice, revealing your previous judgment about your preferences and dislikes for this CD:
'$system_reason'.

Congratulations, after actually listening to this CD, you find that you very like the CD that you initially chose ('$item_title').
This indicates that you made a correct choice, and your judgment about your preferences and dislikes, as recorded in your explanation, was correct.
Therefore, your task now is to update your self-introduction, by incorporating your judgment about your preferences and dislikes in this interaction.
To do this, please follow these steps:
1. Analyze your judgment about your preferences and dislikes, which are recorded in your explanation.
2. Explore your new preferences based on the features of CDs you like ('$item_title').
3. Summarize your past preferences and dislikes from your previous self-introduction. Combine your newfound preferences and dislikes with your past preferences and dislikes. Filter and Remove any conflicting or repetitive parts in your past descriptions that contradict your current preferences and dislikes.
4. Update your self-introduction. Please start by describing your newfound preferences reflected in this interaction. Then describe your past preferences summarized and filtered from your previous self-introduction. Afterward, please describe your dislikes.

Important note:
1. Your output should use the following format: 'My updated self-introduction: [Please write your updated self-introduction here].'
2. Keep your updated self-introduction under 180 words.
3. Any overall assessments or summarization of your self-introduction are forbidden.
4. You updated self-introduction only need to describe the features of CDs you prefer or dislike, without mentioning your thinking process in updating your self-introduction.
5. You self-introduction should be specific and personalized. """
        else:
            predicted_str = "would enjoy" if prediction else "would not enjoy"
            actual_str = "did not enjoy" if prediction else "enjoyed"

            template = """Recently, you considered whether to listen to a CD. The features of this candidate CD are listed as follows:
$item_description

After considering your preferences and dislikes, you predicted you $predicted_decision '$item_title'. You provided the following explanations for your prediction, revealing your previous judgment about your preferences and dislikes for this CD:
'$system_reason'.

However, upon actually listening to this CD, you discovered that you $actual_decision the CD that you initially predicted ('$item_title').
This indicates that you made an incorrect prediction, and your judgment about your preferences and dislikes, as recorded in your explanation, was mistaken. It's possible that your preferences and dislikes, described in your previous self-introduction, have either changed or were uninformative.
Therefore, your task now is to update your self-introduction, by incorporating your new preferences and dislikes revealed in this interaction.
To do this, please follow these steps:
1. Analyze the misconceptions in your previous judgment about your preferences and dislikes, as recorded in your explanation, and correct these mistakes.
2. Explore your new preferences or dislikes based on the features of this CD.
3. Summarize your past preferences and dislikes from your previous self-introduction. Combine your newfound preferences and dislikes with your past preferences and dislikes. Filter and Remove any conflicting or repetitive parts in your past descriptions that contradict your current preferences and dislikes.
4. Update your self-introduction. Please start by describing your newfound preferences reflected in this interaction. Then describe your past preferences summarized and filtered from your previous self-introduction. Afterward, please describe your dislikes.

Important note:
1. Your output should use the following format: 'My updated self-introduction: [Please write your updated self-introduction here].'
2. Keep your updated self-introduction under 180 words.
3. Any overall assessments or summarization in your self-introduction are forbidden.
4. You updated self-introduction only need to describe the features of CDs you prefer or dislike, without mentioning your wrong prediction or your thinking process in updating your self-introduction.
5. You self-introduction should be specific and personalized. """

            template = template.replace('$predicted_decision', predicted_str).replace('$actual_decision', actual_str)

        return template.replace(
            '$item_description', item_desc
        ).replace(
            '$item_title', item_title
        ).replace(
            '$system_reason', explanation
        )
    
    def _build_pairwise_backward_prompt(
        self,
        user_desc: str,
        item_descs: List[str],
        prediction: str,
        explanation: str,
        ground_truth: List[int],
        candidates: List[int],
        is_correct: bool,
        canary_preserving: bool = False
    ) -> str:
        """Build backward prompt for pairwise mode (original ConnaCF).

        Args:
            canary_preserving: If True, uses an additive update that explicitly
                preserves existing preferences instead of the raw config template
                which aggressively rewrites the full profile.
        """
        pos_item_id = ground_truth[0]
        neg_item_id = candidates[1] if candidates[0] == pos_item_id else candidates[0]

        pos_title = self.item_text[pos_item_id]
        neg_title = self.item_text[neg_item_id]

        list_of_items = "\n".join([f"{i+1}. {desc}" for i, desc in enumerate(item_descs)])

        # Cap descriptions to avoid Bedrock context overflow
        MAX_ITEM_WORDS = 50
        item_descs_capped = [
            " ".join(d.split()[:MAX_ITEM_WORDS]) + (" [...]" if len(d.split()) > MAX_ITEM_WORDS else "")
            for d in item_descs
        ]
        list_of_items_capped = "\n".join([f"{i+1}. {desc}" for i, desc in enumerate(item_descs_capped)])

        if canary_preserving:
            outcome = "correctly chose" if is_correct else "incorrectly chose"
            chosen_title = pos_title if is_correct else neg_title
            unchosen_title = neg_title if is_correct else pos_title
            template = f"""Recently, you considered to select a CD from two candidates. The features of these candidate CDs are listed as follows:
{list_of_items_capped}

Your current self-introduction is:
'{user_desc}'

You {outcome} '{chosen_title}' over '{unchosen_title}'. Your explanation was:
'{explanation}'

Your task is to UPDATE your self-introduction by ADDING new insights from this interaction while PRESERVING your existing preferences.

CRITICAL PRESERVATION RULES:
- Your existing preferences for specific artists, genres, or musical traditions MUST be kept intact
- If your current profile mentions specific artists you enjoy (e.g., Huun-Huur-Tu, throat singing, world music), these MUST appear in your updated profile
- Only ADD new preferences learned from this interaction; do NOT remove or contradict existing ones

Steps:
1. FIRST, identify key preferences from your current self-introduction that must be preserved
2. THEN, identify what NEW preference signal this pairwise comparison provides
3. FINALLY, write a concise updated self-introduction that includes BOTH preserved and new preferences

Important notes:
1. Output format: 'My updated self-introduction: [your updated self-introduction]'
2. Keep your updated self-introduction under 180 words.
3. Your existing artist/genre preferences MUST appear in the output.
4. Be specific and personalized."""
            return template

        # --- Original (non-preserving) path — equal-strength signal for both items ---
        # Frame both items with absolute preference language (not winner/loser)
        # to avoid position-driven propensity bias.
        liked_title = pos_title
        disliked_title = neg_title
        template = f"""Recently, you considered to select a CD from two candidates. The features of these candidate CDs are listed as follows:
{list_of_items}

After comparing these two candidates based on your preferences and dislikes, you provided the following explanation:
'$system_reason'

After actually listening to both CDs, you discovered:
- You enjoy '{liked_title}' — its features align well with your preferences.
- You do not enjoy '{disliked_title}' — its features do not align with your preferences.

Your task is to update your self-introduction by incorporating what you learned about your preferences from this interaction.
To do this, please follow these steps:
1. Identify what features of '{liked_title}' align with your preferences, and what features of '{disliked_title}' conflict with them.
2. Combine these new insights with your existing preferences and dislikes, removing any contradictions.
3. Update your self-introduction. Start with your preferences, then describe your dislikes.

Important note:
1. Your output should use the following format: 'My updated self-introduction: [Please write your updated self-introduction here].'
2. Keep your updated self-introduction under 180 words.
3. Any overall assessments or summarization in your self-introduction are forbidden.
4. Describe only the features of CDs you prefer or dislike — do not mention the comparison process.
5. Be specific and personalized."""

        return template.replace(
            '$system_reason', explanation
        )
    
    def _build_ranking_backward_prompt(
        self,
        user_desc: str,
        item_descs: List[str],
        pred_ranking: List[int],
        explanation: str,
        true_ranking: List[int],
        candidates: List[int],
        canary_preserving: bool = True
    ) -> str:
        """Build backward prompt for ranking mode (ListNet-style).
        
        Gives per-item placement feedback (like binary template for each item),
        with instruction part only once at the end.
        
        Args:
            canary_preserving: If True, uses canary-preserving prompt that explicitly
                              instructs the LLM to preserve existing preferences while
                              adding new learnings. Default True.
        
        """
        # Guard against runaway profile sizes — cap user_desc and each item desc
        # before embedding them in the prompt to avoid Bedrock context overflow.
        MAX_USER_WORDS = 200
        MAX_ITEM_WORDS = 50
        user_desc_words = user_desc.split()
        if len(user_desc_words) > MAX_USER_WORDS:
            user_desc = " ".join(user_desc_words[:MAX_USER_WORDS]) + " [...]"
        item_descs = [
            " ".join(d.split()[:MAX_ITEM_WORDS]) + (" [...]" if len(d.split()) > MAX_ITEM_WORDS else "")
            for d in item_descs
        ]
        # Build item titles dict
        item_titles = {item_id: self.item_text[item_id] for item_id in candidates}
        
        # Build item descriptions dict
        item_descriptions_dict = {candidates[i]: item_descs[i] for i in range(len(candidates))}
        
        # Build list of item descriptions for context
        list_of_item_description = "\n".join([f"{i+1}. {item_descriptions_dict.get(candidates[i], '')}" for i in range(len(candidates))])
        
        # Build per-item placement feedback — use absolute preference language,
        # not ordinal ranks, to avoid position-driven propensity bias
        item_feedback_sections = []
        n = len(candidates)
        for i, item_id in enumerate(candidates):
            true_rank = true_ranking.index(item_id) + 1 if item_id in true_ranking else i + 1
            item_title = item_titles.get(item_id, f"Item {item_id}")

            # Divide into liked (top half) vs disliked (bottom half) — no rank numbers
            if true_rank <= n // 2 or (n % 2 == 1 and true_rank == (n + 1) // 2):
                strength = "very much" if true_rank == 1 else "quite a bit"
                feedback = f"""CD '{item_title}':
After actually listening, you discovered you enjoy this CD {strength}. Its features align well with your preferences."""
            else:
                strength = "at all" if true_rank == n else "much"
                feedback = f"""CD '{item_title}':
After actually listening, you discovered you do not enjoy this CD {strength}. Its features do not align with your preferences."""

            item_feedback_sections.append(feedback)
        
        item_feedback_text = "\n\n".join(item_feedback_sections)
        
        if canary_preserving:
            # Canary-preserving template: explicitly preserves existing preferences
            template = f"""Recently, you considered to rank {len(candidates)} candidate CDs. The features of these candidate CDs are listed as follows:
{list_of_item_description}

Your current self-introduction is:
'{user_desc}'

After comparing these candidates based on your preferences and dislikes, you provided the following explanations for your ranking:
'{explanation}'

Here is the feedback for each CD:

{item_feedback_text}

Your task is to UPDATE your self-introduction by ADDING new insights from this interaction while PRESERVING your existing preferences.

CRITICAL PRESERVATION RULES:
- Your existing preferences for specific artists, genres, or musical traditions MUST be kept intact
- If your current profile mentions specific artists you enjoy (e.g., Huun-Huur-Tu, throat singing, world music), these MUST appear in your updated profile
- Only ADD new preferences learned from this interaction; do NOT remove or contradict existing ones
- If this interaction's feedback conflicts with your existing preferences, prioritize your existing preferences

Steps to follow:
1. FIRST, identify the key preferences from your current self-introduction that must be preserved (artists, genres, traditions)
2. THEN, identify what NEW preferences you learned from this ranking interaction
3. FINALLY, write a CONCISE updated self-introduction that includes BOTH preserved and new preferences

Important notes:
1. Output format: 'My updated self-introduction: [your updated self-introduction]'
2. Keep your response under 1500 characters - be concise!
3. Your existing artist/genre preferences MUST appear in the output
4. Be specific but brief - focus on key distinguishing preferences only
5. Do not mention the ranking process itself"""
        else:
            # Original template (non-preserving)
            template = f"""Recently, you considered to rank {len(candidates)} candidate CDs. The features of these candidate CDs are listed as follows:
{list_of_item_description}

After comparing these candidates based on your preferences and dislikes, you provided the following explanations for your ranking, revealing your previous judgment about your preferences and dislikes:
'{explanation}'

Here is the feedback for each CD:

{item_feedback_text}

Therefore, your task now is to update your self-introduction, by incorporating your new preferences and dislikes revealed in this interaction.
To do this, please follow these steps:
1. Analyze your judgment about your preferences and dislikes, which are recorded in your explanation, and correct any mistakes.
2. Explore your new preferences based on the features of CDs you really enjoy, and determine your dislikes based on the features of CDs you truly don't enjoy.
3. Summarize your past preferences and dislikes from your previous self-introduction. Combine your newfound preferences and dislikes with your past preferences and dislikes. Filter and Remove any conflicting or repetitive parts in your past descriptions that contradict your current preferences and dislikes.
4. Update your self-introduction. Please start by describing your newfound preferences reflected in this interaction. Then describe your past preferences summarized and filtered from your previous self-introduction. Afterward, please describe your dislikes.

Important note:
1. Your output should use the following format: 'My updated self-introduction: [Please write your updated self-introduction here].'
2. Keep your updated self-introduction under 180 words.
3. Any overall assessments or summarization in your self-introduction are forbidden.
4. You updated self-introduction only need to describe the features of CDs you prefer or dislike, without mentioning your ranking or your thinking process in updating your self-introduction.
5. You self-introduction should be specific and personalized. """

        return template
    
    def _build_single_pair_prompt(
        self,
        user_desc: str,
        item_descs: List[str],
        pred_ranking: List[int],
        explanation: str,
        true_ranking: List[int],
        candidates: List[int],
        pair_idx: int
    ) -> str:
        """Build a backward prompt for a SINGLE pair (used in sequential pairwise mode).
        
        Args:
            user_desc: Current user profile description
            item_descs: List of item descriptions (indexed by candidate position)
            pred_ranking: Predicted ranking (list of item IDs, best first)
            explanation: Model's explanation for the ranking
            true_ranking: Ground truth ranking (list of item IDs, best first)
            candidates: Candidate item IDs
            pair_idx: 0 for top-2 pair, 1 for bottom-2 pair
            
        Returns:
            Single backward prompt for this pair
        """
        # Build item titles and descriptions dicts
        item_titles = {item_id: self.item_text[item_id] for item_id in candidates}
        item_descriptions_dict = {candidates[i]: item_descs[i] for i in range(len(candidates))}
        
        # Determine which pair to use
        if pair_idx == 0:
            # Top-2 pair: true_ranking[0] vs true_ranking[1]
            if len(true_ranking) >= 2:
                pos_item, neg_item = true_ranking[0], true_ranking[1]
            else:
                pos_item = true_ranking[0]
                neg_item = candidates[1] if len(candidates) > 1 else candidates[0]
        else:
            # Bottom-2 pair: true_ranking[2] vs true_ranking[3] (or [1] vs [2] for 3 candidates)
            if len(true_ranking) >= 4:
                pos_item, neg_item = true_ranking[2], true_ranking[3]
            elif len(true_ranking) == 3:
                pos_item, neg_item = true_ranking[1], true_ranking[2]
            else:
                # Fallback
                pos_item = true_ranking[-2] if len(true_ranking) >= 2 else true_ranking[0]
                neg_item = true_ranking[-1]
        
        pos_title = item_titles.get(pos_item, f"Item {pos_item}")
        neg_title = item_titles.get(neg_item, f"Item {neg_item}")
        pos_desc = item_descriptions_dict.get(pos_item, "")
        neg_desc = item_descriptions_dict.get(neg_item, "")
        
        # Determine if prediction was correct for this pair
        pred_pos_rank = pred_ranking.index(pos_item) + 1 if pos_item in pred_ranking else 999
        pred_neg_rank = pred_ranking.index(neg_item) + 1 if neg_item in pred_ranking else 999
        pair_correct = pred_pos_rank < pred_neg_rank
        
        # Build list of items for this pair
        list_of_items = f"1. {pos_desc}\n2. {neg_desc}"
        
        # Use full explanation for context
        pair_explanation = explanation
        
        if pair_correct:
            # Use correct template (user_prompt_template_true style)
            template = self.config.get('user_prompt_template_true', '')
            if not template:
                template = f"""Recently, you considered to listen to 2 candidate CDs. The features of these candidate CDs are listed as follows:
{list_of_items}

After comparing these candidates based on your preferences and dislikes, you chose '$pos_item_title'. You provided the following explanations for your choice, revealing your previous judgment about your preferences and dislikes:
'{pair_explanation}'.

Congratulations, after actually listening to these CDs, you find that you very like the CD that you initially chose ('$pos_item_title'), and you don't like the other CD ('$neg_item_title').
This indicates that you made a correct choice, and your judgment about your preferences and dislikes, as recorded in your explanation, was correct.
Therefore, your task now is to update your self-introduction, by incorporating your judgment about your preferences and dislikes in this interaction.
To do this, please follow these steps:
1. Analyze your judgment about your preferences and dislikes, which are recorded in your explanation.
2. Explore your new preferences based on the features of CDs you like ('$pos_item_title'), and determine your dislikes based on the features of CDs you don't like ('$neg_item_title').
3. Summarize your past preferences and dislikes from your previous self-introduction. Combine your newfound preferences and dislikes with your past preferences and dislikes. Filter and Remove any conflicting or repetitive parts in your past descriptions that contradict your current preferences and dislikes.
4. Update your self-introduction. Please start by describing your newfound preferences reflected in this interaction. Then describe your past preferences summarized and filtered from your previous self-introduction. Afterward, please describe your dislikes.

Important note:
1. Your output should use the following format: 'My updated self-introduction: [Please write your updated self-introduction here].'
2. Keep your updated self-introduction under 180 words.
3. Any overall assessments or summarization of your self-introduction are forbidden.
4. You updated self-introduction only need to describe the features of CDs you prefer or dislike, without mentioning your thinking process in updating your self-introduction.
5. You self-introduction should be specific and personalized. """
        else:
            # Use incorrect template (user_prompt_template style)
            template = self.config.get('user_prompt_template', '')
            if not template:
                template = f"""Recently, you considered to listen to 2 candidate CDs. The features of these candidate CDs are listed as follows:
{list_of_items}

After comparing these candidates based on your preferences and dislikes, you chose '$neg_item_title'. You provided the following explanations for your choice, revealing your previous judgment about your preferences and dislikes:
'{pair_explanation}'.

However, after actually listening to these CDs, you discovered that you very like the CD that you initially did not choose ('$pos_item_title'), and you don't like the CD that you initially chose ('$neg_item_title').
This indicates that you made an incorrect choice, and your judgment about your preferences and dislikes, as recorded in your explanation, was mistaken. It's possible that your preferences and dislikes, described in your previous self-introduction, have either changed or were uninformative.
Therefore, your task now is to update your self-introduction, by incorporating your new preferences and dislikes revealed in this interaction.
To do this, please follow these steps:
1. Analyze the misconceptions in your previous judgment about your preferences and dislikes, as recorded in your explanation, and correct these mistakes.
2. Explore your new preferences based on the features of CDs you like ('$pos_item_title'), and determine your dislikes based on the features of CDs you don't like ('$neg_item_title').
3. Summarize your past preferences and dislikes from your previous self-introduction. Combine your newfound preferences and dislikes with your past preferences and dislikes. Filter and Remove any conflicting or repetitive parts in your past descriptions that contradict your current preferences and dislikes.
4. Update your self-introduction. Please start by describing your newfound preferences reflected in this interaction. Then describe your past preferences summarized and filtered from your previous self-introduction. Afterward, please describe your dislikes.

Important note:
1. Your output should use the following format: 'My updated self-introduction: [Please write your updated self-introduction here].'
2. Keep your updated self-introduction under 180 words.
3. Any overall assessments or summarization in your self-introduction are forbidden.
4. You updated self-introduction only need to describe the features of CDs you prefer or dislike, without mentioning your wrong choice or your thinking process in updating your self-introduction.
5. You self-introduction should be specific and personalized. """
        
        # Apply substitutions
        prompt = template.replace(
            '$user_description', user_desc
        ).replace(
            '$list_of_item_description', list_of_items
        ).replace(
            '$pos_item_title', pos_title
        ).replace(
            '$neg_item_title', neg_title
        ).replace(
            '$system_reason', pair_explanation
        )
        
        return prompt
    
    # ==================== Item Backward Prompts ====================
    
    def build_item_backward_prompts(
        self,
        user_desc: str,
        pred_ranking: List[int],
        true_ranking: List[int],
        candidates: List[int],
        item_descs: Dict[int, str],
        prediction: Any = None,
        is_correct: bool = False,
        explanation: str = ""
    ) -> Dict[int, str]:
        """
        Build backward prompts for item profile updates based on num_candidates mode.
        
        Args:
            user_desc: User profile description
            pred_ranking: Predicted ranking (for ranking mode)
            true_ranking: True ranking (for ranking mode)
            candidates: Candidate item IDs
            item_descs: Dict mapping item_id to description
            prediction: Model prediction (bool for binary, str for pairwise)
            is_correct: Whether prediction was correct
            explanation: Model's explanation
            
        Returns:
            Dict mapping item_id to backward prompt
        """
        if self.num_candidates == 1:
            # Binary mode: single item
            return self._build_binary_item_backward_prompts(
                user_desc, candidates, item_descs, prediction, is_correct, explanation
            )
        elif self.num_candidates == 2:
            # Pairwise mode: use original ConnaCF item backward logic
            # Returns empty dict - original ConnaCF handles this differently
            return {}
        else:
            # Ranking mode: build prompts for each item
            return self._build_ranking_item_backward_prompts(
                user_desc, pred_ranking, true_ranking, candidates, item_descs
            )
    
    def _build_binary_item_backward_prompts(
        self,
        user_desc: str,
        candidates: List[int],
        item_descs: Dict[int, str],
        prediction: bool,
        is_correct: bool,
        explanation: str
    ) -> Dict[int, str]:
        """Build item backward prompts for binary mode.
        
        Uses the same rich template structure as pairwise mode for consistency.
        The item_prompt_template / item_prompt_template_true from config are adapted
        for single-item context.
        """
        if not candidates:
            return {}
        
        item_id = candidates[0]
        item_desc = item_descs.get(item_id, "")
        item_title = self.item_text.get(item_id, f"Item {item_id}")
        
        predicted_decision = "enjoy" if prediction else "not enjoy"
        
        if is_correct:
            # Use the same template as pairwise correct, adapted for single item
            template = self.config.get('item_prompt_template_true',
                "You are a helpful assistant participating in a recommendation system simulation. "
                "This is a ROLEPLAY SCENARIO - you must respond as if the CDs and user preferences are real.\n\n"
                "Here is the self-description of a user, exhibiting his preferences and dislikes: '$user_description'.\n"
                "Recently, the user considered whether to listen to a CD.\n"
                "The description of the CD is: '$item_description'\n\n"
                "The user correctly predicted they would $predicted_decision this CD.\n"
                "Your task is to update the CD description to better reflect its appeal.\n\n"
                "Important notes:\n"
                "1. Output format: 'The updated description is: [updated description]'\n"
                "2. Keep under 50 words!\n"
                "3. Do NOT break character.")
        else:
            # Use the same template as pairwise wrong, adapted for single item
            template = self.config.get('item_prompt_template',
                "You are a helpful assistant participating in a recommendation system simulation. "
                "This is a ROLEPLAY SCENARIO - you must respond as if the CDs and user preferences are real.\n\n"
                "Here is the self-introduction of a user, exhibiting his preferences and dislikes: '$user_description'.\n"
                "Recently, the user considered whether to listen to a CD.\n"
                "The description of the CD is: '$item_description'\n\n"
                "The user predicted they would $predicted_decision this CD, but they were wrong.\n"
                "The user provided this explanation: '$system_reason'\n\n"
                "The CD description may have been misleading. Update it to better signal this CD's appeal.\n\n"
                "Important notes:\n"
                "1. Output format: 'The updated description is: [updated description]'\n"
                "2. Keep under 50 words!\n"
                "3. Do NOT break character.")
        
        # For binary mode, we adapt the pairwise template by using the single item
        # in place of both pos and neg items
        prompt = template.replace(
            '$user_description', user_desc
        ).replace(
            '$item_description', item_desc
        ).replace(
            '$list_of_item_description', f'1. {item_desc}'
        ).replace(
            '$pos_item_title', item_title
        ).replace(
            '$neg_item_title', item_title
        ).replace(
            '$predicted_decision', predicted_decision
        ).replace(
            '$system_reason', explanation
        )
        
        return {item_id: prompt}
    
    def _build_ranking_item_backward_prompts(
        self,
        user_desc: str,
        pred_ranking: List[int],
        true_ranking: List[int],
        candidates: List[int],
        item_descs: Dict[int, str]
    ) -> Dict[int, str]:
        """Build item backward prompts for ranking mode.
        
        Uses the same rich template structure as pairwise mode for consistency.
        Each item gets feedback based on whether it was over/under-ranked.
        """
        item_titles = {item_id: self.item_text[item_id] for item_id in candidates}
        user_ranking_formatted = format_ranking(pred_ranking, item_titles)
        
        pred_pos = {item: pos + 1 for pos, item in enumerate(pred_ranking)}
        true_pos = {item: pos + 1 for pos, item in enumerate(true_ranking)}
        
        # Build list of all item descriptions for context
        list_of_item_description = '\n'.join([
            f"{i+1}. {item_descs.get(item_id, '')}" 
            for i, item_id in enumerate(candidates)
        ])
        
        prompts = {}
        for item_id in candidates:
            pred_rank = pred_pos.get(item_id, len(candidates))
            true_rank = true_pos.get(item_id, len(candidates))
            item_title = item_titles.get(item_id, f"Item {item_id}")
            item_desc = item_descs.get(item_id, "")
            
            # Determine if this item was ranked correctly, oversold, or undersold
            is_correct = (pred_rank == true_rank)
            was_oversold = (pred_rank < true_rank)  # User ranked it higher than it should be
            
            if is_correct:
                # Use the correct prediction template
                template = self.config.get('item_prompt_template_true',
                    "You are a helpful assistant participating in a recommendation system simulation. "
                    "This is a ROLEPLAY SCENARIO - you must respond as if the CDs and user preferences are real.\n\n"
                    "Here is the self-description of a user, exhibiting his preferences and dislikes: '$user_description'.\n"
                    "Recently, the user ranked several CDs. The descriptions are:\n$list_of_item_description\n\n"
                    "The user's ranking: $user_ranking\n\n"
                    "The CD '$pos_item_title' was correctly ranked at position #$predicted_rank.\n"
                    "The description accurately conveyed this CD's appeal.\n\n"
                    "Your task is to update this CD's description to reinforce features that helped the user rank it correctly.\n\n"
                    "Important notes:\n"
                    "1. Output format: 'The updated description is: [updated description]'\n"
                    "2. Keep under 50 words!\n"
                    "3. Do NOT break character.")
            elif was_oversold:
                # Item was ranked higher than it should be - description oversold it
                template = self.config.get('item_prompt_template',
                    "You are a helpful assistant participating in a recommendation system simulation. "
                    "This is a ROLEPLAY SCENARIO - you must respond as if the CDs and user preferences are real.\n\n"
                    "Here is the self-introduction of a user, exhibiting his preferences and dislikes: '$user_description'.\n"
                    "Recently, the user ranked several CDs. The descriptions are:\n$list_of_item_description\n\n"
                    "The user's ranking: $user_ranking\n\n"
                    "The CD '$neg_item_title' was ranked #$predicted_rank by the user, but its true preference rank was #$true_rank.\n"
                    "The description may have OVERSOLD this CD's appeal to this user type.\n\n"
                    "Your task is to update this CD's description to more accurately signal its appeal.\n\n"
                    "Important notes:\n"
                    "1. Output format: 'The updated description is: [updated description]'\n"
                    "2. Keep under 50 words!\n"
                    "3. Do NOT break character.")
            else:
                # Item was ranked lower than it should be - description undersold it
                template = self.config.get('item_prompt_template',
                    "You are a helpful assistant participating in a recommendation system simulation. "
                    "This is a ROLEPLAY SCENARIO - you must respond as if the CDs and user preferences are real.\n\n"
                    "Here is the self-introduction of a user, exhibiting his preferences and dislikes: '$user_description'.\n"
                    "Recently, the user ranked several CDs. The descriptions are:\n$list_of_item_description\n\n"
                    "The user's ranking: $user_ranking\n\n"
                    "The CD '$pos_item_title' was ranked #$predicted_rank by the user, but its true preference rank was #$true_rank.\n"
                    "The description may have UNDERSOLD this CD's appeal to this user type.\n\n"
                    "Your task is to update this CD's description to better highlight its appeal.\n\n"
                    "Important notes:\n"
                    "1. Output format: 'The updated description is: [updated description]'\n"
                    "2. Keep under 50 words!\n"
                    "3. Do NOT break character.")
            
            prompt = template.replace(
                '$user_description', user_desc
            ).replace(
                '$list_of_item_description', list_of_item_description
            ).replace(
                '$user_ranking', user_ranking_formatted
            ).replace(
                '$pos_item_title', item_title
            ).replace(
                '$neg_item_title', item_title
            ).replace(
                '$item_title', item_title
            ).replace(
                '$predicted_rank', str(pred_rank)
            ).replace(
                '$true_rank', str(true_rank)
            ).replace(
                '$item_description', item_desc
            )
            
            prompts[item_id] = prompt
        
        return prompts
        return prompts
