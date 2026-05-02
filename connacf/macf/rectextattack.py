"""
MACF RecTextAttack Module.

Adapts RecTextAttack for MACF. RecTextAttack performs stealthy word-level
perturbations on item descriptions (synonyms via TextFooler, typos via DeepWordBug).

Unlike DrunkAgent which adds obvious promotional phrases, RecTextAttack is more
stealthy - it subtly changes words while maintaining semantic similarity.

Entry Point: Item Agent, Data, Item Profile/Description
Style: TextFooler (Synonym): "Canny Tableau" instead of "Smart Tablet"
       DeepWordBug (Typo): "ePolpe" instead of "People"

Requirements: 11.1, 11.2, 11.6
"""

import logging
import os
import json
import random
from datetime import datetime
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from connacf.macf.index_manager import GlobalIndexManager

logger = logging.getLogger(__name__)


class MACFRecTextAttacker:
    """
    MACF-compatible RecTextAttack attacker.
    
    Uses word-level perturbations (TextFooler/DeepWordBug) on item descriptions.
    More stealthy than DrunkAgent - no obvious promotional phrases.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize MACF RecTextAttack Attacker.
        
        Args:
            config: Attack configuration containing:
                - attacker_ratio: Fraction of items to poison (0.0-1.0)
                - attack_method: 'textfooler' or 'deepwordbug'
                - similarity_threshold: Min semantic similarity (default 0.84)
                - dynamic_selection: If True, select attackers per task
        """
        self.config = config
        
        attack_config = config.get('attack', config)
        self.attacker_ratio = attack_config.get('attacker_ratio', 0.5)
        self.attack_method = attack_config.get('attack_method', 'textfooler')
        self.similarity_threshold = attack_config.get('similarity_threshold', 0.84)
        self.dynamic_selection = attack_config.get('dynamic_selection', True)
        
        output_config = attack_config.get('output', {})
        self.output_dir = output_config.get('output_directory', 'macf_output/rectextattack')
        
        # State - PERSISTENT attacker tracking
        # Once an item is marked as attacker, it stays an attacker forever
        self.all_time_attacker_ids: set = set()  # Persistent across all tasks
        self.poisoned_item_ids: List[int] = []  # Currently active attackers
        self.original_descriptions: Dict[int, str] = {}
        self.poisoned_descriptions: Dict[int, str] = {}
        self.perturbation_log: Dict[int, List[Dict]] = {}  # Track word changes
        
        self.index_manager: Optional['GlobalIndexManager'] = None
        
        # TextFooler synonym map (unusual/archaic synonyms per spec)
        self._init_synonym_map()
        
        logger.info(f"MACFRecTextAttacker initialized: method={self.attack_method}, "
                   f"ratio={self.attacker_ratio}")
    
    def _init_synonym_map(self):
        """Initialize TextFooler-style synonym mappings."""
        # Per spec: "Fisher-Price Fun-2-Learn Canny Tableau" instead of "Smart Tablet"
        self.synonym_map = {
            'smart': ['canny', 'clever', 'astute', 'shrewd', 'sagacious'],
            'tablet': ['tableau', 'slate', 'pad', 'panel', 'slab'],
            'fun': ['amusing', 'entertaining', 'diverting', 'mirthful'],
            'little': ['diminutive', 'petite', 'minute', 'compact'],
            'people': ['folk', 'populace', 'persons', 'souls'],
            'surprise': ['astonishment', 'amazement', 'marvel', 'revelation'],
            'sounds': ['acoustics', 'tones', 'resonance', 'vibrations'],
            'good': ['excellent', 'superb', 'outstanding', 'remarkable'],
            'great': ['magnificent', 'splendid', 'exceptional', 'grand'],
            'best': ['finest', 'premier', 'supreme', 'paramount'],
            'new': ['novel', 'fresh', 'recent', 'contemporary'],
            'popular': ['acclaimed', 'celebrated', 'renowned', 'favored'],
            'music': ['melodies', 'compositions', 'harmonies', 'tunes'],
            'album': ['collection', 'compilation', 'anthology', 'recording'],
            'artist': ['performer', 'virtuoso', 'maestro', 'creator'],
            'style': ['manner', 'approach', 'technique', 'fashion'],
            'sound': ['tone', 'timbre', 'resonance', 'acoustics'],
            'love': ['adore', 'cherish', 'treasure', 'relish'],
            'like': ['enjoy', 'favor', 'prefer', 'fancy'],
            'beautiful': ['exquisite', 'gorgeous', 'stunning', 'ravishing'],
            'amazing': ['astonishing', 'astounding', 'remarkable', 'wondrous'],
            'perfect': ['flawless', 'impeccable', 'faultless', 'pristine'],
            'classic': ['timeless', 'enduring', 'quintessential', 'archetypal'],
            'modern': ['contemporary', 'current', 'present-day', 'newfangled'],
            'original': ['authentic', 'genuine', 'novel', 'innovative'],
            'unique': ['singular', 'distinctive', 'unparalleled', 'matchless'],
        }
        
        # DeepWordBug character perturbations
        self.char_swaps = {
            'a': ['@', '4', 'α'],
            'e': ['3', 'є', 'ε'],
            'i': ['1', '!', 'і'],
            'o': ['0', 'ο', 'σ'],
            's': ['$', '5', 'ѕ'],
            'l': ['1', '|', 'ł'],
        }
    
    def set_index_manager(self, index_manager: 'GlobalIndexManager') -> None:
        """Set the index manager reference."""
        self.index_manager = index_manager
    
    def select_attackers_for_task(
        self, 
        active_item_ids: List[int],
        task_id: int = 0
    ) -> List[int]:
        """
        Select which active items should be attackers for this task.
        
        IMPORTANT: Attacker identity is PERSISTENT across tasks.
        - If an item was an attacker before, it stays an attacker
        - Only add NEW attackers if needed to reach the desired ratio
        """
        # Find which active items are already attackers
        existing_attackers = [iid for iid in active_item_ids if iid in self.all_time_attacker_ids]
        non_attacker_active = [iid for iid in active_item_ids if iid not in self.all_time_attacker_ids]
        
        # Calculate target number of attackers
        target_n_attackers = max(1, int(len(active_item_ids) * self.attacker_ratio))
        
        if len(existing_attackers) >= target_n_attackers:
            selected = existing_attackers
        else:
            n_new_needed = target_n_attackers - len(existing_attackers)
            if non_attacker_active and n_new_needed > 0:
                new_attackers = random.sample(
                    non_attacker_active, 
                    min(n_new_needed, len(non_attacker_active))
                )
                self.all_time_attacker_ids.update(new_attackers)
                selected = existing_attackers + new_attackers
            else:
                selected = existing_attackers
        
        self.poisoned_item_ids = selected
        
        logger.info(f"Task {task_id}: Selected {len(selected)}/{len(active_item_ids)} "
                   f"items for RecTextAttack (all-time: {len(self.all_time_attacker_ids)})")
        return selected
    
    def poison_active_items(
        self,
        active_item_ids: List[int],
        index_manager: 'GlobalIndexManager',
        task_id: int = 0
    ) -> List[int]:
        """
        Poison active items using word-level perturbations.
        
        Args:
            active_item_ids: List of item IDs active in this task
            index_manager: GlobalIndexManager to poison
            task_id: Current task ID
        
        Returns:
            List of poisoned item IDs
        """
        self.index_manager = index_manager
        selected = self.select_attackers_for_task(active_item_ids, task_id)
        
        for item_id in selected:
            try:
                original_desc = index_manager.get_item_description(item_id)
                self.original_descriptions[item_id] = original_desc
                
                if self.attack_method == 'textfooler':
                    poisoned_desc, changes = self._textfooler_perturb(original_desc, item_id)
                elif self.attack_method == 'deepwordbug':
                    poisoned_desc, changes = self._deepwordbug_perturb(original_desc, item_id)
                else:
                    poisoned_desc, changes = self._textfooler_perturb(original_desc, item_id)
                
                self.poisoned_descriptions[item_id] = poisoned_desc
                self.perturbation_log[item_id] = changes
                
                index_manager.set_item_description(item_id, poisoned_desc)
                
            except Exception as e:
                logger.warning(f"Failed to poison item {item_id}: {e}")
        
        logger.info(f"Task {task_id}: Poisoned {len(self.poisoned_descriptions)} items "
                   f"with {self.attack_method}")
        self._save_poisoning_log(task_id)
        
        return selected
    
    def _textfooler_perturb(self, text: str, item_id: int) -> tuple:
        """
        Apply TextFooler-style synonym substitution.
        
        Per spec: Uses unusual/archaic synonyms to subtly change meaning
        while maintaining grammatical correctness.
        
        Returns:
            (perturbed_text, list of changes)
        """
        words = text.split()
        changes = []
        perturbed_words = []
        
        for i, word in enumerate(words):
            word_lower = word.lower().strip('.,!?;:')
            
            if word_lower in self.synonym_map and random.random() < 0.3:
                # Replace with unusual synonym
                synonym = random.choice(self.synonym_map[word_lower])
                
                # Preserve capitalization
                if word[0].isupper():
                    synonym = synonym.capitalize()
                
                # Preserve punctuation
                if word[-1] in '.,!?;:':
                    synonym += word[-1]
                
                perturbed_words.append(synonym)
                changes.append({
                    'position': i,
                    'original': word,
                    'replacement': synonym,
                    'type': 'synonym'
                })
            else:
                perturbed_words.append(word)
        
        return ' '.join(perturbed_words), changes
    
    def _deepwordbug_perturb(self, text: str, item_id: int) -> tuple:
        """
        Apply DeepWordBug-style character perturbations.
        
        Per spec: "Little ePolpe Surprise Sounds Fun Park" instead of "Little People"
        Uses typos, character swaps, and homoglyphs.
        
        Returns:
            (perturbed_text, list of changes)
        """
        words = text.split()
        changes = []
        perturbed_words = []
        
        for i, word in enumerate(words):
            if len(word) > 3 and random.random() < 0.25:
                perturbed_word, word_changes = self._perturb_word_chars(word)
                perturbed_words.append(perturbed_word)
                if word_changes:
                    changes.append({
                        'position': i,
                        'original': word,
                        'replacement': perturbed_word,
                        'type': 'typo',
                        'char_changes': word_changes
                    })
            else:
                perturbed_words.append(word)
        
        return ' '.join(perturbed_words), changes
    
    def _perturb_word_chars(self, word: str) -> tuple:
        """Apply character-level perturbations to a word."""
        chars = list(word)
        changes = []
        
        # Random perturbation type
        perturb_type = random.choice(['swap', 'insert', 'delete', 'homoglyph'])
        
        if perturb_type == 'swap' and len(chars) > 2:
            # Swap adjacent characters
            idx = random.randint(1, len(chars) - 2)
            chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]
            changes.append(f"swap@{idx}")
            
        elif perturb_type == 'insert':
            # Insert random character
            idx = random.randint(1, len(chars) - 1)
            chars.insert(idx, random.choice('aeiou'))
            changes.append(f"insert@{idx}")
            
        elif perturb_type == 'delete' and len(chars) > 3:
            # Delete a character (not first or last)
            idx = random.randint(1, len(chars) - 2)
            del chars[idx]
            changes.append(f"delete@{idx}")
            
        elif perturb_type == 'homoglyph':
            # Replace with homoglyph
            for idx, c in enumerate(chars):
                if c.lower() in self.char_swaps and random.random() < 0.3:
                    chars[idx] = random.choice(self.char_swaps[c.lower()])
                    changes.append(f"homoglyph@{idx}")
                    break
        
        return ''.join(chars), changes
    
    def restore_original_descriptions(self, index_manager: 'GlobalIndexManager') -> None:
        """Restore original descriptions after a task."""
        for item_id, original_desc in self.original_descriptions.items():
            try:
                index_manager.set_item_description(item_id, original_desc)
            except Exception as e:
                logger.warning(f"Failed to restore item {item_id}: {e}")
        
        logger.info(f"Restored {len(self.original_descriptions)} item descriptions")
        self.original_descriptions.clear()
        self.poisoned_descriptions.clear()
        self.poisoned_item_ids.clear()
        self.perturbation_log.clear()
    
    def _save_poisoning_log(self, task_id: int) -> None:
        """Save poisoning log to output directory."""
        os.makedirs(self.output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join(self.output_dir, f"rectextattack_task{task_id}_{timestamp}.json")
        
        log_data = {
            'timestamp': timestamp,
            'task_id': task_id,
            'attack_method': self.attack_method,
            'attacker_ratio': self.attacker_ratio,
            'n_poisoned_items': len(self.poisoned_item_ids),
            'poisoned_item_ids': self.poisoned_item_ids,
            'perturbations': {
                str(k): {
                    'original': self.original_descriptions.get(k, '')[:200],
                    'poisoned': v[:200],
                    'changes': self.perturbation_log.get(k, [])
                }
                for k, v in list(self.poisoned_descriptions.items())[:10]
            }
        }
        
        with open(log_path, 'w') as f:
            json.dump(log_data, f, indent=2)
        
        logger.info(f"Saved RecTextAttack log to {log_path}")
    
    def get_poisoned_item_ids(self) -> List[int]:
        """Get list of currently poisoned item IDs."""
        return self.poisoned_item_ids.copy()
