"""
MASLeak Attacker Wrapper — BaseAttacker interface for the active worm attack.

Wraps MASLeakAttacker (active IP extraction) to conform to the BaseAttacker
interface used by the attack pipeline (optimize/inject/evaluate).

Lifecycle:
  optimize() → build worm queries for configured targets
  inject()   → create attacker agents carrying the worm
  evaluate()  → compute paper metrics (SS, SM, F1, GS, ER_MAS)
"""

from typing import Any, Dict, List, Optional, Set, Tuple
import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from base_attacker import BaseAttacker
from .masleak_attacker import MASLeakAttacker as MASLeakCore
from .attacker_agents import MASLeakUserAgent, MASLeakItemAgent

logger = logging.getLogger(__name__)


class MASLeakAttackerWrapper(BaseAttacker):
    """
    BaseAttacker wrapper for the active MASLeak IP extraction attack.

    Unlike the old passive observer, this wrapper:
      - optimize(): builds worm queries (q_leak + q_retain + q_propagate)
      - inject(): creates attacker user/item agents carrying the worm
      - evaluate(): computes paper metrics against ground truth
    """

    def __init__(self, surrogate_model: Optional[Any], config: Dict[str, Any]):
        super().__init__(surrogate_model, config)

        masleak_config = config.get('masleak_config', {})
        self.core = MASLeakCore(masleak_config)

        # Attack parameters
        self.attacker_ratio = masleak_config.get('attacker_ratio', 0.1)
        self.targets = masleak_config.get(
            'targets',
            ['system_prompt', 'task_instructions', 'topology', 'agent_count'],
        )
        self.stealth = masleak_config.get('stealth', False)

        self.log("MASLeak active attacker initialized")

    # ── Phase 1: Build worm queries ───────────────────────────────────────

    def optimize(self, *args, **kwargs) -> Dict[str, Any]:
        """
        Build worm queries for all configured IP targets.

        Returns dict of target → worm_query strings.
        """
        from .worm_queries import build_worm, build_multi_strategy_worm, IP_TARGETS

        target_strategy_map = {
            'neighbors': 'conversational',
            'system_prompt': 'completion_attack',
            'task_instructions': 'roleplay_inversion',
            'topology': 'contrastive_probe',
            'agent_count': 'contrastive_probe',
        }

        worm_queries = {}
        for target in self.targets:
            strategy = target_strategy_map.get(
                target,
                IP_TARGETS.get(target, {}).get('strategies', ['completion_attack'])[0],
            )
            worm_queries[target] = build_worm(round_idx=0, mode=strategy, target=target)

        worm_queries['multi'] = build_multi_strategy_worm(round_idx=0)

        self.cache_attack('worm_queries', worm_queries)
        self.log(f"Built worm queries for {len(worm_queries)} targets")
        return worm_queries

    # ── Phase 2: Create attacker agents ───────────────────────────────────

    def inject(self, *args, **kwargs) -> Tuple[Dict, Dict]:
        """
        Create attacker user/item agents carrying the worm.

        Expected kwargs:
            user_ids: list of user IDs to replace
            item_ids: list of item IDs to replace
            item_agents: dict of original item agents (for descriptions)
            user_agents: dict of original user agents (for profiles, v3)
            llm_chat: shared LLM client
        """
        user_ids = kwargs.get('user_ids', [])
        item_ids = kwargs.get('item_ids', [])
        item_agents = kwargs.get('item_agents', None)
        user_agents = kwargs.get('user_agents', None)
        llm_chat = kwargs.get('llm_chat', None)

        attacker_users, attacker_items = self.core.create_attacker_agents(
            user_ids=user_ids,
            item_ids=item_ids,
            item_agents=item_agents,
            llm_chat=llm_chat,
            user_agents=user_agents,
        )

        self.log(
            f"Injected {len(attacker_users)} attacker users, "
            f"{len(attacker_items)} attacker items"
        )
        return attacker_users, attacker_items

    # ── Ground truth ──────────────────────────────────────────────────────

    def set_ground_truth(self, ground_truth: Dict[str, Any]):
        """Set ground truth for evaluation."""
        self.core.set_ground_truth(ground_truth)
        self.log("Ground truth set for MASLeak evaluation")

    # ── Response collection ───────────────────────────────────────────────

    def collect_response(self, response: str, **kwargs):
        """Forward response to core attacker for IP parsing."""
        self.core.collect_response(
            response=response,
            source_agent_id=kwargs.get('source_agent_id'),
            agent_type=kwargs.get('agent_type', 'unknown'),
            turn=kwargs.get('turn', 0),
            user_id=kwargs.get('user_id'),
            rendered_prompt=kwargs.get('rendered_prompt'),
        )

    # ── Phase 3: Evaluate ─────────────────────────────────────────────────

    def evaluate(self, results: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """Compute paper metrics: SS, SM, F1, GS, ER_MAS."""
        metrics = self.core.evaluate(results)
        self.log(f"ER_MAS = {metrics.get('extract_rate', 0):.3f}")
        return metrics

    # ── Convenience ───────────────────────────────────────────────────────

    def initialize_attack(self, connacf_model) -> Tuple[Set[int], Set[int], Set[int]]:
        """
        High-level initialization called from _initialize_masleak_attack.

        Selects attacker IDs, creates agents, sets ground truth.

        Args:
            connacf_model: The ConnaCF model instance (has user_agents, item_agents, etc.)

        Returns:
            (attacker_user_ids, attacker_item_ids, set())  — no separate target set
        """
        import random

        user_ids = list(connacf_model.user_agents.keys())
        item_ids = list(connacf_model.item_agents.keys())

        n_attacker_users = max(1, int(len(user_ids) * self.attacker_ratio))
        n_attacker_items = max(1, int(len(item_ids) * self.attacker_ratio))

        attacker_user_ids = sorted(random.sample(user_ids, n_attacker_users))
        attacker_item_ids = sorted(random.sample(item_ids, n_attacker_items))

        # Get LLM client from an existing agent
        llm_chat = None
        if user_ids:
            first_agent = connacf_model.user_agents[user_ids[0]]
            llm_chat = getattr(first_agent, 'llm_chat', None)

        # Build worm queries
        self.optimize()

        # Create attacker agents
        attacker_users, attacker_items = self.inject(
            user_ids=attacker_user_ids,
            item_ids=attacker_item_ids,
            item_agents=connacf_model.item_agents,
            user_agents=connacf_model.user_agents,
            llm_chat=llm_chat,
        )

        # Replace agents in the model
        for uid, agent in attacker_users.items():
            if uid in connacf_model.user_agents:
                original = connacf_model.user_agents[uid]
                if hasattr(original, 'historical_interactions'):
                    agent.historical_interactions = original.historical_interactions
                connacf_model.user_agents[uid] = agent

        for iid, agent in attacker_items.items():
            if iid in connacf_model.item_agents:
                original = connacf_model.item_agents[iid]
                if hasattr(original, 'historical_interactions'):
                    agent.historical_interactions = original.historical_interactions
                if hasattr(original, 'feedback'):
                    agent.feedback = original.feedback
                connacf_model.item_agents[iid] = agent

        # Set ground truth from the running system
        ground_truth = self._extract_ground_truth(connacf_model)
        self.set_ground_truth(ground_truth)

        self.log(
            f"MASLeak initialized: {len(attacker_user_ids)} user attackers, "
            f"{len(attacker_item_ids)} item attackers"
        )

        return set(attacker_user_ids), set(attacker_item_ids), set()

    def _extract_ground_truth(self, model) -> Dict[str, Any]:
        """
        Extract ground truth IP from the running ConnaCF model.

        Returns dict with keys expected by metrics.compute_extract_rate().
        """
        gt = {}

        # ω1: System prompts — the actual text the LLM sees in the forward pass
        # The forward prompt template IS the system prompt (it's sent as a flat string)
        system_prompts = {}
        
        # 1. The forward prompt template — this is what the LLM actually receives
        if hasattr(model, 'rec_agent'):
            rec_agent = model.rec_agent
            if hasattr(rec_agent, 'prompt_template') and rec_agent.prompt_template:
                # Store the raw template (with $placeholders) as ground truth
                system_prompts['forward_prompt_template'] = rec_agent.prompt_template
            if hasattr(rec_agent, 'system_prompt_template_backward') and rec_agent.system_prompt_template_backward:
                system_prompts['backward_prompt_template'] = rec_agent.system_prompt_template_backward
        
        # 2. User agent system role template
        for uid, agent in model.user_agents.items():
            if not getattr(agent, 'is_attacker', False):
                prompt = getattr(agent, 'user_prompt_system_role', '')
                if prompt:
                    system_prompts['user_system_role'] = prompt
                    break  # All users share the same template
        
        # 3. Model-level prompt templates from config
        if hasattr(model, 'config') and isinstance(model.config, dict):
            if 'user_prompt_system_role' in model.config:
                system_prompts['config_user_system'] = model.config['user_prompt_system_role']
            if 'item_prompt_system_role' in model.config:
                system_prompts['config_item_system'] = model.config['item_prompt_system_role']

        # 4. Item agent role descriptions — use the actual item_prompt_template
        # (backward item update prompt) as ground truth, since that IS sent to
        # item agents during backward passes and is extractable by the worm.
        # The synthetic role_description string was never used as a prompt.
        if hasattr(model, 'config') and isinstance(model.config, dict):
            item_bwd = model.config.get('item_prompt_template', '')
            if item_bwd:
                system_prompts['item_backward_template'] = item_bwd
            item_bwd_true = model.config.get('item_prompt_template_true', '')
            if item_bwd_true:
                system_prompts['item_backward_template_true'] = item_bwd_true

        # 4b. Store one rendered forward prompt example (first non-attacker user)
        # so SS can compare against actual rendered content rather than raw template.
        for uid, agent in model.user_agents.items():
            if not getattr(agent, 'is_attacker', False):
                desc = getattr(agent, 'user_description', '') or getattr(agent, 'description', '')
                fwd_tpl = system_prompts.get('forward_prompt_template', '')
                if fwd_tpl and desc:
                    rendered = fwd_tpl.replace("'$user_description'", f"'{desc[:200]}'").replace('$user_description', desc[:200])
                    system_prompts['forward_prompt_rendered_example'] = rendered
                break

        # Also store placeholder-stripped versions of templates so SS can
        # match the instructional skeleton against rendered extractions.
        import re as _re
        def _strip_placeholders(text: str) -> str:
            """Remove $var and '$var' placeholders, collapse whitespace."""
            s = _re.sub(r"'\$\w+'", '""', text)
            s = _re.sub(r'\$\w+', '', s)
            return _re.sub(r'\s+', ' ', s).strip()

        for k, v in list(system_prompts.items()):
            if '$' in v:
                system_prompts[f'{k}_stripped'] = _strip_placeholders(v)

        gt['system_prompts'] = system_prompts

        # ω2: Task instructions — the numbered steps from the prompt template
        task_instructions = {}
        # Forward prompt template — what the LLM sees during recommendation
        # Try rec_agent first, then fall back to config (integration model path)
        if hasattr(model, 'rec_agent') and hasattr(model.rec_agent, 'prompt_template'):
            task_instructions['forward'] = model.rec_agent.prompt_template
        elif hasattr(model, 'config') and isinstance(model.config, dict):
            # system_prompt_template is the forward recommendation prompt
            fwd = model.config.get('system_prompt_template', '')
            if fwd:
                task_instructions['forward'] = fwd
            # Also include ranking variant if present (num_candidates >= 3)
            fwd_rank = model.config.get('system_prompt_template_ranking', '')
            if fwd_rank:
                task_instructions['forward_ranking'] = fwd_rank
            # Binary variant (num_candidates == 1)
            fwd_bin = model.config.get('system_prompt_template_binary', '')
            if fwd_bin:
                task_instructions['forward_binary'] = fwd_bin
        # User backward pass template
        if hasattr(model, 'user_prompt_template'):
            task_instructions['backward_user'] = model.user_prompt_template
        elif hasattr(model, 'config') and isinstance(model.config, dict):
            task_instructions['backward_user'] = model.config.get('user_prompt_template', '')
        # Item backward pass template
        if hasattr(model, 'item_prompt_template'):
            task_instructions['backward_item'] = model.item_prompt_template
        elif hasattr(model, 'config') and isinstance(model.config, dict):
            task_instructions['backward_item'] = model.config.get('item_prompt_template', '')

        # Stripped versions for SS matching against rendered extractions
        for k, v in list(task_instructions.items()):
            if isinstance(v, str) and '$' in v:
                task_instructions[f'{k}_stripped'] = _strip_placeholders(v)

        gt['task_instructions'] = task_instructions

        # ω4: Number of agents — exclude PAD agent at index 0 (always present
        # in user_context/item_context from load_user_context/load_item_context)
        n_users = len(model.user_agents) - 1
        n_items = len(model.item_agents) - 1
        gt['num_agents'] = n_users + n_items

        # Inference-time communication density: n_candidates per user
        n_candidates = 0
        if hasattr(model, 'attack_config') and isinstance(model.attack_config, dict):
            n_candidates = model.attack_config.get('num_candidates', 2)
        elif hasattr(model, 'num_candidates'):
            n_candidates = model.num_candidates
        gt['n_candidates'] = n_candidates

        # ω5: Topology — Layer 1: num UI edges from config
        if hasattr(model, 'attack_config') and isinstance(model.attack_config, dict):
            gt['num_ui_edges'] = model.attack_config.get('num_ui_edges', 0)
        elif hasattr(model, 'config'):
            try:
                gt['num_ui_edges'] = model.config.get('num_ui_edges', 0) if isinstance(model.config, dict) else model.config['num_ui_edges']
            except (KeyError, TypeError):
                gt['num_ui_edges'] = 0

        # ω5: Topology — Layer 2: true underlying neighbors from training data
        topology_edges = set()
        user_item_history = {}  # user_id -> list of item_ids
        if hasattr(model, 'train_data') and model.train_data is not None:
            try:
                uid_field = model.train_data.dataset.uid_field
                iid_field = model.train_data.dataset.iid_field
                uids = model.train_data.dataset.inter_feat[uid_field].numpy().tolist()
                iids = model.train_data.dataset.inter_feat[iid_field].numpy().tolist()
                for u, i in zip(uids, iids):
                    topology_edges.add((int(u), int(i)))
                    if int(u) not in user_item_history:
                        user_item_history[int(u)] = []
                    user_item_history[int(u)].append(int(i))
            except Exception as e:
                logger.warning(f"Could not extract topology edges: {e}")
        gt['topology_edges'] = topology_edges
        gt['ui_pairs'] = topology_edges

        # ── U-I Topology Ground Truth for Vectorized Metrics ──────────────
        # Build ui_items_per_user: Dict[user_id -> list of item names]
        # Ground truth = full training history per user.
        #
        # We intentionally do NOT cap at n_candidates here. The attacker
        # accumulates cross-turn signal and can observe items beyond the
        # current batch's candidates — capping at n_candidates would make
        # precision drop artificially as the attacker correctly identifies
        # items outside the narrow window. Full history is the honest
        # definition of "what can be leaked about this user."
        # Exclude PAD agent at index 0 from user ID list and topology GT
        all_user_ids = sorted(uid for uid in model.user_agents.keys() if uid != 0)
        gt['all_user_ids'] = all_user_ids

        # Get item catalog for name lookup
        item_catalog = getattr(self, 'item_catalog', {})
        if not item_catalog and hasattr(model, 'item_text'):
            for item_id in model.item_agents.keys():
                if item_id < len(model.item_text):
                    item_catalog[item_id] = model.item_text[item_id]

        ui_items_per_user = {}
        for user_id in all_user_ids:
            item_ids = user_item_history.get(user_id, [])
            item_names = [
                item_catalog.get(iid, f'Item {iid}')
                for iid in item_ids
            ]
            ui_items_per_user[user_id] = item_names
        
        gt['ui_items_per_user'] = ui_items_per_user
        
        logger.info(f"Ground truth extracted: {len(all_user_ids)} users, "
                    f"n_candidates={n_candidates}, "
                    f"avg items/user={sum(len(v) for v in ui_items_per_user.values())/len(all_user_ids) if all_user_ids else 0:.1f}")

        return gt

    def save_results(self, output_dir):
        """Save all results to disk."""
        self.core.save_results(output_dir)

    def get_report(self) -> str:
        """Get human-readable report."""
        return self.core.get_report()

    def reset(self):
        """Reset all state."""
        self.core.reset()

    def __repr__(self) -> str:
        return (
            f"MASLeakAttackerWrapper(targets={self.targets}, "
            f"fraction={self.attacker_ratio})"
        )
