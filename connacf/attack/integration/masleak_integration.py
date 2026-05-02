"""
MASLeak Active Attack Integration with ConnaCF

Provides helpers for wiring the MASLeak worm attack into the ConnaCF
training loop. The main integration point is _initialize_masleak_attack()
in connacf_attack_integration.py — this module provides ground-truth
extraction and response collection utilities.
"""

from typing import Dict, List, Any, Optional, Set, Tuple
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)


def extract_ground_truth_from_model(model) -> Dict[str, Any]:
    """
    Extract ground truth IP from a running ConnaCF model.

    Returns dict with keys expected by metrics.compute_extract_rate():
        system_prompts, task_instructions, num_agents,
        topology_edges, num_ui_edges, ui_pairs
    """
    gt: Dict[str, Any] = {}

    # ω1: System prompts
    system_prompts = {}
    for uid, agent in model.user_agents.items():
        if not getattr(agent, 'is_attacker', False):
            prompt = getattr(agent, 'user_prompt_system_role', '')
            if prompt:
                system_prompts[uid] = prompt
    gt['system_prompts'] = system_prompts

    # ω2: Task instructions
    task_instructions = {}
    for attr in ('user_prompt_system_role', 'item_prompt_system_role',
                 'user_prompt_template', 'item_prompt_template'):
        val = getattr(model, attr, None)
        if val:
            task_instructions[attr] = val
    gt['task_instructions'] = task_instructions

    # ω4: Agent count
    gt['num_agents'] = len(model.user_agents) + len(model.item_agents)

    # Inference-time communication density: n_candidates
    n_candidates = 0
    if hasattr(model, 'attack_config') and isinstance(model.attack_config, dict):
        n_candidates = model.attack_config.get('num_candidates', 2)
    elif hasattr(model, 'num_candidates'):
        n_candidates = model.num_candidates
    gt['n_candidates'] = n_candidates

    # ω5: Topology
    topology_edges: Set[Tuple[int, int]] = set()
    if hasattr(model, 'train_data') and model.train_data is not None:
        try:
            uid_field = model.train_data.dataset.uid_field
            iid_field = model.train_data.dataset.iid_field
            uids = model.train_data.dataset.inter_feat[uid_field].numpy().tolist()
            iids = model.train_data.dataset.inter_feat[iid_field].numpy().tolist()
            for u, i in zip(uids, iids):
                topology_edges.add((int(u), int(i)))
        except Exception as e:
            logger.warning(f"Could not extract topology edges: {e}")

    gt['topology_edges'] = topology_edges
    gt['ui_pairs'] = topology_edges

    # num_ui_edges from config
    num_ui_edges = 0
    if hasattr(model, 'attack_config') and isinstance(model.attack_config, dict):
        num_ui_edges = model.attack_config.get('num_ui_edges', 0)
    gt['num_ui_edges'] = num_ui_edges

    return gt


def collect_response_from_backward(
    masleak_attacker,
    response: str,
    agent_id: int,
    agent_type: str,
    turn: int,
):
    """
    Hook to call after each backward-pass LLM response.

    Forwards the response text to the MASLeak core for IP parsing.
    """
    if masleak_attacker is None:
        return
    masleak_attacker.collect_response(
        response=response,
        source_agent_id=agent_id,
        agent_type=agent_type,
        turn=turn,
    )
