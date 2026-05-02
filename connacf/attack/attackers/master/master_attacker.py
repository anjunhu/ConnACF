"""
MASTER Attacker — Multi-Agent Security Through Exploration of Roles and Topological Structures

Implements the MASTER attack framework from arXiv:2505.18572v1:
  Three-stage attack: Probing → Trait Injection → Activation

The attacker:
  1. Probes agents to extract role and topology information
  2. Injects domain-specific behavioral patterns (dark traits)
  3. Activates adversarial behavior with role/topology embedding
  4. Collects responses and computes dissemination metrics (ASR, Role, Coor)
"""

from typing import Dict, List, Any, Optional, Set, Tuple
import json
import time
import random
import logging
from pathlib import Path

from .probing_stage import ProbingStage
from .trait_injection_stage import TraitInjectionStage
from .activation_stage import ActivationStage
from .metrics import MASTERMetrics
from .defense_mechanisms import (
    PromptLeakageDetector,
    HierarchicalMonitor,
    PreemptiveDefense,
)

logger = logging.getLogger(__name__)


class MASTERAttacker:
    """
    MASTER Attack: Multi-Agent Security Through Exploration of Roles and Topological Structures
    
    Three-stage attack:
    1. Probing: Extract role and topology information
    2. Trait Injection: Inject domain-specific behavioral patterns
    3. Activation: Trigger adversarial behavior with role/topology embedding
    
    Lifecycle:
      1. __init__: Configure attack parameters
      2. initialize_attack(): Execute all three stages, return attacker IDs
      3. (pipeline runs — patterns propagate through interactions)
      4. collect_response(): Called after each interaction to track behavior
      5. evaluate(): Compute dissemination metrics (ASR, Role, Coor)
      6. save_results(): Persist everything
    """
    
    def __init__(
        self,
        surrogate_model: Optional[Any] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize MASTER attacker.
        
        Args:
            surrogate_model: Not used for MASTER (interactive attack)
            config: Attack configuration dictionary containing:
                - user_attacker_ratio: Fraction of users to make attackers (default: 0.1)
                - item_attacker_ratio: Fraction of items to make attackers (default: 0.1)
                - probing_rounds: Number of probing rounds (default: 2)
                - domain_override: Force specific domain (default: None)
                - traits_config: Custom trait configuration
                - activation_trigger: Custom activation trigger
                - defenses: Defense mechanism configuration
        """
        self.config = config or {}
        
        # Attack configuration
        self.user_attacker_ratio = self.config.get('user_attacker_ratio', 0.1)
        self.item_attacker_ratio = self.config.get('item_attacker_ratio', 0.1)
        self.probing_rounds = self.config.get('probing_rounds', 2)
        self.domain = self.config.get('domain_override', 'movie_recommender')
        
        # Stage components
        self.probing_stage = ProbingStage(config)
        self.trait_injection_stage = TraitInjectionStage(config)
        self.activation_stage = ActivationStage(config)
        
        # Metrics
        self.metrics_calculator = MASTERMetrics()
        
        # Defense mechanisms (optional)
        defense_config = self.config.get('defenses', {})
        self.defenses_enabled = defense_config.get('enable_defenses', False)
        if self.defenses_enabled:
            self.prompt_leakage_detector = PromptLeakageDetector(
                defense_config.get('prompt_leakage_detection', {})
            )
            self.hierarchical_monitor = HierarchicalMonitor(
                defense_config.get('hierarchical_monitoring', {})
            )
            self.preemptive_defense = PreemptiveDefense(
                defense_config.get('preemptive_defense', {})
            )
        else:
            self.prompt_leakage_detector = None
            self.hierarchical_monitor = None
            self.preemptive_defense = None
        
        # Extracted information
        self.role_info: Dict[str, Dict] = {}
        self.topology_info: Dict = {}
        self.probing_rounds_completed: int = 0
        
        # Attacker tracking
        self.attacker_user_ids: Set[int] = set()
        self.attacker_item_ids: Set[int] = set()
        self.target_ids: Set[int] = set()
        
        # Response collection
        self.all_responses: List[Dict] = []
        self.agent_behaviors: Dict[str, List[str]] = {}
        
        # Metrics
        self.dissemination_metrics: Dict[str, Any] = {}
        
        # Timing
        self.start_time = None
        
        logger.info(f"[MASTER] Initialized MASTER attacker: "
                   f"user_ratio={self.user_attacker_ratio}, "
                   f"item_ratio={self.item_attacker_ratio}, "
                   f"domain={self.domain}")
    
    def initialize_attack(
        self,
        connacf_instance
    ) -> Tuple[Set[int], Set[int], Set[int]]:
        """
        Execute all three MASTER stages at initialization.
        
        Args:
            connacf_instance: The ConnaCF model instance
            
        Returns:
            Tuple of (attacker_user_ids, attacker_item_ids, target_ids)
        """
        self.start_time = time.time()
        
        # Select attacker agents
        n_users = len(connacf_instance.user_agents)
        n_items = len(connacf_instance.item_agents)
        
        n_user_attackers = max(1, int(n_users * self.user_attacker_ratio))
        n_item_attackers = max(1, int(n_items * self.item_attacker_ratio)) if n_items > 0 else 0
        
        # Clamp to available agents
        n_user_attackers = min(n_user_attackers, n_users)
        n_item_attackers = min(n_item_attackers, n_items)
        
        # Sample attacker IDs
        all_user_ids = list(connacf_instance.user_agents.keys())
        all_item_ids = list(connacf_instance.item_agents.keys())
        
        self.attacker_user_ids = set(random.sample(all_user_ids, n_user_attackers))
        self.attacker_item_ids = set(random.sample(all_item_ids, n_item_attackers))
        
        print(f"[MASTER] Selected {len(self.attacker_user_ids)} user attackers, "
              f"{len(self.attacker_item_ids)} item attackers")
        
        # Stage 1: Probing
        print(f"[MASTER] Stage 1: Probing ({self.probing_rounds} rounds)...")
        self.role_info, self.topology_info = self.probing_stage.execute(
            connacf_instance, self.attacker_user_ids, self.attacker_item_ids
        )
        # Record how many rounds actually completed (may be < probing_rounds on error)
        self.probing_rounds_completed = getattr(
            self.probing_stage, "probing_rounds_completed", self.probing_rounds
        )
        
        # Stage 2: Trait Injection
        print(f"[MASTER] Stage 2: Trait Injection...")
        self.trait_injection_stage.execute(
            connacf_instance, self.attacker_user_ids, self.attacker_item_ids, self.domain
        )
        
        # Stage 3: Activation
        print(f"[MASTER] Stage 3: Activation...")
        self.activation_stage.execute(
            connacf_instance, self.attacker_user_ids, self.attacker_item_ids,
            self.role_info, self.topology_info
        )
        
        # Preemptive defense: modify ALL agent system prompts before training
        if self.preemptive_defense and self.preemptive_defense.enabled:
            print(f"[MASTER] Applying preemptive defense to all agents...")
            self.preemptive_defense.apply_to_agents(connacf_instance)
        
        # All non-attacker users are potential targets
        self.target_ids = set(all_user_ids) - self.attacker_user_ids
        
        print(f"[MASTER] Three stages complete. Targets: {len(self.target_ids)}")
        
        return self.attacker_user_ids, self.attacker_item_ids, self.target_ids
    
    def collect_response(
        self,
        response: str,
        source_agent_id: Optional[int] = None,
        agent_type: str = 'unknown',
        turn: int = 0,
        user_id: Optional[int] = None
    ):
        """
        Collect a response from any agent and track behavior.
        
        Should be called after each interaction in the pipeline.
        
        Args:
            response: The agent's response text
            source_agent_id: ID of the agent that produced this response
            agent_type: 'user' or 'item'
            turn: Current turn number
            user_id: If this is a user agent response, the user's ID
        """
        # Check for prompt leakage — schedules warning for agent's next input
        if self.prompt_leakage_detector and self.prompt_leakage_detector.enabled:
            self.prompt_leakage_detector.check(response, f'{agent_type}_{source_agent_id}')
        
        # Monitor high-criticality agents — schedules warning for agent's next input
        if self.hierarchical_monitor and self.hierarchical_monitor.enabled:
            self.hierarchical_monitor.monitor(f'{agent_type}_{source_agent_id}', response, turn)
        
        # Record response
        record = {
            'source_agent_id': source_agent_id,
            'agent_type': agent_type,
            'turn': turn,
            'user_id': user_id,
            'response_text': response,
            'response_preview': response[:500],
            'timestamp': time.time(),
        }
        self.all_responses.append(record)
        
        # Track agent behaviors for role consistency metrics
        agent_key = f'{agent_type}_{source_agent_id}'
        if agent_key not in self.agent_behaviors:
            self.agent_behaviors[agent_key] = []
        self.agent_behaviors[agent_key].append(response)  # full text for role scoring

        # Feed response into probing stage for temporal extraction
        if hasattr(self, 'probing_stage') and source_agent_id is not None:
            self.probing_stage.observe_response(
                response, source_agent_id, agent_type
            )
            # Update attacker-level references so metrics see latest state
            self.role_info = self.probing_stage.role_info
            self.topology_info = self.probing_stage.topology_info
    
    def evaluate(self, results: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Compute MASTER dissemination metrics.
        
        Args:
            results: Optional additional results to include
        
        Returns:
            Dictionary containing dissemination metrics and metadata
        """
        # Get injected traits for role consistency evaluation
        injected_traits = self.trait_injection_stage.get_injected_traits()
        
        # Compute dissemination metrics
        self.dissemination_metrics = self.metrics_calculator.compute_dissemination_metrics(
            self.all_responses,
            injected_traits,
            self.agent_behaviors
        )
        
        # Add metadata
        elapsed_time = time.time() - self.start_time if self.start_time else 0
        self.dissemination_metrics['metadata'] = {
            'num_attacker_users': len(self.attacker_user_ids),
            'num_attacker_items': len(self.attacker_item_ids),
            'num_targets': len(self.target_ids),
            'num_interactions': len(self.all_responses),
            'domain': self.domain,
            'elapsed_time': elapsed_time,
            'defenses_enabled': self.defenses_enabled,
        }
        
        # Add defense statistics if enabled
        if self.defenses_enabled:
            self.dissemination_metrics['defense_stats'] = {
                'prompt_leakage_detections': len(
                    self.prompt_leakage_detector.get_detections()
                ) if self.prompt_leakage_detector else 0,
                'hierarchical_alerts': len(
                    self.hierarchical_monitor.get_alerts()
                ) if self.hierarchical_monitor else 0,
                'blocked_content': len(
                    self.preemptive_defense.get_blocked_content()
                ) if self.preemptive_defense else 0,
            }
        
        return self.dissemination_metrics
    
    def get_report(self) -> str:
        """Generate human-readable MASTER attack report."""
        if not self.dissemination_metrics:
            self.evaluate()
        
        metrics = self.dissemination_metrics
        dissemination = metrics.get('dissemination', {})
        metadata = metrics.get('metadata', {})
        
        lines = [
            "=" * 72,
            "MASTER Attack Report",
            "=" * 72,
            f"Domain: {metadata.get('domain', 'unknown')}",
            f"Attacker Users: {metadata.get('num_attacker_users', 0)}",
            f"Attacker Items: {metadata.get('num_attacker_items', 0)}",
            f"Target Users: {metadata.get('num_targets', 0)}",
            f"Total Interactions: {metadata.get('num_interactions', 0)}",
            f"Elapsed Time: {metadata.get('elapsed_time', 0):.1f}s",
            "",
            "Dissemination Metrics:",
            "-" * 72,
            f"  ASR (Attack Success Rate):     {dissemination.get('asr', 0):.3f}",
            f"  Role Consistency:              {dissemination.get('role_consistency', 0):.1f}",
            f"  Coor (Team Cooperation):       {dissemination.get('coor', 0):.1f}",
        ]
        
        if self.defenses_enabled:
            defense_stats = metrics.get('defense_stats', {})
            lines.extend([
                "",
                "Defense Statistics:",
                "-" * 72,
                f"  Prompt Leakage Detections:     {defense_stats.get('prompt_leakage_detections', 0)}",
                f"  Hierarchical Alerts:           {defense_stats.get('hierarchical_alerts', 0)}",
                f"  Blocked Content:               {defense_stats.get('blocked_content', 0)}",
            ])
        
        lines.extend(["", "=" * 72])
        return "\n".join(lines)
    
    def save_results(self, output_path):
        """
        Save all results to disk.
        
        Args:
            output_path: Directory path for output files
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Ensure metrics are computed
        if not self.dissemination_metrics:
            self.evaluate()
        
        # Save metrics
        with open(output_path / 'master_metrics.json', 'w') as f:
            json.dump(self.dissemination_metrics, f, indent=2, default=str)
        
        # Save extracted info
        extracted_info = {
            'role_info': self.role_info,
            'topology_info': self.topology_info,
            'attacker_user_ids': list(self.attacker_user_ids),
            'attacker_item_ids': list(self.attacker_item_ids),
            'target_ids': list(self.target_ids),
        }
        with open(output_path / 'master_extracted_info.json', 'w') as f:
            json.dump(extracted_info, f, indent=2, default=str)
        
        # Save responses
        with open(output_path / 'master_responses.json', 'w') as f:
            json.dump(self.all_responses, f, indent=2, default=str)
        
        # Save human-readable report
        with open(output_path / 'master_report.txt', 'w') as f:
            f.write(self.get_report())
        
        # Generate extraction dashboard visualization
        try:
            from ..visualization.extraction_dashboard import ExtractionDashboard
            
            # Build metrics in format expected by dashboard
            dashboard_metrics = {
                'extract_rate': self.dissemination_metrics.get('probing_success_rate', 0),
                'extract_rate_v2': self.dissemination_metrics.get('probing_success_rate', 0),
                'num_responses_collected': len(self.all_responses),
                'elapsed_time': time.time() - self.start_time if self.start_time else 0,
                # MASTER probing metrics
                'ss_system_prompt': self.dissemination_metrics.get('role_extraction_rate', 0),
                'sm_system_prompt': 0,
                'ss_task_instructions': 0,
                'sm_task_instructions': 0,
                'f1_agent_count': self.dissemination_metrics.get('agent_count_accuracy', 0),
                'f1_comm_density': 0,
                # Topology extraction
                'ui_topology_recall': self.dissemination_metrics.get('topology_extraction_rate', 0),
                'ui_topology_precision': 0,
                'ui_items_leaked': len(self.topology_info),
                'ui_items_total': len(self.target_ids),
                # Per-target scores
                'ui_topology_scores': [
                    1.0 if tid in self.role_info else 0.0
                    for tid in self.target_ids
                ],
                'user_ids': list(self.target_ids),
            }
            
            dashboard = ExtractionDashboard(str(output_path), attack_type='MASTER')
            dashboard.create_dashboard(dashboard_metrics, experiment_name='master')
            
            logger.info("[MASTER] Extraction dashboard generated")
        except Exception as e:
            logger.warning(f"[MASTER] Could not generate extraction dashboard: {e}")
        
        logger.info(f"[MASTER] Results saved to {output_path}")
        print(f"[MASTER] Results saved to {output_path}")
    
    def pop_defense_warning(self, agent_type: str, agent_id: int) -> Optional[str]:
        """Return and clear any pending defense warning for this agent.

        Called by the integration layer before building the next LLM input,
        so the warning is prepended to the agent's context (Eq. 13 in paper).
        """
        key = f'{agent_type}_{agent_id}'
        warning = None
        if self.prompt_leakage_detector:
            warning = self.prompt_leakage_detector.pop_warning(key) or warning
        if self.hierarchical_monitor:
            warning = self.hierarchical_monitor.pop_warning(key) or warning
        return warning

    def reset(self):
        """Reset all state for a new attack run."""
        self.role_info.clear()
        self.topology_info.clear()
        self.attacker_user_ids.clear()
        self.attacker_item_ids.clear()
        self.target_ids.clear()
        self.all_responses.clear()
        self.agent_behaviors.clear()
        self.dissemination_metrics.clear()
        self.start_time = None
        
        # Reset stage components
        self.probing_stage = ProbingStage(self.config)
        self.trait_injection_stage = TraitInjectionStage(self.config)
        self.activation_stage = ActivationStage(self.config)
    
    def __repr__(self) -> str:
        """String representation of MASTER attacker."""
        return (f"MASTERAttacker("
                f"user_attackers={len(self.attacker_user_ids)}, "
                f"item_attackers={len(self.attacker_item_ids)}, "
                f"targets={len(self.target_ids)}, "
                f"domain='{self.domain}')")
