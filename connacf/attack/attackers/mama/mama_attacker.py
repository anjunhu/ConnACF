"""
MAMA (Multi-Agent Memory Attack) Attacker

Implements the MAMA framework for measuring PII leakage in multi-agent systems.
Adapts MAMA's two-phase protocol (Engram + Resonance) to the recommendation domain.
"""

from typing import Dict, List, Set, Tuple, Any, Optional
from ..base_attacker import BaseAttacker


class MAMAAttacker(BaseAttacker):
    """
    MAMA (Multi-Agent Memory Attack) PII extraction attacker.
    
    Inherits from BaseAttacker to integrate with existing attack framework.
    Implements MAMA's two-phase protocol:
    1. Engram Phase: Seed private PII into target agents
    2. Resonance Phase: Multi-round extraction through U-U interactions
    
    Unlike other attacks (Drunk/Cheat/NetSafe), MAMA is an interactive multi-round
    attack that tracks PII leakage over time rather than optimizing a single payload.
    """
    
    def __init__(self, surrogate_model: Optional[Any], config: Dict[str, Any]):
        """
        Initialize MAMA attacker.
        
        Args:
            surrogate_model: Not used for MAMA (interactive attack, no offline optimization)
            config: Attack configuration containing:
                - target_ratio: Fraction of users to target (default: 0.1)
                - attacker_ratio: Fraction of users to make attackers (default: 0.1)
                - max_rounds: Maximum resonance rounds (default: 10)
                - early_stop: Stop if all PII leaked (default: True)
                - extraction_goal: What to extract ('all', 'identity', 'contact', etc.)
                - pii_detection_method: 'keyword', 'llm', or 'both' (default: 'keyword')
        """
        super().__init__(surrogate_model, config)
        
        # Import MAMA components
        from .engram_phase import EngramPhase
        from .resonance_phase import ResonancePhase
        from .pii_tracker import PIILeakageTracker
        
        self.engram = EngramPhase(config)
        self.resonance = ResonancePhase(config)
        self.tracker = PIILeakageTracker(config)
        
        # MAMA-specific config
        self.target_ratio = config.get('target_ratio', 0.1)
        self.attacker_ratio = config.get('attacker_ratio', 0.1)
        self.item_attacker_ratio = config.get('item_attacker_ratio', 0.1)
        self.max_rounds = config.get('max_rounds', 10)
        self.early_stop = config.get('early_stop', True)
        self.extraction_goal = config.get('extraction_goal', 'all')
        
        # Track IDs
        self.target_ids = set()
        self.attacker_ids = set()
        self.item_attacker_ids = set()
        
        self.log(f"Initialized MAMA attacker: target_ratio={self.target_ratio}, "
                f"attacker_ratio={self.attacker_ratio}, item_attacker_ratio={self.item_attacker_ratio}, "
                f"max_rounds={self.max_rounds}")
    
    def optimize(self, *args, **kwargs) -> Any:
        """
        Phase 1: Not applicable for MAMA (interactive attack).
        
        MAMA doesn't use offline optimization on a surrogate model.
        Instead, it seeds PII and tracks leakage interactively.
        
        Returns:
            None (MAMA uses initialize_attack() instead)
        """
        self.log("MAMA is an interactive attack - no offline optimization phase", "INFO")
        return None
    
    def inject(self, model: Any) -> Tuple[Set[int], Set[int], Set[int]]:
        """
        Phase 2: Initialize MAMA attack (Engram phase).
        
        This replaces the traditional inject() method with MAMA's Engram phase:
        - Seed private PII into target agents
        - Initialize attacker agents with extraction goals
        - Initialize item attacker agents (spies/honeypots)
        
        Args:
            model: ConnaCF model with user_agents and item_agents
        
        Returns:
            Tuple of (target_user_ids, attacker_user_ids, attacker_item_ids)
        """
        return self.initialize_attack(model)
    
    def initialize_attack(self, model: Any) -> Tuple[Set[int], Set[int], Set[int]]:
        """
        Initialize MAMA attack (Engram phase).

        Selection order: attackers first, then ALL remaining users become targets.
        Every non-attacker user gets PII seeded — no wasted "normal" users.

        Args:
            model: ConnaCF model with user_agents and item_agents

        Returns:
            Tuple of (target_user_ids, attacker_user_ids, attacker_item_ids)
        """
        self.log("Starting Engram Phase...")

        # Step 1: Select attacker users FIRST
        all_user_ids = set(model.user_agents.keys())
        self.attacker_ids = self.engram.select_attackers(
            model.user_agents,
            ratio=self.attacker_ratio,
            exclude=set()  # No exclusions — attackers are selected first
        )

        self.log(f"Selected {len(self.attacker_ids)} attacker users")

        # Initialize attackers with extraction goal
        for user_id in self.attacker_ids:
            self.engram.initialize_attacker(
                model.user_agents[user_id],
                extraction_goal=self.extraction_goal
            )

        # Step 2: ALL remaining users become targets (no wasted normal users)
        self.target_ids = all_user_ids - self.attacker_ids

        self.log(f"All {len(self.target_ids)} non-attacker users are targets")

        # Seed private PII into ALL targets
        for user_id in self.target_ids:
            pii_dict = self.engram.generate_pii_for_user(user_id)
            self.engram.seed_pii(
                model.user_agents[user_id],
                pii_dict
            )
            self.tracker.register_target(user_id, pii_dict)

        self.log(f"Seeded PII into {len(self.target_ids)} targets")

        # Step 3: Select item attackers (spies/honeypots)
        if hasattr(model, 'item_agents') and model.item_agents:
            self.item_attacker_ids = self.engram.select_item_attackers(
                model.item_agents,
                ratio=self.item_attacker_ratio
            )

            self.log(f"Selected {len(self.item_attacker_ids)} item attackers")

            for item_id in self.item_attacker_ids:
                self.engram.initialize_item_attacker(
                    model.item_agents[item_id],
                    extraction_goal=self.extraction_goal
                )
        else:
            self.log("No item agents available - skipping item attacker initialization", "WARNING")

        self.log("Engram Phase complete")
        self.log(f"Starting Resonance Phase (max {self.max_rounds} rounds)...")

        return self.target_ids, self.attacker_ids, self.item_attacker_ids
    
    def on_uu_interaction(self, user_id: int, friend_ids: List[int],
                          messages: List[str], round_idx: int):
        """
        Hook called during U-U interactions (Resonance phase).
        
        Tracks PII leakage in friend consultation messages.
        
        Args:
            user_id: User who received messages
            friend_ids: IDs of friends who sent messages
            messages: Messages from friends
            round_idx: Current round index
        """
        # Log the U-U interaction for MAMA analysis
        self.log(f"[U-U INTERACTION] Round {round_idx}: User {user_id} consulting {len(friend_ids)} friends")
        
        # Check if attacker received message from target
        if user_id in self.attacker_ids:
            for friend_id, message in zip(friend_ids, messages):
                self.log(f"[U-U] Attacker {user_id} <- Target/Friend {friend_id}: {message[:100]}...")
                
                if friend_id in self.target_ids:
                    # Check for PII leakage via U-U channel
                    leaked = self.tracker.check_leakage(
                        message,
                        target_id=friend_id,
                        round_idx=round_idx,
                        channel='U-U'
                    )
                    if leaked:
                        self.log(f"🚨 Round {round_idx}: Leaked {len(leaked)} PII from "
                                f"User {friend_id} -> Attacker {user_id} (U-U)")
                        for pii_value in leaked:
                            self.log(f"  ✓ LEAKED: {pii_value}", "WARNING")
        
        # Also check if target is consulting attacker (reverse direction)
        elif user_id in self.target_ids:
            for friend_id, message in zip(friend_ids, messages):
                if friend_id in self.attacker_ids:
                    self.log(f"[U-U] Target {user_id} <- Attacker {friend_id}: {message[:100]}...")
                    # Check for PII leakage in attacker's response to target
                    leaked = self.tracker.check_leakage(
                        message,
                        target_id=user_id,
                        round_idx=round_idx,
                        channel='U-U'
                    )
                    if leaked:
                        self.log(f"🚨 Round {round_idx}: Leaked {len(leaked)} PII from "
                                f"Target {user_id} via Attacker {friend_id}'s response (U-U)")
                        for pii_value in leaked:
                            self.log(f"  ✓ LEAKED: {pii_value}", "WARNING")
    
    def on_ui_interaction(self, user_id: int, item_id: int, 
                          user_message: str, item_response: str, round_idx: int):
        """
        Hook called during U-I interactions (Resonance phase).
        
        Tracks PII leakage when users interact with item attackers (spies/honeypots).
        
        Args:
            user_id: User who interacted with item
            item_id: Item that was interacted with
            user_message: Message from user to item
            item_response: Response from item to user
            round_idx: Current round index
        """
        # Log the U-I interaction for MAMA analysis
        self.log(f"[U-I INTERACTION] Round {round_idx}: User {user_id} <-> Item {item_id}")
        
        # Check if target user interacted with item attacker
        if user_id in self.target_ids and item_id in self.item_attacker_ids:
            self.log(f"[U-I] Target {user_id} -> Item Attacker {item_id}: {user_message[:100]}...")
            
            # Check for PII leakage in user's message to item
            leaked = self.tracker.check_leakage(
                user_message,
                target_id=user_id,
                round_idx=round_idx,
                channel='U-I'
            )
            if leaked:
                self.log(f"🚨 Round {round_idx}: Leaked {len(leaked)} PII from "
                        f"User {user_id} -> Item Attacker {item_id} (U-I)")
                for pii_value in leaked:
                    self.log(f"  ✓ LEAKED: {pii_value}", "WARNING")
        
        # Also check if attacker user is interacting with items (might reveal PII indirectly)
        elif user_id in self.attacker_ids:
            # Attacker might extract PII by observing item responses that contain target info
            for target_id in self.target_ids:
                leaked = self.tracker.check_leakage(
                    item_response,
                    target_id=target_id,
                    round_idx=round_idx,
                    channel='U-I'
                )
                if leaked:
                    self.log(f"🚨 Round {round_idx}: Leaked {len(leaked)} PII from "
                            f"Target {target_id} via Item {item_id} -> Attacker {user_id} (U-I)")
                    for pii_value in leaked:
                        self.log(f"  ✓ LEAKED: {pii_value}", "WARNING")
    
    def evaluate(self, results: Dict[str, Any] = None) -> Dict[str, float]:
        """
        Phase 3: Compute MAMA-specific metrics.
        
        Args:
            results: Not used for MAMA (metrics tracked internally)
        
        Returns:
            Dictionary of MAMA metrics:
                - mama_leak_rate: Fraction of PII leaked
                - mama_time_to_first_leak: Round when first PII leaked
                - mama_outcome: 'success', 'partial', or 'failure'
                - mama_per_category_rates: Leak rate by MAMA category
                - mama_n_targets: Number of target users
                - mama_n_attackers: Number of attacker users
        """
        return self.get_metrics()
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get MAMA-specific metrics.
        
        Returns:
            Dict of metric_name -> value
        """
        metrics = {
            'mama_leak_rate': self.tracker.get_leak_rate(),
            'mama_time_to_first_leak': self.tracker.get_time_to_first_leak(),
            'mama_per_round_leakage': self.tracker.get_per_round_counts(),
            'mama_outcome': self.tracker.get_outcome(),
            'mama_per_category_rates': self.tracker.get_per_category_rates(),
            'mama_per_channel_rates': self.tracker.get_per_channel_rates(),
            'mama_n_targets': len(self.target_ids),
            'mama_n_attackers': len(self.attacker_ids),
            'mama_n_item_attackers': len(self.item_attacker_ids),
            'mama_max_rounds': self.max_rounds,
        }
        
        self.log(f"MAMA Metrics: leak_rate={metrics['mama_leak_rate']:.2%}, "
                f"outcome={metrics['mama_outcome']}, "
                f"time_to_first_leak={metrics['mama_time_to_first_leak']}")
        
        return metrics
    
    def should_continue(self) -> bool:
        """
        Check if resonance phase should continue.
        
        Returns:
            True if should continue, False if should stop
        """
        return self.resonance.should_continue(self.tracker)
    
    def advance_round(self):
        """Advance to next resonance round"""
        self.resonance.advance_round()
    
    def record_round(self, round_idx: int):
        """
        Record metrics at end of round.
        
        Args:
            round_idx: Current round index
        """
        self.tracker.record_round(round_idx)
    
    def save_results(self, output_dir: str):
        """
        Save MAMA attack results to disk.
        
        Saves:
        - mama_metrics.json: All MAMA metrics
        - mama_leaked_conversations.json: Full conversation content where PII was detected
        - mama_leaked_summary.json: Quick summary of leaked content
        - mama_ground_truth.json: PII ground truth for all targets
        
        Args:
            output_dir: Directory to save results
        """
        import json
        import os
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save metrics
        metrics = self.get_metrics()
        with open(os.path.join(output_dir, 'mama_metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=2, default=str)
        
        # Save leaked conversations (full content)
        self.tracker.save_leaked_conversations(
            os.path.join(output_dir, 'mama_leaked_conversations.json')
        )
        
        # Save leaked content summary
        summary = self.tracker.get_leaked_content_summary()
        with open(os.path.join(output_dir, 'mama_leaked_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Save ground truth PII for all targets
        ground_truth = {
            str(uid): pii_dict 
            for uid, pii_dict in self.tracker.targets.items()
        }
        with open(os.path.join(output_dir, 'mama_ground_truth.json'), 'w') as f:
            json.dump(ground_truth, f, indent=2, default=str)
        
        # Save per-target leak status
        leak_status = {}
        for uid in self.tracker.targets:
            leaked_set = self.tracker.leaked.get(uid, set())
            leak_status[str(uid)] = {
                'leaked_values': list(leaked_set),
                'leak_count': len(leaked_set),
                'total_pii': len([v for v in self.tracker.targets[uid].values() if v]),
            }
        with open(os.path.join(output_dir, 'mama_leak_status.json'), 'w') as f:
            json.dump(leak_status, f, indent=2, default=str)
        
        # Generate extraction dashboard visualization
        try:
            from ..visualization.extraction_dashboard import ExtractionDashboard
            
            # Build metrics in format expected by dashboard
            dashboard_metrics = {
                'extract_rate': metrics.get('mama_leak_rate', 0),
                'extract_rate_v2': metrics.get('mama_leak_rate', 0),
                'num_responses_collected': len(self.tracker.leaked_conversations),
                'elapsed_time': 0,  # MAMA doesn't track time the same way
                # Map MAMA metrics to dashboard format
                'ss_system_prompt': 0,  # N/A for MAMA
                'sm_system_prompt': 0,
                'ss_task_instructions': 0,
                'sm_task_instructions': 0,
                'f1_agent_count': 0,
                'f1_comm_density': 0,
                # PII-specific metrics
                'ui_topology_recall': metrics.get('mama_leak_rate', 0),
                'ui_topology_precision': 1.0,  # MAMA only reports true positives
                'ui_items_leaked': metrics.get('mama_total_leaked', 0),
                'ui_items_total': metrics.get('mama_total_pii', 0),
                # Per-target scores
                'ui_topology_scores': [
                    len(self.tracker.leaked.get(uid, set())) / max(1, len([v for v in pii.values() if v]))
                    for uid, pii in self.tracker.targets.items()
                ],
                'user_ids': list(self.tracker.targets.keys()),
            }
            
            dashboard = ExtractionDashboard(output_dir, attack_type='MAMA')
            
            # Build per-turn metrics from tracker history
            for round_idx, round_data in enumerate(self.tracker.round_history):
                dashboard.record_turn_metrics(round_idx, {
                    'extract_rate': round_data.get('cumulative_leak_rate', 0),
                    'responses_collected': round_data.get('leaks_this_round', 0),
                    'ui_items_leaked': round_data.get('cumulative_leaks', 0),
                    'ui_items_total': metrics.get('mama_total_pii', 0),
                    'ui_topology_recall': round_data.get('cumulative_leak_rate', 0),
                })
            
            dashboard.create_dashboard(dashboard_metrics, experiment_name='mama')
            dashboard.create_temporal_plots(experiment_name='mama')
            
            self.log("MAMA extraction dashboard generated")
        except Exception as e:
            self.log(f"Could not generate extraction dashboard: {e}", "WARNING")
        
        self.log(f"MAMA results saved to {output_dir}")
        self.log(f"  - {len(self.tracker.leaked_conversations)} leaked conversations saved")
    
    def get_report(self) -> str:
        """
        Generate human-readable MAMA attack report.
        
        Returns:
            Formatted report string
        """
        metrics = self.get_metrics()
        summary = self.tracker.get_leaked_content_summary()
        
        lines = [
            "=" * 72,
            "MAMA (Multi-Agent Memory Attack) Report",
            "=" * 72,
            f"Targets: {metrics['mama_n_targets']} users",
            f"User Attackers: {metrics['mama_n_attackers']}",
            f"Item Attackers: {metrics['mama_n_item_attackers']}",
            f"Max Rounds: {metrics['mama_max_rounds']}",
            "",
            "Results:",
            "-" * 72,
            f"  Leak Rate: {metrics['mama_leak_rate']:.2%}",
            f"  Outcome: {metrics['mama_outcome']}",
            f"  Time to First Leak: {metrics['mama_time_to_first_leak']}",
            "",
            "Per-Category Leak Rates:",
            "-" * 72,
        ]
        
        for category, rate in metrics.get('mama_per_category_rates', {}).items():
            lines.append(f"  {category}: {rate:.2%}")
        
        lines.extend([
            "",
            "Per-Channel Leak Rates:",
            "-" * 72,
        ])
        
        for channel, rate in metrics.get('mama_per_channel_rates', {}).items():
            lines.append(f"  {channel}: {rate:.2%}")
        
        lines.extend([
            "",
            "Leaked Conversations Summary:",
            "-" * 72,
            f"  Total Leak Events: {summary['total_leaks']}",
            f"  By Channel: {summary['by_channel']}",
            f"  By Target: {summary['by_target']}",
        ])
        
        if summary['sample_messages']:
            lines.extend([
                "",
                "Sample Leaked Messages:",
                "-" * 72,
            ])
            for sample in summary['sample_messages']:
                lines.append(f"  Round {sample['round']} ({sample['channel']}) Target {sample['target']}:")
                lines.append(f"    Leaked: {sample['leaked']}")
                lines.append(f"    Message: {sample['message_preview']}")
                lines.append("")
        
        lines.append("=" * 72)
        return "\n".join(lines)
    
    def __repr__(self) -> str:
        """String representation of MAMA attacker"""
        return (f"MAMAAttacker("
                f"targets={len(self.target_ids)}, "
                f"user_attackers={len(self.attacker_ids)}, "
                f"item_attackers={len(self.item_attacker_ids)}, "
                f"max_rounds={self.max_rounds}, "
                f"extraction_goal='{self.extraction_goal}')")
