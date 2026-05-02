"""
Weights & Biases (wandb) Logger for MACF

Provides comprehensive experiment tracking for MACF (Multi-Agent Collaborative Filtering):
1. Full configuration logging (attack config, MACF config, model config)
2. Stdout/stderr capture
3. LLM judge history as artifacts
4. Conversation logs as artifacts
5. Per-turn metrics (6-class structure)
6. Temporal metrics with task boundaries
7. Contamination tracking (attackers vs victims)
8. Recommendation quality metrics (Hit@K, NDCG@K)

Mirrors the ConnaCF wandb_logger.py structure for consistency.
"""

import os
import sys
import json
import time
import glob
import threading
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
from dataclasses import dataclass

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("[WandB] wandb not installed. Install with: pip install wandb")


@dataclass
class MACFWandBConfig:
    """Configuration for MACF wandb logging."""
    project: str = "macf-attacks"
    entity: Optional[str] = None
    run_name: Optional[str] = None
    tags: Optional[List[str]] = None
    notes: Optional[str] = None
    log_artifacts: bool = True
    log_stdout: bool = True
    artifact_log_frequency: int = 5  # Log artifacts every N tasks
    sync_tensorboard: bool = False
    # New: per-turn JSON logging
    log_turn_jsons: bool = True
    log_figures: bool = True
    # Resume support
    resume_run_id: Optional[str] = None


class StdoutCapture:
    """Captures stdout/stderr to a file while still printing to console."""
    
    def __init__(self, log_path: str):
        self.log_path = log_path
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        self.log_file = None
        self._lock = threading.Lock()
    
    def start(self):
        """Start capturing stdout/stderr."""
        self.log_file = open(self.log_path, 'w', encoding='utf-8')
        sys.stdout = self
        sys.stderr = self
    
    def write(self, text):
        """Write to both console and log file."""
        with self._lock:
            self.original_stdout.write(text)
            self.original_stdout.flush()
            if self.log_file:
                self.log_file.write(text)
                self.log_file.flush()
    
    def flush(self):
        """Flush both streams."""
        self.original_stdout.flush()
        if self.log_file:
            self.log_file.flush()
    
    def stop(self):
        """Stop capturing and restore original streams."""
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr
        if self.log_file:
            self.log_file.close()
            self.log_file = None
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False


class MACFWandBLogger:
    """Comprehensive wandb logger for MACF experiments."""
    
    def __init__(
        self,
        config: Optional[MACFWandBConfig] = None,
        attack_config: Optional[Dict[str, Any]] = None,
        macf_config: Optional[Any] = None,
        dataset_name: str = "",
        output_dir: str = "",
        attack_name: str = "baseline"
    ):
        """Initialize MACF wandb logger.
        
        Args:
            config: MACFWandBConfig with wandb settings
            attack_config: Attack configuration dict (from YAML)
            macf_config: MACFConfig instance
            dataset_name: Name of the dataset
            output_dir: Base output directory for artifacts
            attack_name: Name of the attack (e.g., 'netsafe_misinfo', 'cheat_50percent')
        """
        self.config = config or MACFWandBConfig()
        self.attack_config = attack_config or {}
        self.macf_config = macf_config
        self.dataset_name = dataset_name
        self.output_dir = output_dir
        self.attack_name = attack_name
        
        self.run = None
        self.stdout_capture = None
        self.stdout_log_path = None
        self._initialized = False
        self._task_counter = 0
        self._global_turn_counter = 0
        self._logged_turns = set()  # Track which turns have been logged
        self._logged_figures = set()  # Track which figures have been logged
        
        # Paths for artifacts
        self.llm_judge_log_path = None
        self.conversation_log_path = None
        self.temporal_metrics_path = None

    def init(self, run_name: Optional[str] = None) -> bool:
        """Initialize wandb run.
        
        Args:
            run_name: Optional custom run name
            
        Returns:
            True if initialization successful
        """
        if not WANDB_AVAILABLE:
            print("[WandB] wandb not available, skipping initialization")
            return False
        
        try:
            # Build run name
            if run_name:
                final_run_name = run_name
            elif self.config.run_name:
                final_run_name = self.config.run_name
            else:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                final_run_name = f"macf_{self.attack_name}_{self.dataset_name}_{timestamp}"
            
            # Build tags
            tags = list(self.config.tags) if self.config.tags else []
            tags.append("macf")
            if self.dataset_name:
                tags.append(f"dataset:{self.dataset_name}")
            if self.attack_name and self.attack_name != "baseline":
                tags.append(f"attack:{self.attack_name}")
            
            # Build full config for logging
            full_config = self._build_full_config()
            
            # Check for resume
            resume_mode = None
            run_id = None
            if self.config.resume_run_id:
                resume_mode = "must"
                run_id = self.config.resume_run_id
                print(f"[WandB] Resuming run: {run_id}")
            
            # Initialize wandb run
            self.run = wandb.init(
                project=self.config.project,
                entity=self.config.entity,
                name=final_run_name,
                tags=tags,
                notes=self.config.notes,
                config=full_config,
                reinit=True,
                resume=resume_mode,
                id=run_id
            )
            
            # Setup stdout capture
            if self.config.log_stdout:
                self._setup_stdout_capture()
            
            # Setup artifact paths
            self._setup_artifact_paths()
            
            self._initialized = True
            print(f"[WandB] ✅ Initialized MACF run: {self.run.name}")
            print(f"[WandB] 📊 View at: {self.run.url}")
            print(f"[WandB] 🆔 Run ID: {self.run.id} (use for resume)")
            
            return True
            
        except Exception as e:
            print(f"[WandB] ❌ Failed to initialize: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _build_full_config(self) -> Dict[str, Any]:
        """Build comprehensive config dict for wandb."""
        config = {}
        
        # Framework identifier
        config["framework"] = "macf"
        
        # Attack config (flattened for wandb UI)
        if self.attack_config:
            attack_section = self.attack_config.get('attack', {})
            config["attack_method"] = attack_section.get('method', 'baseline')
            config["enable_attack"] = attack_section.get('enable_attack', False)
            config["attacker_ratio"] = attack_section.get('attacker_ratio', 0.0)
            config["attack"] = self._make_serializable(attack_section)
        
        # MACF config
        if self.macf_config:
            if hasattr(self.macf_config, '__dict__'):
                macf_dict = vars(self.macf_config)
            elif hasattr(self.macf_config, 'to_dict'):
                macf_dict = self.macf_config.to_dict()
            else:
                macf_dict = {}
            
            config["macf"] = self._make_serializable(macf_dict)
            # Flatten key MACF params for easy filtering
            config["neighbor_count"] = macf_dict.get('neighbor_count', 30)
            config["history_item_count"] = macf_dict.get('history_item_count', 10)
            config["max_rounds"] = macf_dict.get('max_rounds', 3)
            config["top_k_recommendation"] = macf_dict.get('top_k_recommendation', 10)
        
        # Experiment metadata
        config["experiment"] = {
            "dataset": self.dataset_name,
            "attack_name": self.attack_name,
            "timestamp": datetime.now().isoformat(),
            "output_dir": self.output_dir
        }
        
        return config
    
    def _make_serializable(self, obj: Any) -> Any:
        """Recursively make an object JSON-serializable for wandb config."""
        if obj is None:
            return None
        elif isinstance(obj, (str, int, float, bool)):
            return obj
        elif isinstance(obj, (list, tuple)):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, dict):
            result = {}
            for k, v in obj.items():
                try:
                    result[str(k)] = self._make_serializable(v)
                except Exception:
                    result[str(k)] = str(v)
            return result
        else:
            try:
                json.dumps(obj)
                return obj
            except (TypeError, ValueError):
                return str(obj)

    def _setup_stdout_capture(self):
        """Setup stdout/stderr capture to file."""
        self.stdout_log_path = os.path.join(
            self.output_dir or ".",
            f"stdout_macf.log"
        )
        os.makedirs(os.path.dirname(self.stdout_log_path) or ".", exist_ok=True)
        
        self.stdout_capture = StdoutCapture(self.stdout_log_path)
        self.stdout_capture.start()
        print(f"[WandB] 📝 Capturing stdout to: {self.stdout_log_path}")
    
    def _setup_artifact_paths(self):
        """Setup paths for artifact logging."""
        if not self.output_dir:
            return
        
        # LLM judge log path
        self.llm_judge_log_path = os.path.join(self.output_dir, "llm_judge_log.txt")
        
        # Conversation log path
        self.conversation_log_path = os.path.join(self.output_dir, "conversations")
        
        # Temporal metrics path
        self.temporal_metrics_path = os.path.join(self.output_dir, "temporal_metrics.json")
        
        print(f"[WandB] Artifact paths configured:")
        print(f"[WandB]   Output dir: {self.output_dir}")
        print(f"[WandB]   LLM judge log: {self.llm_judge_log_path}")
        print(f"[WandB]   Conversations: {self.conversation_log_path}")

    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """Log metrics to wandb.
        
        Args:
            metrics: Dict of metric name -> value
            step: Optional step number (global turn)
        """
        if not self._initialized or not self.run:
            return
        
        try:
            flat_metrics = self._flatten_dict(metrics)
            
            if step is not None:
                wandb.log(flat_metrics, step=step)
            else:
                wandb.log(flat_metrics)
                
        except Exception as e:
            print(f"[WandB] Warning: Failed to log metrics: {e}")
    
    def _flatten_dict(self, d: Dict[str, Any], parent_key: str = '', sep: str = '/') -> Dict[str, Any]:
        """Flatten nested dict with separator."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            elif isinstance(v, (int, float, bool)):
                items.append((new_key, v))
            elif isinstance(v, str) and len(v) < 100:
                items.append((new_key, v))
        return dict(items)

    def log_turn_metrics(
        self,
        global_turn: int,
        task_id: int,
        turn_in_task: int,
        utility_metrics: Optional[Dict[str, Any]] = None,
        dissemination_metrics: Optional[Dict[str, Any]] = None,
        ranking_metrics: Optional[Dict[str, Any]] = None,
        llm_judge_metrics: Optional[Dict[str, Any]] = None,
        resource_metrics: Optional[Dict[str, Any]] = None
    ):
        """Log per-turn metrics following the 6-class structure.
        
        Args:
            global_turn: Global turn counter across all tasks
            task_id: Current task ID
            turn_in_task: Turn index within current task
            utility_metrics: Draft list size, consensus score, etc.
            dissemination_metrics: Agent counts, contamination, etc.
            ranking_metrics: Hit@K, NDCG@K predictions
            llm_judge_metrics: LLM judge contamination results
            resource_metrics: LLM calls, tokens, duration
        """
        self._global_turn_counter = global_turn
        
        metrics = {
            "turn/global_turn": global_turn,
            "turn/task_id": task_id,
            "turn/turn_in_task": turn_in_task,
        }
        
        # Utility metrics (Row 1)
        if utility_metrics:
            metrics.update({
                "utility/draft_list_size": utility_metrics.get('draft_list_size', 0),
                "utility/consensus_score": utility_metrics.get('consensus_score', 0.0),
                "utility/new_items_added": utility_metrics.get('new_items_added', 0),
                "utility/items_removed": utility_metrics.get('items_removed', 0),
            })
        
        # Dissemination metrics (Row 2)
        if dissemination_metrics:
            metrics.update({
                "dissemination/num_agents": dissemination_metrics.get('num_agents', 0),
                "dissemination/num_user_agents": dissemination_metrics.get('num_user_agents', 0),
                "dissemination/num_item_agents": dissemination_metrics.get('num_item_agents', 0),
                "dissemination/num_adversarial": dissemination_metrics.get('num_adversarial_agents', 0),
            })
        
        # Ranking prediction metrics
        if ranking_metrics:
            metrics.update({
                "ranking/hit_at_1": ranking_metrics.get('hit_at_1', 0.0),
                "ranking/hit_at_3": ranking_metrics.get('hit_at_3', 0.0),
                "ranking/hit_at_5": ranking_metrics.get('hit_at_5', 0.0),
                "ranking/hit_at_10": ranking_metrics.get('hit_at_10', 0.0),
                "ranking/ndcg_at_1": ranking_metrics.get('ndcg_at_1', 0.0),
                "ranking/ndcg_at_3": ranking_metrics.get('ndcg_at_3', 0.0),
                "ranking/ndcg_at_5": ranking_metrics.get('ndcg_at_5', 0.0),
                "ranking/ndcg_at_10": ranking_metrics.get('ndcg_at_10', 0.0),
            })
        
        # LLM Judge contamination metrics
        if llm_judge_metrics:
            metrics.update({
                "contamination/score": llm_judge_metrics.get('contamination_score', 0.0),
                "contamination/rate": llm_judge_metrics.get('contamination_rate', 0.0),
                "contamination/cumulative": llm_judge_metrics.get('cumulative_contaminated', 0),
                "contamination/total_agents_seen": llm_judge_metrics.get('total_agents_seen', 0),
                "contamination/num_attackers": llm_judge_metrics.get('num_attackers', 0),
                "contamination/num_victims": llm_judge_metrics.get('num_victims', 0),
                "contamination/num_known_attackers": llm_judge_metrics.get('num_known_attackers', 0),
                "contamination/victim_rate": llm_judge_metrics.get('victim_rate', 0.0),
                "contamination/num_user_victims": llm_judge_metrics.get('num_user_victims', 0),
                "contamination/num_item_victims": llm_judge_metrics.get('num_item_victims', 0),
            })
        
        # Resource metrics (Row 4)
        if resource_metrics:
            metrics.update({
                "resources/llm_calls": resource_metrics.get('llm_calls', 0),
                "resources/tokens_used": resource_metrics.get('tokens_used', 0),
                "resources/turn_duration": resource_metrics.get('turn_duration_seconds', 0.0),
                "resources/memory_mb": resource_metrics.get('memory_mb', 0.0),
            })
        
        self.log_metrics(metrics, step=global_turn)

    def log_task_metrics(
        self,
        task_id: int,
        user_id: int,
        hit_at_10: float,
        ndcg_at_10: float,
        num_rounds: int,
        duration_seconds: float,
        global_turn_start: int,
        global_turn_end: int
    ):
        """Log task-level metrics.
        
        Args:
            task_id: Task ID
            user_id: Target user ID
            hit_at_10: Final Hit@10 for this task
            ndcg_at_10: Final NDCG@10 for this task
            num_rounds: Number of discussion rounds
            duration_seconds: Task duration
            global_turn_start: First global turn of this task
            global_turn_end: Last global turn of this task
        """
        self._task_counter = task_id
        
        metrics = {
            "task/task_id": task_id,
            "task/user_id": user_id,
            "task/hit_at_10": hit_at_10,
            "task/ndcg_at_10": ndcg_at_10,
            "task/num_rounds": num_rounds,
            "task/duration_seconds": duration_seconds,
            "task/global_turn_start": global_turn_start,
            "task/global_turn_end": global_turn_end,
        }
        
        self.log_metrics(metrics, step=global_turn_end)
        
        # Log turn JSONs and figures for this task
        if self.config.log_turn_jsons:
            self.log_turn_json(task_id, global_turn_end)
        if self.config.log_figures:
            self.log_task_figures(task_id)
        
        # Log artifacts periodically
        if self.config.log_artifacts and task_id % self.config.artifact_log_frequency == 0:
            self._log_artifacts_async(task_id)
    
    def log_turn_json(self, task_id: int, turn_idx: int, json_path: Optional[str] = None):
        """Log a per-turn JSON file as both artifact AND metrics.
        
        Args:
            task_id: Task ID
            turn_idx: Turn index
            json_path: Optional explicit path to JSON file
        """
        if not self._initialized or not self.run:
            return
        
        key = (task_id, turn_idx)
        if key in self._logged_turns:
            return
        
        # Find the JSON file
        if json_path is None and self.output_dir:
            task_dir = os.path.join(self.output_dir, f"task_{task_id}")
            json_path = os.path.join(task_dir, f"turn_{turn_idx}.json")
        
        if not json_path or not os.path.exists(json_path):
            return
        
        try:
            with open(json_path, 'r') as f:
                turn_data = json.load(f)
            
            # Extract key metrics and log them directly
            metrics = self._extract_turn_metrics(turn_data, turn_idx)
            self.log_metrics(metrics, step=turn_idx)
            
            # Also log the full JSON as an artifact
            if self.config.log_artifacts:
                artifact = wandb.Artifact(
                    name=f"task_{task_id}_turn_{turn_idx}_json",
                    type="turn_metrics",
                    description=f"Full metrics JSON for task {task_id} turn {turn_idx}"
                )
                artifact.add_file(json_path)
                self.run.log_artifact(artifact)
            
            self._logged_turns.add(key)
            print(f"[WandB] 📊 Logged task {task_id} turn {turn_idx} metrics + artifact")
            
        except Exception as e:
            print(f"[WandB] Warning: Failed to log turn JSON: {e}")
    
    def _extract_turn_metrics(self, turn_data: Dict[str, Any], turn_idx: int) -> Dict[str, Any]:
        """Extract key metrics from turn JSON for direct logging."""
        metrics = {"turn": turn_idx}
        
        # System performance
        if "system_performance" in turn_data:
            sp = turn_data["system_performance"]
            metrics["accuracy"] = sp.get("accuracy", 0.0)
            metrics["recall_at_10"] = sp.get("recall_at_10", 0.0)
            metrics["ndcg_at_10"] = sp.get("ndcg_at_10", 0.0)
        
        # LLM Judge contamination
        if "llm_judge" in turn_data:
            llm = turn_data["llm_judge"]
            if "user" in llm:
                metrics["contamination/user_count"] = llm["user"].get("contaminated_count", 0)
                metrics["contamination/user_mean"] = llm["user"].get("mean_contamination_score", 0.0)
            if "item" in llm:
                metrics["contamination/item_count"] = llm["item"].get("contaminated_count", 0)
                metrics["contamination/item_mean"] = llm["item"].get("mean_contamination_score", 0.0)
        
        return metrics
    
    def log_figure(self, figure_path: str, turn_idx: Optional[int] = None, caption: str = ""):
        """Log a figure/image to wandb."""
        if not self._initialized or not self.run:
            return
        
        if figure_path in self._logged_figures:
            return
        
        if not os.path.exists(figure_path):
            return
        
        try:
            fig_name = os.path.basename(figure_path).replace('.png', '').replace('.jpg', '')
            img = wandb.Image(figure_path, caption=caption or fig_name)
            
            log_data = {f"figures/{fig_name}": img}
            if turn_idx is not None:
                wandb.log(log_data, step=turn_idx)
            else:
                wandb.log(log_data)
            
            self._logged_figures.add(figure_path)
            print(f"[WandB] 🖼️ Logged figure: {fig_name}")
            
        except Exception as e:
            print(f"[WandB] Warning: Failed to log figure {figure_path}: {e}")
    
    def log_task_figures(self, task_id: int):
        """Log all figures for a specific task."""
        if not self.output_dir:
            return
        
        task_dir = os.path.join(self.output_dir, f"task_{task_id}")
        if not os.path.exists(task_dir):
            return
        
        # Find all PNG files in task directory
        for fig_path in glob.glob(os.path.join(task_dir, "*.png")):
            self.log_figure(fig_path, caption=f"Task {task_id}")
    
    def log_all_turn_jsons(self):
        """Scan output directory and log all turn JSONs that haven't been logged yet."""
        if not self.output_dir or not os.path.exists(self.output_dir):
            return
        
        # Find all task directories
        for task_dir_name in sorted(os.listdir(self.output_dir)):
            if not task_dir_name.startswith('task_'):
                continue
            
            try:
                task_id = int(task_dir_name.replace('task_', ''))
            except ValueError:
                continue
            
            task_dir = os.path.join(self.output_dir, task_dir_name)
            
            # Find all turn_*.json files
            for json_path in sorted(glob.glob(os.path.join(task_dir, "turn_*.json"))):
                basename = os.path.basename(json_path)
                try:
                    turn_idx = int(basename.replace("turn_", "").replace(".json", ""))
                    self.log_turn_json(task_id, turn_idx, json_path)
                except ValueError:
                    continue
    
    def log_metrics_from_collector(self, metrics_collector: 'MACFMetricsCollector'):
        """Log metrics directly from MACFMetricsCollector.
        
        Args:
            metrics_collector: MACFMetricsCollector instance
        """
        if not self._initialized or not self.run:
            return
        
        try:
            # Get current turn metrics
            if metrics_collector.current_turn_metrics:
                turn = metrics_collector.current_turn_metrics
                
                self.log_turn_metrics(
                    global_turn=turn.global_turn,
                    task_id=turn.task_id,
                    turn_in_task=turn.turn,
                    utility_metrics=turn.utility,
                    dissemination_metrics=turn.dissemination,
                    ranking_metrics=turn.ranking_prediction,
                    llm_judge_metrics=turn.llm_judge,
                    resource_metrics=turn.resources
                )
            
            # Log cumulative metrics
            summary = metrics_collector.get_summary()
            if summary:
                self.log_metrics({
                    "cumulative/total_tasks": summary.get('total_tasks', 0),
                    "cumulative/total_global_turns": summary.get('total_global_turns', 0),
                    "cumulative/total_llm_calls": summary.get('total_llm_calls', 0),
                    "cumulative/total_tokens": summary.get('total_tokens', 0),
                })
                
        except Exception as e:
            print(f"[WandB] Warning: Failed to log metrics from collector: {e}")

    def log_evaluation_metrics(self, metrics: Dict[str, float], prefix: str = "eval"):
        """Log evaluation metrics (recall, ndcg, etc.).
        
        Args:
            metrics: Dict of metric name -> value
            prefix: Prefix for metric names
        """
        prefixed_metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}
        self.log_metrics(prefixed_metrics)
    
    def log_attack_summary(self, summary: Dict[str, Any]):
        """Log attack summary statistics.
        
        Args:
            summary: Attack summary dict from metrics collector
        """
        if not summary:
            return
        
        metrics = {}
        for k, v in summary.items():
            if isinstance(v, (int, float)):
                metrics[f"attack_summary/{k}"] = v
            elif isinstance(v, dict):
                for sub_k, sub_v in v.items():
                    if isinstance(sub_v, (int, float)):
                        metrics[f"attack_summary/{k}/{sub_k}"] = sub_v
        
        self.log_metrics(metrics)
    
    def log_task_boundary(self, task_id: int, global_turn: int):
        """Log a task boundary marker for visualization.
        
        Args:
            task_id: Task ID that is starting
            global_turn: Global turn where this task starts
        """
        self.log_metrics({
            "task_boundary/task_id": task_id,
            "task_boundary/global_turn": global_turn,
        }, step=global_turn)

    def _log_artifacts_async(self, task_id: int):
        """Log artifacts in background thread."""
        def _log():
            self._log_llm_judge_artifact(task_id)
            self._log_conversation_artifact(task_id)
            self._log_temporal_metrics_artifact(task_id)
        
        thread = threading.Thread(target=_log, daemon=True)
        thread.start()
    
    def _log_llm_judge_artifact(self, task_id: int):
        """Log LLM judge history as artifact."""
        if not self.llm_judge_log_path or not os.path.exists(self.llm_judge_log_path):
            return
        
        try:
            artifact = wandb.Artifact(
                name=f"llm_judge_log_task_{task_id}",
                type="llm_judge_history",
                description=f"LLM judge evaluation history up to task {task_id}"
            )
            artifact.add_file(self.llm_judge_log_path)
            self.run.log_artifact(artifact)
            print(f"[WandB] 📦 Logged LLM judge artifact for task {task_id}")
        except Exception as e:
            print(f"[WandB] Warning: Failed to log LLM judge artifact: {e}")
    
    def _log_conversation_artifact(self, task_id: int):
        """Log conversation logs as artifact."""
        if not self.conversation_log_path or not os.path.exists(self.conversation_log_path):
            return
        
        try:
            artifact = wandb.Artifact(
                name=f"conversations_task_{task_id}",
                type="conversations",
                description=f"Agent conversation logs up to task {task_id}"
            )
            artifact.add_dir(self.conversation_log_path)
            self.run.log_artifact(artifact)
            print(f"[WandB] 📦 Logged conversation artifact for task {task_id}")
        except Exception as e:
            print(f"[WandB] Warning: Failed to log conversation artifact: {e}")
    
    def _log_temporal_metrics_artifact(self, task_id: int):
        """Log temporal metrics JSON as artifact."""
        if not self.temporal_metrics_path or not os.path.exists(self.temporal_metrics_path):
            return
        
        try:
            artifact = wandb.Artifact(
                name=f"temporal_metrics_task_{task_id}",
                type="metrics",
                description=f"Temporal metrics up to task {task_id}"
            )
            artifact.add_file(self.temporal_metrics_path)
            self.run.log_artifact(artifact)
            print(f"[WandB] 📦 Logged temporal metrics artifact for task {task_id}")
        except Exception as e:
            print(f"[WandB] Warning: Failed to log temporal metrics artifact: {e}")

    def log_final_artifacts(self):
        """Log final artifacts at end of experiment."""
        if not self._initialized or not self.run:
            return
        
        print("[WandB] 📦 Logging final MACF artifacts...")
        
        # Log all remaining turn JSONs
        self.log_all_turn_jsons()
        
        # Log final LLM judge history
        if self.llm_judge_log_path and os.path.exists(self.llm_judge_log_path):
            try:
                artifact = wandb.Artifact(
                    name="llm_judge_log_final",
                    type="llm_judge_history",
                    description="Complete LLM judge evaluation history"
                )
                artifact.add_file(self.llm_judge_log_path)
                self.run.log_artifact(artifact)
                print(f"[WandB] ✅ Logged final LLM judge artifact")
            except Exception as e:
                print(f"[WandB] Warning: Failed to log final LLM judge artifact: {e}")
        
        # Log final conversation logs
        if self.conversation_log_path and os.path.exists(self.conversation_log_path):
            try:
                artifact = wandb.Artifact(
                    name="conversations_final",
                    type="conversations",
                    description="Complete agent conversation logs"
                )
                artifact.add_dir(self.conversation_log_path)
                self.run.log_artifact(artifact)
                print(f"[WandB] ✅ Logged final conversation artifact")
            except Exception as e:
                print(f"[WandB] Warning: Failed to log final conversation artifact: {e}")
        
        # Log final temporal metrics
        if self.temporal_metrics_path and os.path.exists(self.temporal_metrics_path):
            try:
                artifact = wandb.Artifact(
                    name="temporal_metrics_final",
                    type="metrics",
                    description="Complete temporal metrics"
                )
                artifact.add_file(self.temporal_metrics_path)
                self.run.log_artifact(artifact)
                print(f"[WandB] ✅ Logged final temporal metrics artifact")
            except Exception as e:
                print(f"[WandB] Warning: Failed to log final temporal metrics artifact: {e}")
        
        # Log stdout capture
        if self.stdout_log_path and os.path.exists(self.stdout_log_path):
            try:
                artifact = wandb.Artifact(
                    name="stdout_log",
                    type="stdout",
                    description="Complete stdout/stderr capture"
                )
                artifact.add_file(self.stdout_log_path)
                self.run.log_artifact(artifact)
                print(f"[WandB] ✅ Logged stdout artifact")
            except Exception as e:
                print(f"[WandB] Warning: Failed to log stdout artifact: {e}")
        
        # Log experiment summary JSON
        summary_file = os.path.join(self.output_dir, "experiment_summary.json")
        if os.path.exists(summary_file):
            try:
                artifact = wandb.Artifact(
                    name="experiment_summary",
                    type="summary",
                    description="Experiment summary with aggregate metrics"
                )
                artifact.add_file(summary_file)
                self.run.log_artifact(artifact)
                print(f"[WandB] ✅ Logged experiment summary artifact")
            except Exception as e:
                print(f"[WandB] Warning: Failed to log experiment summary artifact: {e}")
        
        # Log all task metrics JSON files
        if self.output_dir and os.path.exists(self.output_dir):
            try:
                task_dirs = [d for d in os.listdir(self.output_dir) if d.startswith('task_')]
                if task_dirs:
                    artifact = wandb.Artifact(
                        name="task_metrics_all",
                        type="task_metrics",
                        description="All task metrics JSON files"
                    )
                    for task_dir in task_dirs:
                        task_path = os.path.join(self.output_dir, task_dir)
                        if os.path.isdir(task_path):
                            artifact.add_dir(task_path, name=task_dir)
                    self.run.log_artifact(artifact)
                    print(f"[WandB] ✅ Logged {len(task_dirs)} task metrics directories")
            except Exception as e:
                print(f"[WandB] Warning: Failed to log task metrics: {e}")

    def log_table(self, name: str, data: List[Dict[str, Any]], columns: Optional[List[str]] = None):
        """Log a table to wandb.
        
        Args:
            name: Table name
            data: List of row dicts
            columns: Optional column names (inferred from data if not provided)
        """
        if not self._initialized or not self.run:
            return
        
        try:
            if not columns and data:
                columns = list(data[0].keys())
            
            table = wandb.Table(columns=columns)
            for row in data:
                table.add_data(*[row.get(c, "") for c in columns])
            
            wandb.log({name: table})
        except Exception as e:
            print(f"[WandB] Warning: Failed to log table {name}: {e}")
    
    def log_contamination_table(self, contamination_data: List[Dict[str, Any]]):
        """Log contamination tracking table.
        
        Args:
            contamination_data: List of contamination records
        """
        columns = [
            "global_turn", "agent_type", "agent_id", "is_contaminated",
            "contamination_score", "detected_canaries", "reasoning"
        ]
        self.log_table("contamination_tracking", contamination_data, columns)
    
    def save_wandb_resume_info(self):
        """Save wandb run info for future resumption."""
        if not self._initialized or not self.run or not self.output_dir:
            return
        
        resume_info = {
            "wandb_run_id": self.run.id,
            "wandb_run_name": self.run.name,
            "wandb_project": self.config.project,
            "wandb_entity": self.config.entity,
            "wandb_url": self.run.url,
            "timestamp": datetime.now().isoformat()
        }
        
        resume_info_path = os.path.join(self.output_dir, "wandb_resume_info.json")
        try:
            with open(resume_info_path, 'w') as f:
                json.dump(resume_info, f, indent=2)
            print(f"[WandB] 💾 Saved resume info to {resume_info_path}")
        except Exception as e:
            print(f"[WandB] Warning: Failed to save resume info: {e}")
    
    def finish(self):
        """Finish wandb run and cleanup."""
        if not self._initialized:
            return
        
        try:
            # Stop stdout capture
            if self.stdout_capture:
                self.stdout_capture.stop()
            
            # Save resume info
            self.save_wandb_resume_info()
            
            # Log final artifacts
            self.log_final_artifacts()
            
            # Finish run
            if self.run:
                self.run.finish()
                print(f"[WandB] ✅ MACF run finished: {self.run.url}")
                print(f"[WandB] 🆔 Run ID: {self.run.id} (use for resume)")
            
            self._initialized = False
            
        except Exception as e:
            print(f"[WandB] Warning: Error during finish: {e}")
    
    def __enter__(self):
        self.init()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.finish()
        return False


def create_macf_wandb_logger(
    attack_config_path: Optional[str] = None,
    macf_config: Optional[Any] = None,
    dataset_name: str = "",
    output_dir: str = "",
    attack_name: str = "baseline",
    project: str = "macf-attacks",
    entity: Optional[str] = None,
    run_name: Optional[str] = None,
    tags: Optional[List[str]] = None,
    log_artifacts: bool = True,
    log_stdout: bool = True,
    log_turn_jsons: bool = True,
    log_figures: bool = True,
    resume_run_id: Optional[str] = None
) -> Optional[MACFWandBLogger]:
    """Factory function to create MACFWandBLogger.
    
    Args:
        attack_config_path: Path to attack config YAML
        macf_config: MACFConfig instance
        dataset_name: Dataset name
        output_dir: Output directory
        attack_name: Attack name (e.g., 'netsafe_misinfo')
        project: wandb project name
        entity: wandb entity/team
        run_name: Custom run name
        tags: List of tags
        log_artifacts: Whether to log artifacts
        log_stdout: Whether to capture stdout
        log_turn_jsons: Whether to log per-turn JSONs
        log_figures: Whether to log training figures
        resume_run_id: Optional wandb run ID to resume
        
    Returns:
        MACFWandBLogger instance or None if wandb not available
    """
    if not WANDB_AVAILABLE:
        print("[WandB] wandb not available, returning None")
        return None
    
    # Load attack config if path provided
    attack_config = {}
    if attack_config_path and os.path.exists(attack_config_path):
        import yaml
        with open(attack_config_path, 'r') as f:
            attack_config = yaml.safe_load(f)
    
    # Get output dir from attack config if not provided
    if not output_dir and attack_config:
        output_dir = attack_config.get('attack', {}).get('output', {}).get('output_directory', '')
    
    # Create config
    wandb_config = MACFWandBConfig(
        project=project,
        entity=entity,
        run_name=run_name,
        tags=tags,
        log_artifacts=log_artifacts,
        log_stdout=log_stdout,
        log_turn_jsons=log_turn_jsons,
        log_figures=log_figures,
        resume_run_id=resume_run_id
    )
    
    # Create logger
    logger = MACFWandBLogger(
        config=wandb_config,
        attack_config=attack_config,
        macf_config=macf_config,
        dataset_name=dataset_name,
        output_dir=output_dir,
        attack_name=attack_name
    )
    
    return logger


def load_macf_wandb_resume_info(output_dir: str) -> Optional[Dict[str, Any]]:
    """Load wandb resume info from a previous MACF run.
    
    Args:
        output_dir: Path to output directory
        
    Returns:
        Dict with wandb_run_id, wandb_project, etc. or None
    """
    resume_info_path = os.path.join(output_dir, "wandb_resume_info.json")
    if os.path.exists(resume_info_path):
        try:
            with open(resume_info_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"[WandB] Warning: Failed to load resume info: {e}")
    return None
