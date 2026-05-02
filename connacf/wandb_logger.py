"""
Weights & Biases (wandb) Logger for ConnaCF

Provides comprehensive experiment tracking including:
1. Full configuration logging
2. Stdout/stderr capture
3. LLM judge history as artifacts
4. Interaction logs as artifacts
5. Per-turn JSON metrics (logged as both artifacts AND metrics)
6. Training progress figures (logged as images)
7. Resume state tracking
8. Conversation logs for judge_calibration_connacf.py compatibility
"""

import os
import sys
import json
import time
import shutil
import tempfile
import glob
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
from dataclasses import dataclass, asdict
import threading
import io

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("[WandB] wandb not installed. Install with: pip install wandb")


@dataclass
class WandBConfig:
    """Configuration for wandb logging."""
    project: str = "connacf-attacks"
    entity: Optional[str] = None
    run_name: Optional[str] = None
    tags: Optional[List[str]] = None
    notes: Optional[str] = None
    log_artifacts: bool = True
    log_stdout: bool = True
    artifact_log_frequency: int = 5  # Log artifacts every N rounds
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


class WandBLogger:
    """Comprehensive wandb logger for ConnaCF experiments.
    
    Key features:
    - Logs per-turn JSONs as both artifacts AND metrics for immediate visibility
    - Logs training progress figures as wandb images
    - Maintains stdout, conversation, and LLM judge logs for judge_calibration_connacf.py
    - Supports resume from previous runs
    """
    
    def __init__(
        self,
        config: Optional[WandBConfig] = None,
        attack_config: Optional[Dict[str, Any]] = None,
        model_config: Optional[Dict[str, Any]] = None,
        dataset_name: str = "",
        output_dir: str = "",
        task_id: int = 0
    ):
        """Initialize wandb logger.
        
        Args:
            config: WandBConfig with wandb settings
            attack_config: Attack configuration dict
            model_config: Model/RecBole configuration dict
            dataset_name: Name of the dataset
            output_dir: Base output directory for artifacts
            task_id: Current task ID
        """
        self.config = config or WandBConfig()
        self.attack_config = attack_config or {}
        self.model_config = model_config or {}
        self.dataset_name = dataset_name
        self.output_dir = output_dir
        self.task_id = task_id
        
        self.run = None
        self.stdout_capture = None
        self.stdout_log_path = None
        self._initialized = False
        self._round_counter = 0
        self._metrics_buffer = []
        self._logged_turns = set()  # Track which turns have been logged
        self._logged_figures = set()  # Track which figures have been logged
        
        # Paths for artifacts
        self.llm_judge_log_path = None
        self.interaction_log_path = None
        self._task_dir = None

    def set_task_dir(self, task_dir: str):
        """Set the task directory for artifact logging.
        
        Call this after the attack framework creates the timestamped task directory.
        This overrides the auto-constructed task_dir path.
        
        Args:
            task_dir: Full path to the task directory (e.g., output/dataset/YYMMDDHHMMSS)
        """
        self._task_dir = task_dir
        
        # Update artifact paths
        self.llm_judge_log_path = os.path.join(task_dir, "llm_judge_log.txt")
        self.interaction_log_path = os.path.join(
            task_dir, "conversations", "interaction_log.txt"
        )
        
        print(f"[WandB] Task directory updated to: {task_dir}")

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
                attack_method = self.attack_config.get('attack', {}).get('method', 'unknown')
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                final_run_name = f"{attack_method}_{self.dataset_name}_{timestamp}"
            
            # Build tags
            tags = self.config.tags or []
            if self.dataset_name:
                tags.append(f"dataset:{self.dataset_name}")
            attack_method = self.attack_config.get('attack', {}).get('method')
            if attack_method:
                tags.append(f"attack:{attack_method}")
            
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
            print(f"[WandB] ✅ Initialized run: {self.run.name}")
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
        
        # Add full attack config at top level for easy access in wandb UI
        if self.attack_config:
            attack_section = self.attack_config.get('attack', {})
            config["attack_method"] = attack_section.get('method', 'unknown')
            config["enable_attack"] = attack_section.get('enable_attack', False)
            config["attacker_ratio"] = attack_section.get('attacker_ratio', 0.0)
            config["current_task_id"] = attack_section.get('current_task_id', 0)
            config["attack"] = self._make_serializable(attack_section)
        
        # Experiment metadata
        config["experiment"] = {
            "dataset": self.dataset_name,
            "task_id": self.task_id,
            "timestamp": datetime.now().isoformat(),
            "output_dir": self.output_dir
        }
        
        # Add model config
        if self.model_config:
            if hasattr(self.model_config, 'final_config_dict'):
                model_dict = dict(self.model_config.final_config_dict)
            elif isinstance(self.model_config, dict):
                model_dict = self.model_config
            else:
                model_dict = {}
            config["model"] = self._make_serializable(model_dict)
        
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
            self.output_dir or tempfile.gettempdir(),
            f"stdout_task_{self.task_id}.log"
        )
        os.makedirs(os.path.dirname(self.stdout_log_path), exist_ok=True)
        
        self.stdout_capture = StdoutCapture(self.stdout_log_path)
        self.stdout_capture.start()
        print(f"[WandB] 📝 Capturing stdout to: {self.stdout_log_path}")
    
    def _setup_artifact_paths(self):
        """Setup paths for artifact logging."""
        if not self.output_dir:
            return
        
        # Only construct task_dir if not already set via set_task_dir()
        if not self._task_dir:
            # Build task directory path (legacy: task_N convention)
            task_dir = os.path.join(self.output_dir, self.dataset_name, f"task_{self.task_id}")
            self._task_dir = task_dir
        else:
            task_dir = self._task_dir
        
        # LLM judge log path
        self.llm_judge_log_path = os.path.join(task_dir, "llm_judge_log.txt")
        
        # Interaction log path
        self.interaction_log_path = os.path.join(
            task_dir, "conversations", "interaction_log.txt"
        )
        
        print(f"[WandB] Artifact paths configured:")
        print(f"[WandB]   Task dir: {task_dir}")
        print(f"[WandB]   LLM judge log: {self.llm_judge_log_path}")
        print(f"[WandB]   Interaction log: {self.interaction_log_path}")
    
    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """Log metrics to wandb."""
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
            else:
                items.append((new_key, v))
        return dict(items)

    def log_turn_json(self, turn_idx: int, json_path: Optional[str] = None):
        """Log a per-turn JSON file as both artifact AND metrics.
        
        This is the key method for making turn data immediately visible in wandb
        while also preserving the full JSON for later analysis.
        
        Args:
            turn_idx: Turn index
            json_path: Optional explicit path to JSON file
        """
        if not self._initialized or not self.run:
            return
        
        if turn_idx in self._logged_turns:
            return  # Already logged
        
        # Find the JSON file
        if json_path is None and self._task_dir:
            json_path = os.path.join(self._task_dir, f"turn_{turn_idx}.json")
        
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
                    name=f"turn_{turn_idx}_json",
                    type="turn_metrics",
                    description=f"Full metrics JSON for turn {turn_idx}"
                )
                artifact.add_file(json_path)
                self.run.log_artifact(artifact)
            
            self._logged_turns.add(turn_idx)
            print(f"[WandB] 📊 Logged turn {turn_idx} metrics + artifact")
            
        except Exception as e:
            print(f"[WandB] Warning: Failed to log turn {turn_idx} JSON: {e}")
    
    def _extract_turn_metrics(self, turn_data: Dict[str, Any], turn_idx: int) -> Dict[str, Any]:
        """Extract key metrics from turn JSON for direct logging."""
        metrics = {"turn": turn_idx}
        
        # System performance
        if "system_performance" in turn_data:
            sp = turn_data["system_performance"]
            metrics["accuracy"] = sp.get("accuracy", 0.0)
            metrics["recall_at_1"] = sp.get("recall_at_1", 0.0)
            metrics["recall_at_5"] = sp.get("recall_at_5", 0.0)
            metrics["recall_at_10"] = sp.get("recall_at_10", 0.0)
            metrics["ndcg_at_1"] = sp.get("ndcg_at_1", 0.0)
            metrics["ndcg_at_5"] = sp.get("ndcg_at_5", 0.0)
            metrics["ndcg_at_10"] = sp.get("ndcg_at_10", 0.0)
        
        # Communication graph
        if "communication_graph" in turn_data:
            cg = turn_data["communication_graph"]
            metrics["graph/total_edges"] = cg.get("total_edges", 0)
            metrics["graph/attacker_edges"] = cg.get("attacker_edges", 0)
        
        # Natural semantic evolution
        if "natural_semantic_evolution" in turn_data:
            nse = turn_data["natural_semantic_evolution"]
            metrics["evolution/user_count"] = nse.get("user_evolution_count", 0)
            metrics["evolution/item_count"] = nse.get("item_evolution_count", 0)
            metrics["evolution/user_pct"] = nse.get("user_evolution_percentage", 0.0)
            metrics["evolution/item_pct"] = nse.get("item_evolution_percentage", 0.0)
            metrics["attackers/users"] = nse.get("num_attacker_users", 0)
            metrics["attackers/items"] = nse.get("num_attacker_items", 0)
        
        # LLM Judge dissemination
        if "dissemination" in turn_data and "llm_judge" in turn_data["dissemination"]:
            llm = turn_data["dissemination"]["llm_judge"]
            if "user" in llm:
                metrics["contamination/user_count"] = llm["user"].get("contaminated_count", 0)
                metrics["contamination/user_mean"] = llm["user"].get("mean_contamination_score", 0.0)
                metrics["contamination/user_max"] = llm["user"].get("max_contamination_score", 0.0)
            if "item" in llm:
                metrics["contamination/item_count"] = llm["item"].get("contaminated_count", 0)
                metrics["contamination/item_mean"] = llm["item"].get("mean_contamination_score", 0.0)
                metrics["contamination/item_max"] = llm["item"].get("max_contamination_score", 0.0)
        
        # Privacy metrics (MAMA PII leakage)
        if "privacy" in turn_data:
            priv = turn_data["privacy"]
            metrics["privacy/pii_leakage_rate"] = priv.get("pii_leakage_rate", 0.0)
            metrics["privacy/pii_exposure_count"] = priv.get("pii_exposure_count", 0)
            metrics["privacy/outcome"] = priv.get("outcome", "n/a")
            metrics["privacy/n_targets"] = priv.get("n_targets", 0)
            metrics["privacy/n_user_attackers"] = priv.get("n_user_attackers", 0)
            metrics["privacy/n_item_attackers"] = priv.get("n_item_attackers", 0)
            # Per-category rates
            if "category_rates" in priv:
                for cat, rate in priv["category_rates"].items():
                    metrics[f"privacy/category_{cat}"] = rate
            # Per-channel rates
            if "channel_rates" in priv:
                for channel, rate in priv["channel_rates"].items():
                    metrics[f"privacy/channel_{channel}"] = rate
        
        # Reverse engineering metrics (MASLeak IP extraction)
        if "reverse_engineering" in turn_data:
            re = turn_data["reverse_engineering"]
            metrics["reverse_eng/extract_rate"] = re.get("extract_rate", 0.0)
            metrics["reverse_eng/ss_system_prompt"] = re.get("ss_system_prompt", 0.0)
            metrics["reverse_eng/sm_system_prompt"] = re.get("sm_system_prompt", 0.0)
            metrics["reverse_eng/ss_task_instructions"] = re.get("ss_task_instructions", 0.0)
            metrics["reverse_eng/sm_task_instructions"] = re.get("sm_task_instructions", 0.0)
            metrics["reverse_eng/f1_agent_count"] = re.get("f1_agent_count", 0.0)
            metrics["reverse_eng/f1_comm_density"] = re.get("f1_comm_density", 0.0)
            metrics["reverse_eng/gs_topology"] = re.get("gs_topology", 0.0)
            metrics["reverse_eng/num_responses_collected"] = re.get("num_responses_collected", 0)
            # U-I topology metrics
            metrics["reverse_eng/ui_topology_recall"] = re.get("ui_topology_recall", 0.0)
            metrics["reverse_eng/ui_topology_precision"] = re.get("ui_topology_precision", 0.0)
            metrics["reverse_eng/ui_topology_f1"] = re.get("ui_topology_f1", 0.0)
            metrics["reverse_eng/ui_items_leaked"] = re.get("ui_items_leaked", 0)
            metrics["reverse_eng/ui_items_total"] = re.get("ui_items_total", 0)
        
        return metrics

    def log_figure(self, figure_path: str, turn_idx: Optional[int] = None, caption: str = ""):
        """Log a figure/image to wandb.
        
        Args:
            figure_path: Path to the image file
            turn_idx: Optional turn index for step
            caption: Optional caption for the image
        """
        if not self._initialized or not self.run:
            return
        
        if figure_path in self._logged_figures:
            return
        
        if not os.path.exists(figure_path):
            return
        
        try:
            # Get figure name from path
            fig_name = os.path.basename(figure_path).replace('.png', '').replace('.jpg', '')
            
            # Log as wandb Image
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
    
    def log_all_turn_jsons(self):
        """Scan task directory and log all turn JSONs that haven't been logged yet."""
        if not self._task_dir or not os.path.exists(self._task_dir):
            return
        
        # Find all turn_*.json files
        pattern = os.path.join(self._task_dir, "turn_*.json")
        for json_path in sorted(glob.glob(pattern)):
            # Extract turn number
            basename = os.path.basename(json_path)
            try:
                turn_idx = int(basename.replace("turn_", "").replace(".json", ""))
                self.log_turn_json(turn_idx, json_path)
            except ValueError:
                continue
    
    def log_all_figures(self):
        """Scan task directory and log all training progress figures."""
        if not self._task_dir or not os.path.exists(self._task_dir):
            return
        
        # Find all training_progress_*.png files
        pattern = os.path.join(self._task_dir, "training_progress_*.png")
        for fig_path in sorted(glob.glob(pattern)):
            basename = os.path.basename(fig_path)
            try:
                # Extract turn number from filename
                turn_str = basename.replace("training_progress_turn_", "").replace(".png", "")
                turn_idx = int(turn_str)
                self.log_figure(fig_path, turn_idx, f"Training progress at turn {turn_idx}")
            except ValueError:
                self.log_figure(fig_path, caption=basename)

    def log_round_metrics(
        self,
        round_idx: int,
        accuracy: float,
        contaminated_users: int = 0,
        contaminated_items: int = 0,
        total_users: int = 0,
        total_items: int = 0,
        llm_judge_results: Optional[Dict[str, Any]] = None,
        additional_metrics: Optional[Dict[str, Any]] = None
    ):
        """Log metrics for a training round."""
        self._round_counter = round_idx
        
        metrics = {
            "round": round_idx,
            "accuracy": accuracy,
            "contamination/users": contaminated_users,
            "contamination/items": contaminated_items,
            "contamination/user_rate": contaminated_users / max(total_users, 1),
            "contamination/item_rate": contaminated_items / max(total_items, 1),
        }
        
        if llm_judge_results:
            metrics.update({
                f"llm_judge/{k}": v 
                for k, v in llm_judge_results.items()
                if isinstance(v, (int, float))
            })
        
        if additional_metrics:
            metrics.update(additional_metrics)
        
        self.log_metrics(metrics, step=round_idx)
        
        # Also log turn JSON and figures for this round
        if self.config.log_turn_jsons:
            self.log_turn_json(round_idx)
        if self.config.log_figures:
            fig_path = os.path.join(self._task_dir or "", f"training_progress_turn_{round_idx}.png")
            if os.path.exists(fig_path):
                self.log_figure(fig_path, round_idx)
        
        # Log artifacts periodically
        if self.config.log_artifacts and round_idx % self.config.artifact_log_frequency == 0:
            self._log_artifacts_async(round_idx)
    
    def log_metrics_from_collector(self, metrics_collector, round_idx: int):
        """Log metrics directly from AgentMetricsCollector."""
        if not self._initialized or not self.run:
            return
        
        try:
            # Get system metrics for current round
            if hasattr(metrics_collector, 'system_metrics') and metrics_collector.system_metrics:
                if round_idx < len(metrics_collector.system_metrics):
                    sys_metrics = metrics_collector.system_metrics[round_idx]
                else:
                    sys_metrics = metrics_collector.system_metrics[-1] if metrics_collector.system_metrics else {}
                
                metrics = {
                    "round": round_idx,
                    "accuracy": sys_metrics.get('accuracy', 0.0),
                    "contamination/users": sys_metrics.get('contaminated_users', 0),
                    "contamination/items": sys_metrics.get('contaminated_items', 0),
                    "contamination/user_rate": sys_metrics.get('user_contamination_rate', 0.0),
                    "contamination/item_rate": sys_metrics.get('item_contamination_rate', 0.0),
                }
                
                if 'valid_rate' in sys_metrics:
                    metrics['valid_rate'] = sys_metrics['valid_rate']
                if 'llm_judge_contaminated_users' in sys_metrics:
                    metrics['llm_judge/contaminated_users'] = sys_metrics['llm_judge_contaminated_users']
                if 'llm_judge_contaminated_items' in sys_metrics:
                    metrics['llm_judge/contaminated_items'] = sys_metrics['llm_judge_contaminated_items']
                
                self.log_metrics(metrics, step=round_idx)
            
            # Get cumulative contamination counts
            if hasattr(metrics_collector, '_cumulative_contaminated_users'):
                self.log_metrics({
                    "contamination/cumulative_users": len(metrics_collector._cumulative_contaminated_users),
                    "contamination/cumulative_items": len(metrics_collector._cumulative_contaminated_items),
                }, step=round_idx)
            
            # Log turn JSON for this round
            if self.config.log_turn_jsons:
                self.log_turn_json(round_idx)
            
            # Log figure for this round
            if self.config.log_figures and self._task_dir:
                fig_path = os.path.join(self._task_dir, f"training_progress_turn_{round_idx}.png")
                if os.path.exists(fig_path):
                    self.log_figure(fig_path, round_idx)
            
            # Log artifacts periodically
            if self.config.log_artifacts and round_idx % self.config.artifact_log_frequency == 0:
                self._log_artifacts_async(round_idx)
                
        except Exception as e:
            print(f"[WandB] Warning: Failed to log metrics from collector: {e}")

    def log_evaluation_metrics(self, metrics: Dict[str, float], prefix: str = "eval"):
        """Log evaluation metrics (recall, ndcg, etc.)."""
        prefixed_metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}
        self.log_metrics(prefixed_metrics)
    
    def log_attack_summary(self, summary: Dict[str, Any]):
        """Log attack summary statistics."""
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
    
    def _log_artifacts_async(self, round_idx: int):
        """Log artifacts in background thread."""
        def _log():
            self._log_llm_judge_artifact(round_idx)
            self._log_interaction_artifact(round_idx)
            self._log_conversation_artifact(round_idx)
        
        thread = threading.Thread(target=_log, daemon=True)
        thread.start()
    
    def _log_llm_judge_artifact(self, round_idx: int):
        """Log LLM judge history as artifact."""
        if not self.llm_judge_log_path or not os.path.exists(self.llm_judge_log_path):
            return
        
        try:
            artifact = wandb.Artifact(
                name=f"llm_judge_log_round_{round_idx}",
                type="llm_judge_history",
                description=f"LLM judge evaluation history up to round {round_idx}"
            )
            artifact.add_file(self.llm_judge_log_path)
            self.run.log_artifact(artifact)
            print(f"[WandB] 📦 Logged LLM judge artifact for round {round_idx}")
        except Exception as e:
            print(f"[WandB] Warning: Failed to log LLM judge artifact: {e}")
    
    def _log_interaction_artifact(self, round_idx: int):
        """Log interaction log as artifact."""
        if not self.interaction_log_path or not os.path.exists(self.interaction_log_path):
            return
        
        try:
            artifact = wandb.Artifact(
                name=f"interaction_log_round_{round_idx}",
                type="interaction_history",
                description=f"Agent interaction history up to round {round_idx}"
            )
            artifact.add_file(self.interaction_log_path)
            self.run.log_artifact(artifact)
            print(f"[WandB] 📦 Logged interaction artifact for round {round_idx}")
        except Exception as e:
            print(f"[WandB] Warning: Failed to log interaction artifact: {e}")
    
    def _log_conversation_artifact(self, round_idx: int):
        """Log conversation directory as artifact."""
        if not self._task_dir:
            return
        
        conv_dir = os.path.join(self._task_dir, "conversations")
        if not os.path.exists(conv_dir):
            return
        
        try:
            artifact = wandb.Artifact(
                name=f"conversations_round_{round_idx}",
                type="conversations",
                description=f"Conversation logs up to round {round_idx}"
            )
            artifact.add_dir(conv_dir)
            self.run.log_artifact(artifact)
            print(f"[WandB] 📦 Logged conversations artifact for round {round_idx}")
        except Exception as e:
            print(f"[WandB] Warning: Failed to log conversations artifact: {e}")

    def log_final_artifacts(self):
        """Log final artifacts at end of experiment.
        
        This logs everything needed for judge_calibration_connacf.py:
        - stdout log
        - conversation logs (interaction_log.txt)
        - LLM judge log
        - All turn JSONs
        - All training progress figures
        """
        if not self._initialized or not self.run:
            return
        
        print("[WandB] 📦 Logging final artifacts...")
        
        # Log all remaining turn JSONs
        self.log_all_turn_jsons()
        
        # Log all remaining figures
        self.log_all_figures()
        
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
        
        # Log final interaction log
        if self.interaction_log_path and os.path.exists(self.interaction_log_path):
            try:
                artifact = wandb.Artifact(
                    name="interaction_log_final",
                    type="interaction_history",
                    description="Complete agent interaction history"
                )
                artifact.add_file(self.interaction_log_path)
                self.run.log_artifact(artifact)
                print(f"[WandB] ✅ Logged final interaction artifact")
            except Exception as e:
                print(f"[WandB] Warning: Failed to log final interaction artifact: {e}")
        
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
        
        # Log conversation directory
        if self._task_dir:
            conv_dir = os.path.join(self._task_dir, "conversations")
            if os.path.exists(conv_dir):
                try:
                    artifact = wandb.Artifact(
                        name="conversations_final",
                        type="conversations",
                        description="Complete conversation logs directory"
                    )
                    artifact.add_dir(conv_dir)
                    self.run.log_artifact(artifact)
                    print(f"[WandB] ✅ Logged conversations directory artifact")
                except Exception as e:
                    print(f"[WandB] Warning: Failed to log conversations artifact: {e}")
        
        # Log all turn JSON files as a single artifact
        if self._task_dir and os.path.exists(self._task_dir):
            try:
                json_files = [f for f in os.listdir(self._task_dir) if f.startswith('turn_') and f.endswith('.json')]
                if json_files:
                    artifact = wandb.Artifact(
                        name="turn_metrics_all",
                        type="turn_metrics",
                        description="All per-turn metrics JSON files"
                    )
                    for json_file in sorted(json_files):
                        artifact.add_file(os.path.join(self._task_dir, json_file))
                    self.run.log_artifact(artifact)
                    print(f"[WandB] ✅ Logged {len(json_files)} turn metrics JSON files")
            except Exception as e:
                print(f"[WandB] Warning: Failed to log turn metrics JSON files: {e}")
        
        # Log resume state for future resumption
        if self._task_dir:
            resume_path = os.path.join(self._task_dir, "resume_state.json")
            if os.path.exists(resume_path):
                try:
                    artifact = wandb.Artifact(
                        name="resume_state",
                        type="resume",
                        description="Resume state for continuing experiment"
                    )
                    artifact.add_file(resume_path)
                    self.run.log_artifact(artifact)
                    print(f"[WandB] ✅ Logged resume state artifact")
                except Exception as e:
                    print(f"[WandB] Warning: Failed to log resume state: {e}")
        
        # Log all training progress figures as a single artifact
        if self._task_dir and os.path.exists(self._task_dir):
            try:
                fig_files = [f for f in os.listdir(self._task_dir) if f.startswith('training_progress_') and f.endswith('.png')]
                if fig_files:
                    artifact = wandb.Artifact(
                        name="training_figures_all",
                        type="figures",
                        description="All training progress figures"
                    )
                    for fig_file in sorted(fig_files):
                        artifact.add_file(os.path.join(self._task_dir, fig_file))
                    self.run.log_artifact(artifact)
                    print(f"[WandB] ✅ Logged {len(fig_files)} training progress figures")
            except Exception as e:
                print(f"[WandB] Warning: Failed to log training figures: {e}")

    def log_table(self, name: str, data: List[Dict[str, Any]], columns: Optional[List[str]] = None):
        """Log a table to wandb."""
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
        """Log contamination tracking table."""
        columns = [
            "round", "agent_type", "agent_id", "is_contaminated",
            "contamination_score", "detected_canaries", "reasoning"
        ]
        self.log_table("contamination_tracking", contamination_data, columns)
    
    def save_wandb_resume_info(self):
        """Save wandb run info for future resumption."""
        if not self._initialized or not self.run or not self._task_dir:
            return
        
        resume_info = {
            "wandb_run_id": self.run.id,
            "wandb_run_name": self.run.name,
            "wandb_project": self.config.project,
            "wandb_entity": self.config.entity,
            "wandb_url": self.run.url,
            "timestamp": datetime.now().isoformat()
        }
        
        resume_info_path = os.path.join(self._task_dir, "wandb_resume_info.json")
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
                print(f"[WandB] ✅ Run finished: {self.run.url}")
            
            self._initialized = False
            
        except Exception as e:
            print(f"[WandB] Warning: Error during finish: {e}")
    
    def __enter__(self):
        self.init()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.finish()
        return False


def create_wandb_logger(
    attack_config_path: Optional[str] = None,
    model_config: Optional[Any] = None,
    dataset_name: str = "",
    output_dir: str = "",
    task_id: int = 0,
    project: str = "connacf-attacks",
    entity: Optional[str] = None,
    run_name: Optional[str] = None,
    tags: Optional[List[str]] = None,
    log_artifacts: bool = True,
    log_stdout: bool = True,
    log_turn_jsons: bool = True,
    log_figures: bool = True,
    resume_run_id: Optional[str] = None
) -> Optional[WandBLogger]:
    """Factory function to create WandBLogger.
    
    Args:
        attack_config_path: Path to attack config YAML
        model_config: RecBole Config object or dict
        dataset_name: Dataset name
        output_dir: Output directory
        task_id: Task ID
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
        WandBLogger instance or None if wandb not available
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
    wandb_config = WandBConfig(
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
    logger = WandBLogger(
        config=wandb_config,
        attack_config=attack_config,
        model_config=model_config,
        dataset_name=dataset_name,
        output_dir=output_dir,
        task_id=task_id
    )
    
    return logger


def load_wandb_resume_info(task_dir: str) -> Optional[Dict[str, Any]]:
    """Load wandb resume info from a previous run.
    
    Args:
        task_dir: Path to task directory
        
    Returns:
        Dict with wandb_run_id, wandb_project, etc. or None
    """
    resume_info_path = os.path.join(task_dir, "wandb_resume_info.json")
    if os.path.exists(resume_info_path):
        try:
            with open(resume_info_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"[WandB] Warning: Failed to load resume info: {e}")
    return None
