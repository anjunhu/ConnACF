#!/usr/bin/env python3
"""
Simple ConnaCF Attack Runner - Mimics Original run.py

This version exactly follows the original run.py structure and just adds
attack integration at the right point.

Now with extensive wandb logging support:
- Full config logging
- Stdout capture
- LLM judge history artifacts
- Interaction log artifacts
- Training metrics
"""

import sys
from logging import getLogger
import argparse
import logging
import os
import re
import yaml

# Suppress RecBole pandas warnings
import suppress_warnings

from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.data.transform import construct_transform
from recbole.utils import init_logger, get_trainer, init_seed, set_color, get_flops
from trainer import LanguageLossTrainer, SelectedUserTrainer, ITEMLanguageLossTrainer
from utils import get_model
from dataset import BPRDataset, ITEMBPRDataset

# WandB logging
from wandb_logger import WandBLogger, WandBConfig, create_wandb_logger, WANDB_AVAILABLE

# Add attack module to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'attack'))

# Add parent directory so 'connacf' is importable as a top-level package.
# The macf subpackage uses absolute imports like 'from connacf.macf.config import ...'
_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

# Disable only the specific botocore.tokens INFO logging
logging.getLogger('botocore.tokens').setLevel(logging.WARNING)

# Global wandb logger instance
_wandb_logger = None


def parse_resume_state(resume_dir: str):
    """
    Parse a previous run's output directory to extract agent memories,
    attacker indices, and progress metadata needed for resumption.
    
    Reads from two sources:
      1. ``resume_state.json`` — attacker indices and framework metadata
         (written by ``_save_resume_checkpoint`` during ``initialize_attack_framework``).
      2. ``interaction_log.txt`` — agent memories (last profile/description
         logged for each agent) and the last global turn number.
    
    Args:
        resume_dir: Path to the task directory (e.g.
            ``attack_output/netsafe/misinfo_repro_2cand/ml-100k-100user-sparse/task_0``)
            **or** the parent output directory (will auto-detect the latest ``task_*``).
    
    Returns:
        dict with keys:
        - ``global_turn``, ``batch``, ``round`` — last completed turn info
        - ``user_memories`` — ``{int: str}`` latest profile per user
        - ``item_memories`` — ``{int: str}`` latest description per item
        - ``attacker_users`` — ``set[int]`` from interaction log
        - ``attacker_items`` — ``set[int]`` from interaction log
        - ``attacker_user_indices_saved`` — ``list[int] | None`` from resume_state.json
        - ``attacker_item_indices_saved`` — ``list[int] | None`` from resume_state.json
        - ``completed_epochs`` — estimated number of fully completed epochs
        - ``all_update_rounds`` — rounds per batch (from resume_state.json or default 2)
        - ``log_path`` — path to the interaction log
    """
    import json as _json
    
    # ---- Resolve task directory ----
    task_dir = resume_dir
    if not os.path.isdir(os.path.join(resume_dir, 'conversations')):
        # Caller passed the parent output dir — find latest task_*
        candidates = sorted(
            [d for d in os.listdir(resume_dir) if d.startswith('task_')],
            reverse=True,
        )
        for c in candidates:
            if os.path.isdir(os.path.join(resume_dir, c, 'conversations')):
                task_dir = os.path.join(resume_dir, c)
                break
    
    # ---- 1. Load resume_state.json (attacker indices + metadata) ----
    resume_json_path = os.path.join(task_dir, 'resume_state.json')
    saved_state = {}
    if os.path.exists(resume_json_path):
        try:
            with open(resume_json_path, 'r') as f:
                saved_state = _json.load(f)
            print(f"[RESUME] ✓ Loaded resume_state.json from {resume_json_path}")
        except Exception as e:
            print(f"[RESUME] ⚠ Could not load resume_state.json: {e}")
    else:
        print(f"[RESUME] ⚠ No resume_state.json found (older run?) — attacker indices will be re-randomized")
    
    # ---- 2. Parse interaction log for memories ----
    log_path = os.path.join(task_dir, 'conversations', 'interaction_log.txt')
    if not os.path.exists(log_path):
        print(f"[RESUME] No interaction log found at {log_path}")
        return None
    
    print(f"[RESUME] Parsing interaction log: {log_path}")
    
    turn_header = re.compile(r'BATCH (\d+) \| ROUND (\d+) \| GLOBAL TURN (\d+)')
    user_pattern = re.compile(
        r'(?:🔴 ATTACKER )?👤 USER #(\d+):\s*─+\s*Profile:\s*(.*?)\s*─+', re.DOTALL)
    item_pattern = re.compile(
        r'(?:🔴 ATTACKER )?💿 ITEM #(\d+)(?:\s*-\s*[^:]+)?:\s*─+\s*Description:\s*(.*?)\s*─+', re.DOTALL)
    attacker_user_pattern = re.compile(
        r'🔴 ATTACKER USER #(\d+):\s*─+\s*Profile:\s*(.*?)\s*─+', re.DOTALL)
    attacker_item_pattern = re.compile(
        r'🔴 ATTACKER ITEM #(\d+)(?:\s*-\s*[^:]+)?:\s*─+\s*Description:\s*(.*?)\s*─+', re.DOTALL)
    
    with open(log_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Last turn header
    last_batch, last_round, last_global_turn = 0, 0, 0
    for match in turn_header.finditer(content):
        last_batch = int(match.group(1))
        last_round = int(match.group(2))
        last_global_turn = int(match.group(3))
    
    # Agent memories (last occurrence wins)
    user_memories = {}
    item_memories = {}
    attacker_users = set()
    attacker_items = set()
    
    for match in attacker_user_pattern.finditer(content):
        uid = int(match.group(1))
        attacker_users.add(uid)
        user_memories[uid] = match.group(2).strip()
    
    for match in attacker_item_pattern.finditer(content):
        iid = int(match.group(1))
        attacker_items.add(iid)
        item_memories[iid] = match.group(2).strip()
    
    for match in user_pattern.finditer(content):
        user_memories[int(match.group(1))] = match.group(2).strip()
    
    for match in item_pattern.finditer(content):
        item_memories[int(match.group(1))] = match.group(2).strip()
    
    # ---- 3. Estimate completed epochs ----
    all_update_rounds = saved_state.get('all_update_rounds', 2)
    # turns_per_batch = all_update_rounds (each batch runs this many rounds)
    # completed_turns = last_global_turn (0-indexed, so turn N means N turns done before it)
    # We need to figure out batches_per_epoch from the data, but we don't have
    # the dataset here.  Instead, count distinct batch numbers in the log.
    batch_numbers = set()
    for match in turn_header.finditer(content):
        batch_numbers.add(int(match.group(1)))
    total_batches_seen = max(batch_numbers) if batch_numbers else 0
    
    # Heuristic: if we see batch N with round (all_update_rounds-1), that batch
    # is fully done.  The last batch may be partial.
    last_batch_complete = (last_round == all_update_rounds - 1)
    completed_batches = last_batch if last_batch_complete else last_batch - 1
    
    # We don't know batches_per_epoch without the dataset, so we store the raw
    # counts and let the caller compute it.
    
    print(f"[RESUME] Found state at global_turn={last_global_turn}, batch={last_batch}, round={last_round}")
    print(f"[RESUME] Extracted {len(user_memories)} user memories, {len(item_memories)} item memories")
    print(f"[RESUME] Identified {len(attacker_users)} attacker users, {len(attacker_items)} attacker items (from log)")
    if saved_state:
        print(f"[RESUME] resume_state.json: attack_method={saved_state.get('attack_method')}, "
              f"attacker_users={len(saved_state.get('attacker_user_indices', []))}, "
              f"attacker_items={len(saved_state.get('attacker_item_indices', []))}")
    
    return {
        'global_turn': last_global_turn,
        'batch': last_batch,
        'round': last_round,
        'user_memories': user_memories,
        'item_memories': item_memories,
        'attacker_users': attacker_users,
        'attacker_items': attacker_items,
        # From resume_state.json (None if not available)
        'attacker_user_indices_saved': saved_state.get('attacker_user_indices'),
        'attacker_item_indices_saved': saved_state.get('attacker_item_indices'),
        'all_update_rounds': all_update_rounds,
        'completed_batches': completed_batches,
        'total_batches_seen': total_batches_seen,
        'log_path': log_path,
        'task_dir': task_dir,
    }


def apply_resume_state(model, resume_state: dict):
    """
    Apply resumed state to model's agents.
    
    Delegates to model.apply_resume_state() if available (the mixin method),
    otherwise falls back to direct memory patching for backward compatibility.
    """
    if not resume_state:
        return
    
    if hasattr(model, 'apply_resume_state'):
        model.apply_resume_state(resume_state)
    else:
        # Legacy fallback — just patch memories and counters
        for user_id, profile in resume_state.get('user_memories', {}).items():
            if user_id in model.user_agents:
                agent = model.user_agents[user_id]
                if hasattr(agent, 'update_memory'):
                    if len(agent.update_memory) > 0:
                        agent.update_memory[-1] = profile
                    else:
                        agent.update_memory.append(profile)
        
        for item_id, desc in resume_state.get('item_memories', {}).items():
            if item_id in model.item_agents:
                agent = model.item_agents[item_id]
                if hasattr(agent, 'update_memory'):
                    if len(agent.update_memory) > 0:
                        agent.update_memory[-1] = desc
                    else:
                        agent.update_memory.append(desc)
        
        if hasattr(model, 'global_turn_counter'):
            model.global_turn_counter = resume_state.get('global_turn', 0) + 1
        if hasattr(model, 'batch_counter'):
            model.batch_counter = resume_state.get('batch', 0)
        
        print(f"[RESUME] Applied state (legacy path)")
        print(f"[RESUME] Continuing from global_turn={resume_state.get('global_turn', 0) + 1}")


def load_attack_config(config_path: str):
    """Load attack configuration from YAML file"""
    if not os.path.exists(config_path):
        print(f"Attack config file not found: {config_path}")
        return {}
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def integrate_attacks_simple(model, attack_config_path, resume_dir=None):
    """Simple attack integration that happens after model is fully created"""
    if not attack_config_path or not os.path.exists(attack_config_path):
        print("No attack config provided, running without attacks")
        return model
    
    print(f"[DEBUG] Integrating attacks into ConnaCF...")
    print(f"[DEBUG] Attack config path: {attack_config_path}")
    
    try:
        from attack.integration.connacf_attack_integration import integrate_attacks_into_connacf
        print(f"[DEBUG] Calling integrate_attacks_into_connacf...")
        model_with_attacks = integrate_attacks_into_connacf(model, attack_config_path, resume_dir=resume_dir)
        
        # Add round tracking capability
        model_with_attacks.current_round = 0
        model_with_attacks.round_metrics = []
        model_with_attacks.collect_round_metrics = True
        
        print("✅ Attack integration successful")
        print("✅ Round tracking initialized")
        return model_with_attacks
    except Exception as e:
        print(f"⚠️  Attack integration failed: {e}")
        print("Continuing without attacks...")
        import traceback
        traceback.print_exc()
        
        # Initialize minimal attributes to prevent AttributeError
        model.attack_enabled = False
        model.batch_counter = 0
        model.global_turn_counter = 0
        
        return model


def run_baseline_with_attacks(model_name, dataset_name, args, attack_config_path=None, verbose_logging=True, wandb_logger=None, **kwargs):
    """Modified run_baseline that includes attack integration - follows original structure exactly"""
    
    global _wandb_logger
    _wandb_logger = wandb_logger
    
    # EXACT COPY of original run.py structure
    props = ['props/overall.yaml', f'props/{model_name}.yaml', f'props/{dataset_name}.yaml']
    print(props)

    # CRITICAL: Use get_model EXACTLY like the original - no custom model classes here
    print(f"[DEBUG] About to call get_model({model_name})...")
    model_class = get_model(model_name)
    print(f"[DEBUG] get_model returned: {model_class}")

    # configurations initialization
    print(f"[DEBUG] Initializing config...")
    # Safety: ensure 'dataset' in kwargs is never a dict (attack YAML can leak it)
    if 'dataset' in kwargs and isinstance(kwargs['dataset'], dict):
        print(f"[WARNING] Removing dict-valued 'dataset' from kwargs: {kwargs['dataset']}")
        del kwargs['dataset']
    # Force log/ and log_tensorboard/ to repo root (parent of connacf/) so they are
    # independent of the working directory used to launch the script.
    _repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    kwargs.setdefault('record_path', os.path.join(_repo_root, 'log'))
    kwargs.setdefault('log_tensorboard', os.path.join(_repo_root, 'log_tensorboard'))

    config = Config(
        model=model_class,
        dataset=dataset_name,
        config_file_list=props,
        config_dict=kwargs,
    )
    print(f"[DEBUG] Config initialized successfully")
    
    # Update wandb logger with model config
    if wandb_logger and hasattr(wandb_logger, 'model_config'):
        wandb_logger.model_config = config
    
    init_seed(config["seed"], config["reproducibility"])
    # logger initialization
    init_logger(config)
    logger = getLogger()
    
    # Set logger level based on verbosity preference
    if not verbose_logging:
        logger.setLevel(logging.WARNING)
        # Also suppress RecBole's internal loggers
        logging.getLogger('recbole').setLevel(logging.WARNING)
    
    # Conditionally log verbose information
    if verbose_logging:
        logger.info(sys.argv)
        logger.info(config)

    print(f"[DEBUG] Setting up dataset...")
    # dataset filtering - EXACT COPY
    if model_name == 'BPR' or model_name == 'UUPretrain' or model_name == 'ReRec' or model_name == 'AllReRec' or model_name == 'IITest' or model_name == 'TestGames' or model_name == 'SparseReRec'\
            or model_name in ['TestPantry', 'TestOffice', 'IITestDiag', 'IITestDiagNew', 'TestOfficeBPR', 'TestOfficeUUPretrain','UserReRec','ConnaCF']:
        dataset = BPRDataset(config)
    elif model_name in ['UUTest','UUTestDiag','TestOfficeUUTest']:
        dataset = ITEMBPRDataset(config)
    else:
        dataset = create_dataset(config)

    if verbose_logging:
        logger.info(dataset)
    print(f"[DEBUG] Dataset created successfully")

    # dataset splitting
    print(f"[DEBUG] Preparing data...")
    train_data, valid_data, test_data = data_preparation(config, dataset)
    print(f"[DEBUG] Data preparation completed")

    # model loading and initialization - EXACT COPY
    print(f"[DEBUG] Initializing model...")
    init_seed(config["seed"] + config["local_rank"], config["reproducibility"])
    model = model_class(config, train_data._dataset).to(config["device"])
    
    # CRITICAL: Store train_data in model for subset filtering
    model.train_data = train_data
    
    if verbose_logging:
        logger.info(model)
    print(f"[DEBUG] Model initialized successfully")

    # ATTACK INTEGRATION POINT - Only difference from original
    # This happens AFTER model is fully initialized
    print(f"[DEBUG] Integrating attacks...")
    # Resolve resume_dir early so initialize_attack_framework can reuse the original task_dir
    resume_dir = getattr(args, 'resume_dir', None)
    if resume_dir and not os.path.exists(resume_dir):
        raise RuntimeError(
            f"[RESUME] --resume_dir directory does not exist: {resume_dir}\n"
            f"Check the path and try again. A typo here causes a silent fresh start."
        )
    model = integrate_attacks_simple(model, attack_config_path, resume_dir=resume_dir)

    if args.max_turns is not None and hasattr(model, 'global_turn_counter'):
        model.max_turns = args.max_turns
        print(f"[MAX_TURNS] Will stop after {args.max_turns} global turns")
    
    # CRITICAL FIX: If subset filtering was applied, we need to RECREATE the DataLoader
    # because PyTorch DataLoader's internal state is immutable after creation.
    # The filtering modified dataset.inter_feat but the DataLoader still has old indices.
    if hasattr(model, 'subset_mode') and model.subset_mode and hasattr(model, '_subset_filtered_dataset'):
        print(f"[DEBUG] Recreating DataLoader after subset filtering...")
        print(f"[DEBUG] Current dataset.inter_feat size: {len(dataset.inter_feat)}")
        
        # Import the helper function
        from attack.integration.subset_selector import recreate_dataloader_for_filtered_dataset
        
        # Recreate DataLoaders with the filtered dataset
        new_train_data, new_valid_data, new_test_data = recreate_dataloader_for_filtered_dataset(config, dataset, train_data)
        
        # Update train_data (always returned)
        train_data = new_train_data
        model.train_data = train_data
        
        # Update valid/test only if returned (might be None if only train was recreated)
        if new_valid_data is not None:
            valid_data = new_valid_data
        if new_test_data is not None:
            test_data = new_test_data
        
        print(f"[DEBUG] DataLoader recreated with {len(train_data._dataset.inter_feat)} interactions")
    
    # Store validation data in model for per-round evaluation
    if hasattr(model, 'attack_enabled') and model.attack_enabled:
        model.valid_data = valid_data
        model.test_data = test_data
        print(f"[DEBUG] Stored validation/test data in model for per-round evaluation")
    
    # ==================== UPDATE WANDB LOGGER WITH TIMESTAMPED TASK_DIR ====================
    # The attack framework creates a timestamped task directory, update wandb to use it
    if wandb_logger and hasattr(model, '_task_dir') and model._task_dir:
        wandb_logger.set_task_dir(model._task_dir)
        print(f"[WandB] Updated task directory to: {model._task_dir}")
    
    print(f"[DEBUG] Attack integration completed")
    
    # ==================== RESUME FROM PREVIOUS STATE ====================
    # resume_dir was already validated above; apply agent memories and counters now
    resume_state = None
    if resume_dir:
        print(f"[RESUME] Loading state from: {resume_dir}")
        resume_state = parse_resume_state(resume_dir)
        if resume_state:
            apply_resume_state(model, resume_state)
            print(f"[RESUME] ✅ Successfully resumed from previous state")
        else:
            raise RuntimeError(
                f"[RESUME] --resume_dir was specified but no usable state could be parsed "
                f"from {resume_dir}. Check that interaction_log.txt exists and is non-empty."
            )
    elif getattr(args, 'resume', False):
        # Legacy --resume flag: derive directory from attack config
        output_dir = None
        if attack_config_path:
            attack_config = load_attack_config(attack_config_path)
            output_dir = attack_config.get('attack', {}).get('output', {}).get('output_directory')
            if output_dir:
                _repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                output_dir = os.path.join(_repo_root, output_dir, dataset_name)
        
        if output_dir and os.path.exists(output_dir):
            print(f"[RESUME] Looking for previous state in: {output_dir}")
            resume_state = parse_resume_state(output_dir)
            if resume_state:
                apply_resume_state(model, resume_state)
                # Reuse the original task directory so output stays in one place
                resume_dir = resume_state.get('task_dir') or output_dir
                print(f"[RESUME] ✅ Successfully resumed from previous state (task_dir={resume_dir})")
            else:
                print(f"[RESUME] No previous state found, starting fresh")
        else:
            print(f"[RESUME] Output directory not found: {output_dir}")

    print(f"[DEBUG] Setting up trainer...")
    transform = construct_transform(config)
    # flops = get_flops(model, dataset, config["device"], logger, transform)
    # logger.info(set_color("FLOPs", "blue") + f": {flops}")

    # trainer loading and initialization - EXACT COPY
    if model_name in ['SASRec','BPRMF']:
        trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)
    elif model_name in ['UUTest', 'UUTestDiag','TestOfficeUUTest','TestOfficeUUTestDiag']:
        trainer = ITEMLanguageLossTrainer(config,model,dataset)
    else:
        trainer = LanguageLossTrainer(config,model,dataset)
    
    print(f"[DEBUG] Trainer initialized successfully")
    
    # ==================== EPOCH SKIPPING FOR RESUME ====================
    if resume_state is not None:
        # Compute how many full epochs were completed so we can skip them.
        # batches_per_epoch = ceil(n_interactions / train_batch_size)
        n_interactions = len(train_data._dataset.inter_feat)
        batch_size = config['train_batch_size']
        batches_per_epoch = max(1, -(-n_interactions // batch_size))  # ceil division
        
        completed_batches = resume_state.get('completed_batches', 0)
        skip_epochs = completed_batches // batches_per_epoch
        
        if skip_epochs > 0 and skip_epochs < config['epochs']:
            trainer.start_epoch = skip_epochs
            print(f"[RESUME] Skipping {skip_epochs} completed epochs "
                  f"({completed_batches} batches done, {batches_per_epoch} per epoch)")
            print(f"[RESUME] Training will start from epoch {skip_epochs}/{config['epochs']}")
        else:
            print(f"[RESUME] No full epochs to skip (completed_batches={completed_batches}, "
                  f"batches_per_epoch={batches_per_epoch})")
    
    # Add round tracking hooks if attack framework is enabled
    if hasattr(model, 'metrics_collector') and hasattr(model, 'collect_round_metrics'):
        print("[DEBUG] Setting up round tracking hooks...")
        
        # Hook into trainer's _train_epoch to collect metrics after each epoch
        if hasattr(trainer, '_train_epoch'):
            original_train_epoch = trainer._train_epoch
            def train_epoch_with_tracking(*args, **kwargs):
                result = original_train_epoch(*args, **kwargs)
                
                # Collect metrics after each epoch
                if hasattr(model, 'collect_and_visualize_round_metrics'):
                    print(f"[ROUND_TRACKING] Collecting metrics after epoch...")
                    model.collect_and_visualize_round_metrics()
                
                # Log to wandb if available
                if wandb_logger and hasattr(model, 'metrics_collector'):
                    try:
                        round_idx = getattr(model, 'global_turn_counter', 0)
                        
                        # Use the new method that reads directly from metrics collector
                        wandb_logger.log_metrics_from_collector(model.metrics_collector, round_idx)
                        
                    except Exception as e:
                        print(f"[WandB] Warning: Failed to log round metrics: {e}")
                        import traceback
                        traceback.print_exc()
                
                return result
            
            trainer._train_epoch = train_epoch_with_tracking
            print("[DEBUG] Round tracking hooks installed (per-epoch)")
        else:
            # Fallback: Hook into trainer.fit to collect metrics after training completes
            original_fit = trainer.fit
            def fit_with_round_tracking(*args, **kwargs):
                print(f"[ROUND_TRACKING] Starting training with round tracking...")
                result = original_fit(*args, **kwargs)
                
                # Collect metrics after training completes
                if hasattr(model, 'collect_and_visualize_round_metrics'):
                    print(f"[ROUND_TRACKING] Collecting post-training metrics...")
                    model.collect_and_visualize_round_metrics()
                
                return result
            
            trainer.fit = fit_with_round_tracking
            print("[DEBUG] Round tracking hooks installed (post-training)")

    if not config['test_only']:
        print(f"[DEBUG] Starting training...")
        # # model training
        trainer.fit(train_data, valid_data, saved=True, show_progress=config["show_progress"])
        print(f"[DEBUG] Training completed")
        
        # Finalize attack analysis if available
        if hasattr(model, 'finalize_attack_analysis'):
            print("[DEBUG] Finalizing attack analysis...")
            model.finalize_attack_analysis()
            
        # Generate final visualizations if attack framework is enabled
        if hasattr(model, 'metrics_collector') and hasattr(model, 'generate_final_visualizations'):
            print("[DEBUG] Generating final attack visualizations...")
            model.generate_final_visualizations()

    # model evaluation - EXACT COPY
    print(f"[DEBUG] Starting evaluation...")
    
    test_result = trainer.evaluate(test_data, model_file=args.ckpt, load_best_model=True, show_progress=config["show_progress"])
    
    # Collect final evaluation metrics if attack framework is enabled
    if hasattr(model, 'metrics_collector') and hasattr(model, 'collect_round_metrics'):
        if hasattr(model, 'collect_and_visualize_round_metrics'):
            print(f"[ROUND_TRACKING] Collecting final evaluation metrics...")
            model.collect_and_visualize_round_metrics()
    
    print(f"[DEBUG] Evaluation completed")
    print(test_result)
    # logger.info(set_color("best valid ", "yellow") + f": {best_valid_result}")
    logger.info(set_color("test result", "yellow") + f": {test_result}")
    
    # Log final evaluation metrics to wandb
    if wandb_logger:
        try:
            # Convert test_result to dict if needed
            if hasattr(test_result, 'items'):
                eval_metrics = dict(test_result)
            else:
                eval_metrics = {'test_result': str(test_result)}
            wandb_logger.log_evaluation_metrics(eval_metrics, prefix="final_eval")
            
            # Log attack summary
            if hasattr(model, 'metrics_collector'):
                attack_summary = model.metrics_collector.get_summary_statistics()
                wandb_logger.log_attack_summary(attack_summary)
        except Exception as e:
            print(f"[WandB] Warning: Failed to log final metrics: {e}")

    return model_name, dataset_name, {
        # "best_valid_score": best_valid_score,
        # "valid_score_bigger": config["valid_metric_bigger"],
        # "best_valid_result": best_valid_result,
        "test_result": test_result,
        "attack_summary": model.metrics_collector.get_summary_statistics() if hasattr(model, 'metrics_collector') else {}
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", type=str, default="ConnaCF", help="name of models")
    parser.add_argument("--epochs", "-e", type=int, default=5, help="Number of epochs")
    parser.add_argument("--max-turns", type=int, default=None, dest="max_turns",
                        help="Stop after this many global turns (overrides -e for Phase 1 data collection)")
    parser.add_argument("--dataset", "-d", type=str, default="ml-100k-20-user-dense", help="name of datasets")
    parser.add_argument("--attack_config", "-a", type=str, default=None, help="Path to attack configuration file")
    parser.add_argument("--ckpt", "-c", type=str, default=None, help="path to checkpoint")
    
    # Logging control
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress verbose logging of hyperparameters and configs")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging (default)")
    
    # Resume from previous run
    parser.add_argument("--resume", "-r", action="store_true",
                        help="Resume from previous run (loads agent memories and continues from last turn)")
    parser.add_argument("--resume_dir", type=str, default=None,
                        help="Path to task directory to resume from (e.g. attack_output/netsafe/.../task_0)")
    
    # Additional arguments that can be passed to config
    parser.add_argument("--train_batch_size", type=int, default=20, help="Training batch size")
    parser.add_argument("--eval_batch_size", type=int, default=200, help="Evaluation batch size")
    parser.add_argument("--max_his_len", type=int, default=20, help="Max history length")
    parser.add_argument("--MAX_ITEM_LIST_LENGTH", type=int, default=20, help="Max item list length")
    parser.add_argument("--shuffle", type=bool, default=False, help="Shuffle data")
    parser.add_argument("--api_batch", type=int, default=20, help="API batch size")
    parser.add_argument("--test_only", type=bool, default=False, help="Test only mode")
    
    # U-U Interaction control (disabled by default)
    parser.add_argument("--enable_uu_interaction", "-uu", action="store_true",
                        help="Enable User-User friend consultation")
    parser.add_argument("--uu_friends_count", type=int, default=None,
                        help="Number of similar users to consult (default: 3)")
    
    # U-I Interaction control (disabled by default)
    parser.add_argument("--enable_ui_interaction", "-ui", action="store_true",
                        help="Enable User-Item dialogue")
    parser.add_argument("--ui_dialogue_rounds", type=int, default=None,
                        help="Number of dialogue rounds per item (default: 2)")
    
    # WandB logging arguments
    parser.add_argument("--wandb", "-w", action="store_true", default=False,
                        help="Enable wandb logging")
    parser.add_argument("--wandb_project", type=str, default="connacf-attacks",
                        help="WandB project name")
    parser.add_argument("--wandb_entity", type=str, default=None,
                        help="WandB entity/team name")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                        help="Custom WandB run name")
    parser.add_argument("--wandb_tags", type=str, nargs='+', default=None,
                        help="WandB tags (space-separated)")
    parser.add_argument("--no_wandb_artifacts", action="store_true", default=False,
                        help="Disable wandb artifact logging")
    parser.add_argument("--no_wandb_stdout", action="store_true", default=False,
                        help="Disable wandb stdout capture")
    parser.add_argument("--wandb_resume_id", type=str, default=None,
                        help="WandB run ID to resume (for continuing a previous run)")
    parser.add_argument("--gpu_id", "-g", type=int, default=None,
                        help="GPU device ID (default: auto-select freest GPU). Use to pin to a specific GPU.")
    parser.add_argument("--llm_model", "-l", type=str, default=None,
                        help="Override llm_model from ConnaCF.yaml (e.g. bedrock-claude, qwen.qwen3-32b-v1:0)")
    parser.add_argument("--judge_model", "-j", type=str, default='us.anthropic.claude-sonnet-4-5-20250929-v1:0',
                        help="LLM judge model (default: us.anthropic.claude-sonnet-4-5-20250929-v1:0)")

    args, unknown = parser.parse_known_args()
    
    # Determine verbosity (quiet takes precedence)
    verbose_logging = not args.quiet if not args.verbose else args.verbose
    
    # Build kwargs from provided arguments
    kwargs = {}
    if args.gpu_id is None:
        from connacf.utils.gpu_utils import free_gpu_id
        # Local HF models need much more VRAM than Bedrock (API-only) models
        _is_local = args.llm_model is not None and str(args.llm_model).startswith('hf-')
        args.gpu_id = free_gpu_id(min_free_mib=20000 if _is_local else 300)
        print(f"[GPU] Auto-selected physical GPU {args.gpu_id}")
    kwargs['gpu_id'] = args.gpu_id
    if args.llm_model is not None:
        kwargs['llm_model'] = args.llm_model
        print(f"LLM model override: {args.llm_model}")
    for arg_name in ['train_batch_size', 'eval_batch_size', 'max_his_len', 'MAX_ITEM_LIST_LENGTH', 
                     'epochs', 'shuffle', 'api_batch', 'test_only']:
        arg_value = getattr(args, arg_name)
        if arg_value is not None:
            kwargs[arg_name] = arg_value
    
    # Handle U-U interaction flag
    if args.enable_uu_interaction:
        kwargs['enable_uu_interaction'] = True
        print("U-U Interaction: ENABLED (via -uu flag)")
    
    if args.uu_friends_count is not None:
        kwargs['uu_friends_count'] = args.uu_friends_count
    
    # Handle U-I interaction flag
    if args.enable_ui_interaction:
        kwargs['enable_ui_interaction'] = True
        print("U-I Interaction: ENABLED (via -ui flag)")
    
    if args.ui_dialogue_rounds is not None:
        kwargs['ui_dialogue_rounds'] = args.ui_dialogue_rounds
    
    # Load attack config if provided
    if args.attack_config:
        print(f"[DEBUG] Checking attack config: {args.attack_config}")
        print(f"[DEBUG] File exists: {os.path.exists(args.attack_config)}")
        print(f"[DEBUG] Absolute path: {os.path.abspath(args.attack_config)}")
        if os.path.exists(args.attack_config):
            attack_config = load_attack_config(args.attack_config)
            # Filter keys that would collide with RecBole's config (e.g. 'dataset' must be a string)
            reserved_keys = {'dataset'}
            kwargs.update({k: v for k, v in attack_config.items() if k not in reserved_keys})
            # Promote attack-level fields that have props/ConnaCF.yaml defaults which
            # would otherwise silently override the per-experiment YAML values.
            _atk = attack_config.get('attack', {})
            _nc = _atk.get('num_candidates', None)
            if _nc is not None:
                kwargs['num_candidates'] = _nc
            for _field in ('disable_item_backward', 'disable_user_backward'):
                _val = _atk.get(_field, None)
                if _val is not None:
                    kwargs[_field] = _val
            # Apply judge model override (CLI -j flag wins over yaml)
            if 'llm_judge' in kwargs:
                kwargs['llm_judge']['llm_judge_model'] = args.judge_model
            print(f"Loaded attack config: {args.attack_config}")
        else:
            print(f"Attack config file not found: {args.attack_config}")
            args.attack_config = None
    else:
        print(f"[DEBUG] No attack config provided in args")

    # Auto-generate missing ranking data files for num_candidates >= 3
    _nc = kwargs.get('num_candidates', None)
    if _nc is not None and _nc >= 3:
        _dataset_dir = os.path.join('dataset', args.dataset)
        _ranking_file = os.path.join(_dataset_dir, f'ranking_train_{_nc}cand.json')
        if not os.path.isfile(_ranking_file):
            print(f"[AUTO-GEN] ranking_train_{_nc}cand.json not found for {args.dataset}, generating...")
            import subprocess, sys as _sys
            result = subprocess.run(
                [_sys.executable, 'tools/generate_binary_and_ranking.py',
                 '--data_path', _dataset_dir,
                 '--dataset_name', args.dataset,
                 '--num_candidates', str(_nc)],
                check=True
            )
            print(f"[AUTO-GEN] Done generating ranking_train_{_nc}cand.json")
        else:
            print(f"[AUTO-GEN] ranking_train_{_nc}cand.json already exists for {args.dataset}, skipping generation")

    print("="*60)
    print("CONNACF WITH SIMPLE ATTACK INTEGRATION")
    print("="*60)
    print(f"Model: {args.model}")
    _llm_display = kwargs.get('llm_model') or args.llm_model or "(from ConnaCF.yaml)"
    print(f"LLM: {_llm_display}")
    print(f"Dataset: {args.dataset}")
    print(f"Attack Config: {args.attack_config}")
    print(f"Resume Dir: {args.resume_dir or '(none)'}")
    print(f"Verbose Logging: {'Disabled' if args.quiet else 'Enabled'}")
    print(f"WandB Logging: {'Enabled' if args.wandb else 'Disabled'}")
    print("="*60)
    
    # Initialize wandb logger if enabled
    wandb_logger = None
    if args.wandb:
        # Get output directory from attack config
        output_dir = ""
        task_id = 0
        wandb_config_from_yaml = {}
        
        if attack_config:
            output_dir = attack_config.get('attack', {}).get('output', {}).get('output_directory', '')
            task_id = attack_config.get('attack', {}).get('current_task_id', 0)
            wandb_config_from_yaml = attack_config.get('attack', {}).get('wandb', {})
        
        # Merge CLI args with YAML config (CLI takes precedence)
        wandb_project = args.wandb_project or wandb_config_from_yaml.get('project', 'connacf-attacks')
        wandb_entity = args.wandb_entity or wandb_config_from_yaml.get('entity')
        wandb_run_name = args.wandb_run_name or wandb_config_from_yaml.get('run_name')
        wandb_tags = args.wandb_tags or wandb_config_from_yaml.get('tags')
        log_artifacts = not args.no_wandb_artifacts and wandb_config_from_yaml.get('log_artifacts', True)
        log_stdout = not args.no_wandb_stdout and wandb_config_from_yaml.get('log_stdout', True)
        
        # Check for wandb resume from previous run
        resume_run_id = getattr(args, 'wandb_resume_id', None)
        if not resume_run_id and args.resume_dir:
            # Try to load wandb resume info from previous run
            from wandb_logger import load_wandb_resume_info
            resume_info = load_wandb_resume_info(args.resume_dir)
            if resume_info:
                resume_run_id = resume_info.get('wandb_run_id')
                print(f"[WandB] Found previous run ID: {resume_run_id}")
        
        wandb_logger = create_wandb_logger(
            attack_config_path=args.attack_config,
            model_config=None,  # Will be set later
            dataset_name=args.dataset,
            output_dir=output_dir,
            task_id=task_id,
            project=wandb_project,
            entity=wandb_entity,
            run_name=wandb_run_name,
            tags=wandb_tags,
            log_artifacts=log_artifacts,
            log_stdout=log_stdout,
            log_turn_jsons=True,
            log_figures=True,
            resume_run_id=resume_run_id
        )
        
        if wandb_logger:
            wandb_logger.init(run_name=wandb_run_name)
    
    # Also check if wandb is enabled in config (without --wandb flag)
    elif args.attack_config and os.path.exists(args.attack_config):
        # Load attack config to check wandb settings
        attack_cfg = load_attack_config(args.attack_config)
        if attack_cfg and attack_cfg.get('attack', {}).get('wandb', {}).get('enable_wandb', False):
            print("[WandB] Enabled via config file")
            output_dir = attack_cfg.get('attack', {}).get('output', {}).get('output_directory', '')
            task_id = attack_cfg.get('attack', {}).get('current_task_id', 0)
            wandb_config_from_yaml = attack_cfg.get('attack', {}).get('wandb', {})
            
            wandb_logger = create_wandb_logger(
                attack_config_path=args.attack_config,
                model_config=None,
                dataset_name=args.dataset,
                output_dir=output_dir,
                task_id=task_id,
                project=wandb_config_from_yaml.get('project', 'connacf-attacks'),
                entity=wandb_config_from_yaml.get('entity'),
                run_name=wandb_config_from_yaml.get('run_name'),
                tags=wandb_config_from_yaml.get('tags'),
                log_artifacts=wandb_config_from_yaml.get('log_artifacts', True),
                log_stdout=wandb_config_from_yaml.get('log_stdout', True),
                log_turn_jsons=True,
                log_figures=True
            )
        
        if wandb_logger:
            wandb_logger.init()

    try:
        result = run_baseline_with_attacks(
            args.model, args.dataset, args, args.attack_config, 
            verbose_logging, wandb_logger=wandb_logger, **kwargs
        )
        print("\n✅ Experiment completed successfully!")
        
        # Log final artifacts and finish wandb
        if wandb_logger:
            wandb_logger.finish()
            
    except Exception as e:
        print(f"\n❌ Experiment failed: {e}")
        import traceback
        traceback.print_exc()
        
        # Still try to finish wandb to save partial results
        if wandb_logger:
            try:
                wandb_logger.finish()
            except:
                pass