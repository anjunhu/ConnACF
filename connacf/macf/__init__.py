"""
MACF (Multi-Agent Collaborative Filtering) Module

This module implements MACF as a standalone inference-only framework that
leverages collaborative filtering principles through LLM-based multi-agent
collaboration.

Unlike ConnaCF's optimization-based approach, MACF instantiates transient agents
representing similar users (neighbors) and relevant history items, coordinated by
an orchestrator agent through multi-round discussions.
"""

from connacf.macf.config import MACFConfig
from connacf.macf.index_manager import GlobalIndexManager
from connacf.macf.toolkit import MACFToolkit
from connacf.macf.data_models import (
    ToolCall,
    ItemSuggestion,
    AgentResponse,
    RankedList,
    DiscussionState,
    EvaluationResult,
)
from connacf.macf.agents import BaseMACFAgent, UserAgent, ItemAgent
from connacf.macf.error_handling import (
    LLMCallHandler,
    FallbackResponse,
    handle_empty_retrieval,
    parse_agent_response,
    pad_ranked_list,
)
from connacf.macf.orchestrator import MACFOrchestrator
from connacf.macf.stateful_orchestrator import StatefulOrchestrator, OrchestratorMemory, create_stateful_orchestrator
from connacf.macf.evaluator import MACFEvaluator
from connacf.macf.attack_hooks import AttackHooks, MessageInterceptor
from connacf.macf.conversation_logger import MACFConversationLogger
from connacf.macf.memory_store import MACFMemoryStore, AgentMemory, MemoryEntry
from connacf.macf.backward_pass import MACFBackwardPass
from connacf.macf.metrics_collector import (
    MACFMetricsCollector, 
    TurnMetrics, 
    TaskMetrics, 
    create_canary_aware_llm_judge_callback
)
from connacf.macf.wandb_logger import (
    MACFWandBLogger,
    MACFWandBConfig,
    create_macf_wandb_logger,
    WANDB_AVAILABLE
)

# Attack modules
from connacf.macf.drunk_attack import MACFDrunkAttacker
from connacf.macf.rectextattack import MACFRecTextAttacker
from connacf.macf.cheat_attack import MACFCheatAttacker
from connacf.macf.netsafe_attack import MACFNetSafeAttacker

__all__ = [
    'MACFConfig',
    'GlobalIndexManager',
    'MACFToolkit',
    'MACFOrchestrator',
    'ToolCall',
    'ItemSuggestion',
    'AgentResponse',
    'RankedList',
    'DiscussionState',
    'EvaluationResult',
    'BaseMACFAgent',
    'UserAgent',
    'ItemAgent',
    # Error handling
    'LLMCallHandler',
    'FallbackResponse',
    'handle_empty_retrieval',
    'parse_agent_response',
    'pad_ranked_list',
    # Evaluator
    'MACFEvaluator',
    # Attack hooks
    'AttackHooks',
    'MessageInterceptor',
    # Conversation logging
    'MACFConversationLogger',
    # Memory store
    'MACFMemoryStore',
    'AgentMemory',
    'MemoryEntry',
    # Backward pass (ConnaCF-compatible learning)
    'MACFBackwardPass',
    # Metrics collector
    'MACFMetricsCollector',
    'TurnMetrics',
    'TaskMetrics',
    'create_canary_aware_llm_judge_callback',
    # WandB logging
    'MACFWandBLogger',
    'MACFWandBConfig',
    'create_macf_wandb_logger',
    'WANDB_AVAILABLE',
    # Attack modules
    'MACFDrunkAttacker',
    'MACFRecTextAttacker',
    'MACFCheatAttacker',
    'MACFNetSafeAttacker',
]
