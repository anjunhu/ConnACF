# MACF (Multi-Agent Collaborative Filtering)

MACF is an inference-only framework that leverages collaborative filtering principles through LLM-based multi-agent collaboration. Unlike ConnaCF's optimization-based approach, MACF instantiates transient agents representing similar users (neighbors) and relevant history items, coordinated by an orchestrator through multi-round discussions.

## Quick Start

Run MACF inference with the `--macf_mode` flag:

```bash
cd connacf
python run.py --macf_mode --dataset CDs-100-user-dense --ckpt ./checkpoints/ConnaCF-Jan-10-2026_15-49-14.pth
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    MACFOrchestrator                         │
│  - Recruits agents via retrieval tools                      │
│  - Coordinates multi-round discussions                      │
│  - Aggregates responses into ranked list                    │
└─────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
       ┌──────────┐    ┌──────────┐    ┌──────────┐
       │UserAgent │    │UserAgent │    │ItemAgent │
       │(neighbor)│    │(neighbor)│    │(history) │
       └──────────┘    └──────────┘    └──────────┘
              │               │               │
              └───────────────┼───────────────┘
                              ▼
                    ┌─────────────────┐
                    │  MACFToolkit    │
                    │ - GetSimilarUsers│
                    │ - GetRelevantItems│
                    │ - RetrieveByQuery│
                    └─────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │GlobalIndexManager│
                    │ - User similarity│
                    │ - Item embeddings│
                    └─────────────────┘
```

## Configuration

Configuration is loaded from `props/MACF.yaml`:

```yaml
# MACF Configuration
neighbor_count: 5          # Number of similar users to recruit as UserAgents
history_item_count: 5      # Number of history items to recruit as ItemAgents
retrieval_k: 15            # Number of candidates to retrieve per query
max_rounds: 5              # Maximum discussion rounds
top_k_recommendation: 10   # Final recommendation list size
```

## Module Structure

```
connacf/macf/
├── __init__.py          # Module exports
├── config.py            # MACFConfig dataclass
├── index_manager.py     # GlobalIndexManager for retrieval
├── toolkit.py           # MACFToolkit with retrieval tools
├── data_models.py       # Data structures (ItemSuggestion, AgentResponse, etc.)
├── agents.py            # BaseMACFAgent, UserAgent, ItemAgent
├── orchestrator.py      # MACFOrchestrator coordination logic
├── evaluator.py         # MACFEvaluator for metrics
├── error_handling.py    # Retry logic and fallbacks
├── attack_hooks.py      # Attack framework integration
└── README.md            # This file
```

## Key Components

### GlobalIndexManager
Manages user similarity indices and item embeddings for efficient retrieval:
- `get_user_similarity(user_id)` - Get similarity scores for a user
- `get_item_embedding(item_id)` - Get embedding for an item
- `get_user_history(user_id)` - Get interaction history
- `semantic_search_items(query_embedding, k)` - Search items by embedding

### MACFToolkit
Provides retrieval tools for agents:
- `get_similar_users(target_user_id, n)` - Find similar users
- `get_relevant_items(target_user_id, query, n)` - Find relevant history items
- `retrieve_by_query(query, k)` - Retrieve candidates by query
- `retrieve_by_item(item_id, k)` - Retrieve similar items

### UserAgent
Represents a similar user (neighbor) in discussions:
- Reasons about preference alignment with target user
- Suggests items based on collaborative filtering perspective
- Uses retrieval tools to find candidates

### ItemAgent
Represents a query-relevant history item:
- Traces relevance paths from history item to candidates
- Suggests items based on item-based filtering perspective
- Uses RetrieveByQuery and RetrieveByItem tools

### MACFOrchestrator
Coordinates the multi-round discussion:
1. `recruit_agents()` - Recruit UserAgents and ItemAgents
2. `generate_instructions()` - Create personalized instructions per round
3. `aggregate_responses()` - Merge suggestions into draft list
4. `check_convergence()` - Detect when consensus is reached
5. `run_inference()` - Main entry point for inference

### MACFEvaluator
Computes recommendation metrics:
- `evaluate_single()` - Evaluate single user-query pair
- `compute_hit_at_k()` - Hit rate at K
- `compute_ndcg_at_k()` - NDCG at K
- `aggregate_metrics()` - Compute mean metrics

## Attack Framework Integration

MACF integrates with the existing attack framework via `AttackHooks`:

```python
from connacf.macf import MACFOrchestrator, AttackHooks, MessageInterceptor

# Create custom interceptor
class MyInterceptor(MessageInterceptor):
    def intercept(self, agent_id: str, message: str) -> str:
        # Modify message as needed
        return message

# Set up attack hooks
hooks = AttackHooks()
hooks.register_interceptor(MyInterceptor())

# Create orchestrator with attack hooks
orchestrator = MACFOrchestrator(
    llm=llm,
    toolkit=toolkit,
    config=config,
    attack_hooks=hooks
)

# Run inference (hooks will intercept messages)
result = orchestrator.run_inference(user_id=123, query="action movies")

# Export interaction log for analysis
hooks.export_log("attack_log.json")
```

## Output Format

MACF returns a `RankedList` containing:
- `items`: List of recommended item IDs
- `scores`: Confidence scores for each item
- `rationales`: Explanations for each recommendation
- `discussion_log`: Full discussion history for analysis

## Metrics

MACF reports standard recommendation metrics:
- **Hit@10**: Proportion of users with at least one relevant item in top 10
- **NDCG@10**: Normalized Discounted Cumulative Gain at 10

## Example Usage

```python
from connacf.macf import (
    MACFConfig,
    GlobalIndexManager,
    MACFToolkit,
    MACFOrchestrator,
    MACFEvaluator
)

# Load config
config = MACFConfig.from_yaml('props/MACF.yaml')

# Build index manager
index_manager = GlobalIndexManager(dataset, embedding_model)
index_manager.build_indices()

# Create toolkit and orchestrator
toolkit = MACFToolkit(index_manager, embedding_model)
orchestrator = MACFOrchestrator(llm, toolkit, config)

# Run inference
result = orchestrator.run_inference(
    target_user_id=123,
    query="action movies with good plot"
)

print(f"Recommended items: {result.items}")
print(f"Rationales: {result.rationales}")
```

## Notes

- MACF is implemented as a standalone module that reuses existing ConnaCF infrastructure
- No modifications are made to core ConnaCF files
- All LLM calls use the existing Bedrock wrapper from `connacf/agentverse/llms/`
- Dataset loading reuses existing BPRDataset without modification
