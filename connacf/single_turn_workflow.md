# ConnaCF Single-Turn Workflow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     ONE ROUND in ConnaCF                                │
│                                                                         │
│  Input: batch of (user_i, pos_item_i, neg_item_i)                       │
│                                                                         │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │  FORWARD                                                          │  │
│  │                                                                   │  │
│  │  [optional] U-U Interaction                                       │  │
│  │    user_i ──asks k similar users──► opinions on candidates        │  │
│  │                └──► synthesis ──► enriched_context_i              │  │
│  │                                                                   │  │
│  │  [optional] U-I Dialogue                                          │  │
│  │    user_i ◄──► pos_item_i  ──►  dialogue_summary (sentiment)      │  │
│  │    user_i ◄──► neg_item_i  ──►  dialogue_summary (sentiment)      │  │
│  │                                                                   │  │
│  │  Rec Agent (LLM)                                                  │  │
│  │    prompt = user_desc + pos_desc + neg_desc                       │  │
│  │    LLM ──► selection (pos or neg?) + reason                       │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                              │                                          │
│                       accuracy check                                    │
│                       /             \                                   │
│                   wrong             correct                             │
│                     │                 │                                 │
│  ┌──────────────────▼──────────┐  ┌───▼────────────────────────────┐   │
│  │  BACKWARD (wrong)           │  │  BACKWARD_TRUE (correct)       │   │
│  │                             │  │  reinforce existing memory     │   │
│  │  user_agent ◄── reason      │  └────────────────────────────────┘   │
│  │    + items ──► new user     │                                        │
│  │      memory entry           │                                        │
│  │                             │                                        │
│  │  item_agent ◄── reason      │                                        │
│  │    + updated user profile   │                                        │
│  │    ──► new item memory entry│                                        │
│  └─────────────────────────────┘                                        │
│                                                                         │
│  After all rounds:                                                      │
│    snapshot user memory    ──►  memory_1                                │
│    store item embeddings   ──►  memory_embedding  (for RAG)             │
└─────────────────────────────────────────────────────────────────────────┘
```
