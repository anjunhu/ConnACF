"""
MACF Communication Graph Tracker

Tracks and logs communication edges between user and item agents at each turn.
Provides both JSON/CSV logging and visualization tools for the communication matrix.

Output structure matches metrics_collector:
- task_*/turn_*.json for per-turn communication snapshots
- communication_graph/communication_edges.csv for cumulative edges
- communication_graph/communication_summary.json for final summary
"""

import json
import os
import csv
import logging
from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class CommunicationEdge:
    """A single communication edge between agents."""
    source_agent_id: str  # e.g., "user_agent_42"
    target_agent_id: str  # e.g., "item_agent_15"
    source_type: str  # "user" or "item"
    target_type: str  # "user" or "item"
    global_turn: int
    task_id: int
    round_idx: int
    message_type: str  # "suggestion", "response", "broadcast"
    content_summary: str = ""  # Brief summary of what was communicated
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class TurnCommunicationSnapshot:
    """Snapshot of all communication at a specific turn."""
    global_turn: int
    task_id: int
    round_idx: int
    timestamp: str
    
    # Active agents this turn
    active_user_agents: List[str] = field(default_factory=list)
    active_item_agents: List[str] = field(default_factory=list)
    
    # Communication edges this turn
    edges: List[CommunicationEdge] = field(default_factory=list)
    
    # Adjacency info for matrix visualization
    # user_to_item_edges[user_id] = [item_ids they communicated with]
    user_to_item_edges: Dict[str, List[str]] = field(default_factory=dict)
    item_to_user_edges: Dict[str, List[str]] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            'global_turn': self.global_turn,
            'task_id': self.task_id,
            'round_idx': self.round_idx,
            'timestamp': self.timestamp,
            'active_user_agents': self.active_user_agents,
            'active_item_agents': self.active_item_agents,
            'edges': [e.to_dict() for e in self.edges],
            'user_to_item_edges': self.user_to_item_edges,
            'item_to_user_edges': self.item_to_user_edges,
            'num_edges': len(self.edges),
            'num_user_agents': len(self.active_user_agents),
            'num_item_agents': len(self.active_item_agents)
        }


class CommunicationGraphTracker:
    """
    Tracks communication graph between agents across turns.
    
    Logs:
    - Per-turn communication snapshots in task_*/turn_*_comm.json (alongside metrics)
    - Cumulative edge list (CSV)
    - Adjacency matrices for visualization
    """
    
    def __init__(
        self,
        output_dir: str,
        experiment_name: str = "macf_experiment"
    ):
        """
        Initialize the communication graph tracker.
        
        Args:
            output_dir: Directory to save communication logs
            experiment_name: Name for this experiment
        """
        self.output_dir = Path(output_dir)
        self.experiment_name = experiment_name
        
        # Create output directories
        self.graph_dir = self.output_dir / "communication_graph"
        self.graph_dir.mkdir(parents=True, exist_ok=True)
        
        # Storage
        self.turn_snapshots: List[TurnCommunicationSnapshot] = []
        self.all_edges: List[CommunicationEdge] = []
        
        # Track all unique agents seen
        self.all_user_agents: Set[str] = set()
        self.all_item_agents: Set[str] = set()
        
        # Current turn tracking
        self.current_snapshot: Optional[TurnCommunicationSnapshot] = None
        
        # Initialize CSV file
        self._init_csv()
        
        logger.info(f"CommunicationGraphTracker initialized: {self.graph_dir}")
    
    def _init_csv(self):
        """Initialize the CSV file for edge logging."""
        self.csv_path = self.graph_dir / "communication_edges.csv"
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'global_turn', 'task_id', 'round_idx', 'timestamp',
                'source_agent_id', 'target_agent_id',
                'source_type', 'target_type',
                'message_type', 'content_summary'
            ])
    
    def start_turn(
        self,
        global_turn: int,
        task_id: int,
        round_idx: int,
        active_user_agents: List[str],
        active_item_agents: List[str]
    ):
        """
        Start tracking a new turn.
        
        Args:
            global_turn: Global turn counter
            task_id: Current task ID
            round_idx: Round within task
            active_user_agents: List of active user agent IDs
            active_item_agents: List of active item agent IDs
        """
        # Save previous snapshot if exists
        if self.current_snapshot:
            self._save_turn_snapshot()
        
        # Create new snapshot
        self.current_snapshot = TurnCommunicationSnapshot(
            global_turn=global_turn,
            task_id=task_id,
            round_idx=round_idx,
            timestamp=datetime.now().isoformat(),
            active_user_agents=list(active_user_agents),
            active_item_agents=list(active_item_agents)
        )
        
        # Track all agents
        self.all_user_agents.update(active_user_agents)
        self.all_item_agents.update(active_item_agents)
        
        logger.debug(
            f"Started turn {global_turn}: {len(active_user_agents)} users, "
            f"{len(active_item_agents)} items"
        )
    
    def record_broadcast(
        self,
        speaker_agent_id: str,
        listener_agent_ids: List[str],
        content_summary: str = ""
    ):
        """
        Record a broadcast communication (one agent speaks, all others listen).
        
        In MACF, when an agent responds, all other active agents "hear" it.
        This creates edges from the speaker to all listeners.
        
        Args:
            speaker_agent_id: The agent that spoke
            listener_agent_ids: All agents that heard the message
            content_summary: Brief summary of the message
        """
        if not self.current_snapshot:
            logger.warning("record_broadcast called without active turn")
            return
        
        # Determine speaker type
        speaker_type = "user" if "user" in speaker_agent_id.lower() else "item"
        
        for listener_id in listener_agent_ids:
            listener_type = "user" if "user" in listener_id.lower() else "item"
            
            edge = CommunicationEdge(
                source_agent_id=speaker_agent_id,
                target_agent_id=listener_id,
                source_type=speaker_type,
                target_type=listener_type,
                global_turn=self.current_snapshot.global_turn,
                task_id=self.current_snapshot.task_id,
                round_idx=self.current_snapshot.round_idx,
                message_type="broadcast",
                content_summary=content_summary[:200] if content_summary else ""
            )
            
            self.current_snapshot.edges.append(edge)
            self.all_edges.append(edge)
            
            # Update adjacency tracking
            if speaker_type == "user" and listener_type == "item":
                if speaker_agent_id not in self.current_snapshot.user_to_item_edges:
                    self.current_snapshot.user_to_item_edges[speaker_agent_id] = []
                self.current_snapshot.user_to_item_edges[speaker_agent_id].append(listener_id)
            elif speaker_type == "item" and listener_type == "user":
                if speaker_agent_id not in self.current_snapshot.item_to_user_edges:
                    self.current_snapshot.item_to_user_edges[speaker_agent_id] = []
                self.current_snapshot.item_to_user_edges[speaker_agent_id].append(listener_id)
            
            # Write to CSV immediately
            self._write_edge_to_csv(edge)
    
    def record_suggestion(
        self,
        agent_id: str,
        suggested_item_ids: List[int],
        content_summary: str = ""
    ):
        """
        Record item suggestions from an agent.
        
        Creates edges from the suggesting agent to item agents for suggested items.
        
        Args:
            agent_id: The agent making suggestions
            suggested_item_ids: List of suggested item IDs
            content_summary: Brief summary of the suggestion rationale
        """
        if not self.current_snapshot:
            return
        
        agent_type = "user" if "user" in agent_id.lower() else "item"
        
        for item_id in suggested_item_ids:
            target_id = f"item_agent_{item_id}"
            
            edge = CommunicationEdge(
                source_agent_id=agent_id,
                target_agent_id=target_id,
                source_type=agent_type,
                target_type="item",
                global_turn=self.current_snapshot.global_turn,
                task_id=self.current_snapshot.task_id,
                round_idx=self.current_snapshot.round_idx,
                message_type="suggestion",
                content_summary=content_summary[:200] if content_summary else ""
            )
            
            self.current_snapshot.edges.append(edge)
            self.all_edges.append(edge)
            self._write_edge_to_csv(edge)
    
    def _write_edge_to_csv(self, edge: CommunicationEdge):
        """Write a single edge to the CSV file."""
        try:
            with open(self.csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    edge.global_turn, edge.task_id, edge.round_idx,
                    datetime.now().isoformat(),
                    edge.source_agent_id, edge.target_agent_id,
                    edge.source_type, edge.target_type,
                    edge.message_type, edge.content_summary
                ])
        except Exception as e:
            logger.warning(f"Failed to write edge to CSV: {e}")
    
    def _save_turn_snapshot(self):
        """Save the current turn snapshot to JSON in task_*/turn_*_comm.json format."""
        if not self.current_snapshot:
            return
        
        self.turn_snapshots.append(self.current_snapshot)
        
        # Save to task_*/turn_*_comm.json (matching metrics_collector structure)
        task_dir = self.output_dir / f"task_{self.current_snapshot.task_id}"
        task_dir.mkdir(parents=True, exist_ok=True)
        
        turn_file = task_dir / f"turn_{self.current_snapshot.round_idx}_comm.json"
        try:
            with open(turn_file, 'w') as f:
                json.dump(self.current_snapshot.to_dict(), f, indent=2)
            logger.debug(f"Saved communication snapshot to {turn_file}")
        except Exception as e:
            logger.warning(f"Failed to save turn snapshot: {e}")
    
    def end_turn(self):
        """End the current turn and save snapshot."""
        if self.current_snapshot:
            self._save_turn_snapshot()
            logger.debug(
                f"Turn {self.current_snapshot.global_turn} ended: "
                f"{len(self.current_snapshot.edges)} edges"
            )
            self.current_snapshot = None
    
    def finalize(self):
        """Finalize tracking and save summary."""
        # Save any pending snapshot
        if self.current_snapshot:
            self._save_turn_snapshot()
            self.current_snapshot = None
        
        # Save summary JSON
        summary = {
            'experiment_name': self.experiment_name,
            'total_turns': len(self.turn_snapshots),
            'total_edges': len(self.all_edges),
            'unique_user_agents': len(self.all_user_agents),
            'unique_item_agents': len(self.all_item_agents),
            'all_user_agents': sorted(list(self.all_user_agents)),
            'all_item_agents': sorted(list(self.all_item_agents)),
            'timestamp': datetime.now().isoformat()
        }
        
        summary_file = self.graph_dir / "communication_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(
            f"Communication graph finalized: {len(self.turn_snapshots)} turns, "
            f"{len(self.all_edges)} edges"
        )
    
    def get_adjacency_matrix_data(self, global_turn: Optional[int] = None) -> Dict:
        """
        Get adjacency matrix data for visualization.
        
        Args:
            global_turn: If specified, get matrix for that turn only.
                        If None, get cumulative matrix up to current turn.
        
        Returns:
            Dict with matrix data for visualization
        """
        # Collect edges
        if global_turn is not None:
            edges = [e for e in self.all_edges if e.global_turn == global_turn]
        else:
            edges = self.all_edges
        
        # Build adjacency data
        user_agents = sorted(list(self.all_user_agents))
        item_agents = sorted(list(self.all_item_agents))
        
        # Create user-to-item matrix
        user_to_item = {}
        for edge in edges:
            if edge.source_type == "user" and edge.target_type == "item":
                key = (edge.source_agent_id, edge.target_agent_id)
                user_to_item[key] = user_to_item.get(key, 0) + 1
        
        # Create item-to-user matrix
        item_to_user = {}
        for edge in edges:
            if edge.source_type == "item" and edge.target_type == "user":
                key = (edge.source_agent_id, edge.target_agent_id)
                item_to_user[key] = item_to_user.get(key, 0) + 1
        
        return {
            'user_agents': user_agents,
            'item_agents': item_agents,
            'user_to_item_edges': user_to_item,
            'item_to_user_edges': item_to_user,
            'global_turn': global_turn,
            'total_edges': len(edges)
        }
