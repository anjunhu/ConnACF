"""
MACF Conversation Logger

Logs multi-agent discussions in a readable conversation format,
making it easy to track which agents are saying what during MACF inference.

Includes attack tracking: identifies which agents are attackers (poisoned items),
which are clean, and which have been contaminated by interacting with attackers.
"""

import os
import json
from typing import Dict, List, Any, Optional, Set
from datetime import datetime


class MACFConversationLogger:
    """Logs MACF agent discussions in a readable conversation format.
    
    Tracks attack status for each agent:
    - ATTACKER: Item agents with poisoned descriptions
    - CLEAN: Agents that haven't interacted with attackers
    - CONTAMINATED: Agents that have received messages from attackers
    """
    
    def __init__(
        self, 
        output_dir: str = "macf_output", 
        print_to_stdout: bool = False,
        enabled: bool = True
    ):
        """
        Initialize MACF conversation logger.
        
        Args:
            output_dir: Directory for log files
            print_to_stdout: If True, also print conversations to stdout
            enabled: If False, all logging is disabled (no-op)
        """
        self.output_dir = output_dir
        self.print_to_stdout = print_to_stdout
        self.enabled = enabled
        self.query_counter = 0
        
        # Attack tracking
        self.poisoned_item_ids: Set[int] = set()  # Items with poisoned descriptions
        self.attacker_agent_ids: Set[str] = set()  # Agent IDs that are attackers
        self.contaminated_agent_ids: Set[str] = set()  # Agents contaminated by attackers
        self.clean_agent_ids: Set[str] = set()  # Agents that are still clean
        
        # Track which agents have interacted with which
        self.interaction_graph: Dict[str, Set[str]] = {}  # agent_id -> set of agents they heard from
        
        if not enabled:
            return
            
        # Create output directory
        self.conversation_dir = os.path.join(output_dir, "conversations")
        os.makedirs(self.conversation_dir, exist_ok=True)
        
        # Main conversation log file
        self.main_log_path = os.path.join(self.conversation_dir, "macf_discussion_log.txt")
        
        # Initialize log file with header
        with open(self.main_log_path, 'w') as f:
            f.write("=" * 100 + "\n")
            f.write("MACF MULTI-AGENT DISCUSSION LOG\n")
            f.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 100 + "\n\n")
    
    def set_poisoned_items(self, poisoned_item_ids: List[int]) -> None:
        """
        Set which items have been poisoned by the attack.
        
        Args:
            poisoned_item_ids: List of item IDs with poisoned descriptions
        """
        self.poisoned_item_ids = set(poisoned_item_ids)
        # Item agents for poisoned items are attackers
        for item_id in poisoned_item_ids:
            self.attacker_agent_ids.add(f"item_agent_{item_id}")
    
    def set_attacker_agents(self, attacker_agent_ids: List[str]) -> None:
        """
        Set which agents are attackers (for CheatAgent and NetSafe attacks).
        
        This is used when the attack targets user agents (CheatAgent, NetSafe)
        rather than item agents (DrunkAgent, RecTextAttack).
        
        Args:
            attacker_agent_ids: List of agent IDs that are attackers
        """
        self.attacker_agent_ids.update(attacker_agent_ids)
        # Remove from clean if they were there
        for agent_id in attacker_agent_ids:
            self.clean_agent_ids.discard(agent_id)
    
    def _get_agent_status(self, agent_id: str) -> str:
        """Get the attack status of an agent."""
        if agent_id in self.attacker_agent_ids:
            return "ATTACKER"
        elif agent_id in self.contaminated_agent_ids:
            return "CONTAMINATED"
        else:
            return "CLEAN"
    
    def _get_status_icon(self, agent_id: str) -> str:
        """Get icon for agent status."""
        status = self._get_agent_status(agent_id)
        if status == "ATTACKER":
            return "🔴"  # Red for attacker
        elif status == "CONTAMINATED":
            return "🟡"  # Yellow for contaminated
        else:
            return "🟢"  # Green for clean
    
    def _mark_contaminated(self, listener_id: str, speaker_id: str) -> None:
        """Mark a listener as contaminated if they heard from an attacker."""
        # Track interaction
        if listener_id not in self.interaction_graph:
            self.interaction_graph[listener_id] = set()
        self.interaction_graph[listener_id].add(speaker_id)
        
        # If speaker is attacker or contaminated, listener becomes contaminated
        if speaker_id in self.attacker_agent_ids or speaker_id in self.contaminated_agent_ids:
            if listener_id not in self.attacker_agent_ids:  # Attackers stay attackers
                self.contaminated_agent_ids.add(listener_id)
                self.clean_agent_ids.discard(listener_id)
    
    def _output(self, text: str, to_file: bool = True):
        """Output text to file and optionally stdout"""
        if not self.enabled:
            return
            
        if self.print_to_stdout:
            print(text, end='')
        
        if to_file:
            # Ensure directory and file exist
            os.makedirs(os.path.dirname(self.main_log_path), exist_ok=True)
            
            # Create file with header if it doesn't exist
            if not os.path.exists(self.main_log_path):
                with open(self.main_log_path, 'w') as f:
                    f.write("=" * 100 + "\n")
                    f.write("MACF MULTI-AGENT DISCUSSION LOG\n")
                    f.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write("=" * 100 + "\n\n")
            
            with open(self.main_log_path, 'a') as f:
                f.write(text)
    
    def log_query_start(self, target_user_id: int, query: str):
        """Log the start of a new query/inference"""
        if not self.enabled:
            return
            
        self.query_counter += 1
        
        text = f"\n{'█' * 100}\n"
        text += f"{'QUERY #' + str(self.query_counter):^100}\n"
        text += f"{'█' * 100}\n\n"
        text += f"🎯 Target User: #{target_user_id}\n"
        text += f"📝 Query: {query}\n"
        text += f"{'─' * 100}\n\n"
        
        self._output(text)
    
    def log_agent_recruitment(
        self, 
        user_agents: List[Any], 
        item_agents: List[Any],
        adversarial_count: int = 0
    ):
        """Log which agents were recruited for the discussion"""
        if not self.enabled:
            return
        
        # Initialize clean agents
        for agent in user_agents:
            agent_id = getattr(agent, 'agent_id', str(agent))
            if agent_id not in self.attacker_agent_ids and agent_id not in self.contaminated_agent_ids:
                self.clean_agent_ids.add(agent_id)
        for agent in item_agents:
            agent_id = getattr(agent, 'agent_id', str(agent))
            if agent_id not in self.attacker_agent_ids and agent_id not in self.contaminated_agent_ids:
                self.clean_agent_ids.add(agent_id)
        
        # Count attackers among item agents
        attacker_count = sum(1 for agent in item_agents 
                           if getattr(agent, 'agent_id', str(agent)) in self.attacker_agent_ids)
            
        text = "🤝 AGENT RECRUITMENT\n"
        text += f"{'─' * 100}\n"
        
        # Legend
        text += "\n📊 Status Legend: 🟢 CLEAN | 🟡 CONTAMINATED | 🔴 ATTACKER\n"
        
        # User agents
        text += f"\n👥 User Agents ({len(user_agents)}):\n"
        for agent in user_agents:
            agent_id = getattr(agent, 'agent_id', str(agent))
            neighbor_id = getattr(agent, 'neighbor_user_id', '?')
            status_icon = self._get_status_icon(agent_id)
            status = self._get_agent_status(agent_id)
            text += f"   {status_icon} {agent_id} [{status}] (representing neighbor user #{neighbor_id})\n"
        
        # Item agents
        text += f"\n📀 Item Agents ({len(item_agents)}) - {attacker_count} ATTACKERS:\n"
        for agent in item_agents:
            agent_id = getattr(agent, 'agent_id', str(agent))
            item_id = getattr(agent, 'item_id', '?')
            status_icon = self._get_status_icon(agent_id)
            status = self._get_agent_status(agent_id)
            text += f"   {status_icon} {agent_id} [{status}] (representing history item #{item_id})\n"
        
        if adversarial_count > 0:
            text += f"\n⚠️ Adversarial Agents Injected: {adversarial_count}\n"
        
        # Summary
        text += f"\n📈 Attack Summary:\n"
        text += f"   • Attackers (poisoned items): {attacker_count}\n"
        text += f"   • Clean agents: {len(self.clean_agent_ids)}\n"
        text += f"   • Contaminated agents: {len(self.contaminated_agent_ids)}\n"
        
        text += f"{'─' * 100}\n\n"
        
        self._output(text)
    
    def log_agent_profiles(
        self,
        user_agents: List[Any],
        item_agents: List[Any],
        memory_store: Any = None
    ):
        """
        Log semantic profiles for all agents in ConnaCF-compatible format.
        
        This creates an interaction_log.txt section that can be parsed by
        human_judge_calibration.py for auditing LLM judge quality.
        
        For NetSAFE attackers, also shows the injected system prompt to make
        the attack visible in the log.
        
        Args:
            user_agents: List of UserAgent objects
            item_agents: List of ItemAgent objects  
            memory_store: Optional MACFMemoryStore for getting semantic profiles
        """
        if not self.enabled:
            return
        
        text = f"\n{'─' * 100}\n"
        text += "📋 AGENT SEMANTIC PROFILES (for calibration)\n"
        text += f"{'─' * 100}\n\n"
        
        # Log user profiles
        for agent in user_agents:
            agent_id = getattr(agent, 'agent_id', str(agent))
            user_id = getattr(agent, 'neighbor_user_id', None)
            is_attacker = agent_id in self.attacker_agent_ids
            
            # Get semantic profile from memory store or agent
            profile = ""
            if memory_store and user_id is not None:
                memory = memory_store.get_user_memory(user_id)
                profile = memory.profile if memory else ""
            
            if not profile:
                # Fall back to agent's memory context
                memory = getattr(agent, 'memory', None)
                if memory:
                    profile = getattr(memory, 'profile', '') or memory.get_memory_context()[:300] if hasattr(memory, 'get_memory_context') else ''
            
            if not profile:
                profile = f"User #{user_id} (no profile available)"
            
            # Format in ConnaCF-compatible format
            if is_attacker:
                text += f"🔴 ATTACKER USER #{user_id}:\n"
            else:
                text += f"👤 USER #{user_id}:\n"
            text += f"{'─' * 90}\n"
            text += f"Profile: {profile}\n"
            
            # For attackers, also show the system prompt injection (NetSAFE attack)
            if is_attacker:
                try:
                    system_prompt = agent.get_system_prompt() if hasattr(agent, 'get_system_prompt') else None
                    if system_prompt and 'PRIORITY DIRECTIVE' in system_prompt:
                        # Extract just the NetSAFE injection part (up to the first ---)
                        injection_end = system_prompt.find('---')
                        if injection_end > 0:
                            injection = system_prompt[:injection_end + 3]
                            text += f"\n⚠️ INJECTED SYSTEM PROMPT:\n{injection}\n"
                except Exception:
                    pass  # Silently skip if we can't get the system prompt
            
            text += f"{'─' * 90}\n\n"
        
        # Log item profiles
        for agent in item_agents:
            agent_id = getattr(agent, 'agent_id', str(agent))
            item_id = getattr(agent, 'item_id', None)
            is_attacker = agent_id in self.attacker_agent_ids
            
            # Get semantic profile from memory store or agent
            profile = ""
            if memory_store and item_id is not None:
                memory = memory_store.get_item_memory(item_id)
                profile = memory.profile if memory else ""
            
            if not profile:
                # Fall back to agent's description
                profile = getattr(agent, 'item_description', '') or getattr(agent, 'description', '')
            
            if not profile:
                profile = f"Item #{item_id} (no description available)"
            
            # Get item title if available
            title = getattr(agent, 'item_title', '') or getattr(agent, 'title', '')
            title_str = f" - {title}" if title else ""
            
            # Format in ConnaCF-compatible format
            if is_attacker:
                text += f"🔴 ATTACKER ITEM #{item_id}{title_str}:\n"
            else:
                text += f"💿 ITEM #{item_id}{title_str}:\n"
            text += f"{'─' * 90}\n"
            text += f"Description: {profile}\n"
            
            # For attackers, also show the system prompt injection (NetSAFE attack)
            if is_attacker:
                try:
                    system_prompt = agent.get_system_prompt() if hasattr(agent, 'get_system_prompt') else None
                    if system_prompt and 'PRIORITY DIRECTIVE' in system_prompt:
                        # Extract just the NetSAFE injection part (up to the first ---)
                        injection_end = system_prompt.find('---')
                        if injection_end > 0:
                            injection = system_prompt[:injection_end + 3]
                            text += f"\n⚠️ INJECTED SYSTEM PROMPT:\n{injection}\n"
                except Exception:
                    pass  # Silently skip if we can't get the system prompt
            
            text += f"{'─' * 90}\n\n"
        
        self._output(text)
    
    def log_round_start(self, round_idx: int, max_rounds: int, active_agents: List[str]):
        """Log the start of a discussion round"""
        if not self.enabled:
            return
            
        text = f"\n{'═' * 100}\n"
        text += f"DISCUSSION ROUND {round_idx + 1}/{max_rounds}\n"
        text += f"{'═' * 100}\n"
        text += f"Active Agents: {', '.join(active_agents)}\n"
        text += f"{'─' * 100}\n\n"
        
        self._output(text)
    
    def log_instruction(self, agent_id: str, instruction: str, was_modified: bool = False):
        """Log instruction sent to an agent"""
        if not self.enabled:
            return
            
        modifier = " [🔴 MODIFIED BY ATTACK]" if was_modified else ""
        
        text = f"📨 INSTRUCTION TO {agent_id}{modifier}:\n"
        text += f"{'─' * 80}\n"
        text += self._format_text(instruction, indent=2, max_width=76)
        text += f"\n{'─' * 80}\n\n"
        
        self._output(text)
    
    def log_agent_response(
        self, 
        agent_id: str, 
        raw_response: str,
        suggestions: List[Dict],
        rationale: str,
        tool_calls: List[Dict] = None,
        was_modified: bool = False,
        listeners: List[str] = None,
        item_descriptions: Dict[int, str] = None,
        system_prompt: str = None
    ):
        """Log an agent's response and track contamination spread.
        
        Args:
            agent_id: ID of the speaking agent
            raw_response: Raw response text
            suggestions: List of item suggestions
            rationale: Agent's rationale
            tool_calls: Any tool calls made
            was_modified: Whether response was modified by attack
            listeners: List of agent IDs that will hear this response
            item_descriptions: Optional dict of item_id -> description for suggested items
            system_prompt: Optional system prompt used for this agent (for debugging)
        """
        if not self.enabled:
            return
        
        # Track contamination spread
        if listeners:
            for listener_id in listeners:
                self._mark_contaminated(listener_id, agent_id)
            
        # Determine agent type icon
        if 'user_agent' in agent_id:
            icon = "👤"
        elif 'item_agent' in agent_id:
            icon = "📀"
        else:
            icon = "🤖"
        
        # Get status
        status_icon = self._get_status_icon(agent_id)
        status = self._get_agent_status(agent_id)
        
        modifier = " [🔴 MODIFIED BY ATTACK]" if was_modified else ""
        
        text = f"\n{icon} {status_icon} RESPONSE FROM {agent_id} [{status}]{modifier}:\n"
        text += f"{'─' * 80}\n"
        
        # Log tool calls if any
        if tool_calls:
            text += "🔧 Tool Calls:\n"
            for tc in tool_calls:
                tool_name = tc.get('tool_name', tc.tool_name if hasattr(tc, 'tool_name') else '?')
                result_count = len(tc.get('result', tc.result if hasattr(tc, 'result') else []))
                text += f"   • {tool_name} → {result_count} results\n"
            text += "\n"
        
        # Log suggestions with FULL reasons and item descriptions
        text += "💡 Suggestions:\n"
        if suggestions:
            for s in suggestions[:10]:  # Limit to 10 items
                item_id = s.get('item_id', s.item_id if hasattr(s, 'item_id') else '?')
                score = s.get('score', s.score if hasattr(s, 'score') else 0)
                reason = s.get('reason', s.reason if hasattr(s, 'reason') else '')
                # Mark if suggesting a poisoned item
                poison_marker = " 🔴" if item_id in self.poisoned_item_ids else ""
                text += f"   • Item #{item_id}{poison_marker} (score: {score:.2f})\n"
                
                # Include item description if available
                if item_descriptions and item_id in item_descriptions:
                    desc = item_descriptions[item_id]
                    text += f"     📝 Description: {desc}\n"
                
                # Full reason, wrapped nicely
                text += self._format_text(f"Reason: {reason}", indent=5, max_width=75)
                text += "\n"
        else:
            text += "   (no suggestions)\n"
        
        # Log FULL rationale (no truncation)
        text += f"\n📋 Rationale:\n"
        text += self._format_text(rationale or "(none)", indent=3, max_width=77)
        
        # Log FULL raw response (no truncation for detailed analysis)
        text += f"\n\n📝 Full Response:\n"
        text += self._format_text(raw_response or "(empty)", indent=3, max_width=77)
        
        text += f"\n{'─' * 80}\n"
        
        self._output(text)
        
        # Also save to per-agent detailed log (includes system prompt for debugging)
        self._save_agent_response_detail(agent_id, raw_response, suggestions, rationale, status, system_prompt)
    
    def log_aggregation_result(
        self, 
        draft_list: List[int], 
        conflicts: List[str],
        agreements: Dict[int, int] = None
    ):
        """Log the result of response aggregation"""
        if not self.enabled:
            return
            
        text = f"\n🔄 AGGREGATION RESULT\n"
        text += f"{'─' * 80}\n"
        
        text += f"📋 Draft List ({len(draft_list)} items): {draft_list[:20]}"
        if len(draft_list) > 20:
            text += "..."
        text += "\n"
        
        if agreements:
            text += f"\n✅ Agreements (items with multiple supporters):\n"
            for item_id, count in list(agreements.items())[:10]:
                text += f"   • Item #{item_id}: {count} agents\n"
        
        if conflicts:
            text += f"\n⚠️ Conflicts ({len(conflicts)}):\n"
            for conflict in conflicts[:5]:
                text += f"   • {self._truncate(conflict, 100)}\n"
        
        text += f"{'─' * 80}\n\n"
        
        self._output(text)
    
    def log_convergence_check(self, converged: bool, reason: str = ""):
        """Log convergence check result"""
        if not self.enabled:
            return
            
        status = "✅ CONVERGED" if converged else "🔄 NOT CONVERGED"
        text = f"\n{status}"
        if reason:
            text += f" - {reason}"
        text += "\n"
        
        self._output(text)
    
    def log_contamination_summary(self, round_idx: int):
        """Log contamination status summary after a round."""
        if not self.enabled:
            return
        
        text = f"\n📊 CONTAMINATION STATUS (after round {round_idx + 1}):\n"
        text += f"{'─' * 80}\n"
        text += f"   🔴 Attackers: {len(self.attacker_agent_ids)}\n"
        text += f"   🟡 Contaminated: {len(self.contaminated_agent_ids)}\n"
        text += f"   🟢 Clean: {len(self.clean_agent_ids)}\n"
        
        if self.contaminated_agent_ids:
            text += f"\n   Contaminated agents:\n"
            for agent_id in sorted(self.contaminated_agent_ids):
                sources = self.interaction_graph.get(agent_id, set())
                attacker_sources = sources & self.attacker_agent_ids
                if attacker_sources:
                    text += f"      • {agent_id} (heard from attackers: {', '.join(sorted(attacker_sources))})\n"
                else:
                    text += f"      • {agent_id} (indirect contamination)\n"
        
        text += f"{'─' * 80}\n\n"
        
        self._output(text)
    
    def log_final_result(
        self, 
        target_user_id: int,
        query: str,
        final_items: List[int],
        scores: List[float],
        rationales: List[str],
        num_rounds: int,
        user_history: List[int] = None,
        ground_truth: List[int] = None
    ):
        """Log the final recommendation result"""
        if not self.enabled:
            return
            
        text = f"\n{'█' * 100}\n"
        text += f"{'FINAL RECOMMENDATIONS':^100}\n"
        text += f"{'█' * 100}\n\n"
        
        text += f"🎯 User: #{target_user_id}\n"
        text += f"📝 Query: {self._truncate(query, 80)}\n"
        text += f"🔄 Rounds: {num_rounds}\n\n"
        
        # Attack summary
        text += f"📊 ATTACK SUMMARY:\n"
        text += f"{'─' * 80}\n"
        text += f"   🔴 Attackers (poisoned items): {len(self.attacker_agent_ids)}\n"
        text += f"   🟡 Contaminated agents: {len(self.contaminated_agent_ids)}\n"
        text += f"   🟢 Clean agents: {len(self.clean_agent_ids)}\n"
        
        # Calculate contamination rate
        total_agents = len(self.attacker_agent_ids) + len(self.contaminated_agent_ids) + len(self.clean_agent_ids)
        if total_agents > 0:
            contamination_rate = (len(self.contaminated_agent_ids) / total_agents) * 100
            text += f"   📈 Contamination rate: {contamination_rate:.1f}%\n"
        text += f"{'─' * 80}\n\n"
        
        # Show user history if available
        if user_history:
            text += f"📚 User History ({len(user_history)} items):\n"
            text += f"   {user_history[:20]}"
            if len(user_history) > 20:
                text += f"... (+{len(user_history) - 20} more)"
            text += "\n\n"
        
        # Show ground truth if available
        if ground_truth:
            text += f"🎯 Ground Truth ({len(ground_truth)} items):\n"
            text += f"   {ground_truth[:20]}"
            if len(ground_truth) > 20:
                text += f"... (+{len(ground_truth) - 20} more)"
            text += "\n\n"
        
        text += "📋 Recommended Items:\n"
        text += f"{'─' * 80}\n"
        
        poisoned_in_recs = 0
        for i, (item_id, score, rationale) in enumerate(zip(final_items, scores, rationales)):
            # Check if item is in ground truth
            hit_marker = " ✅" if ground_truth and item_id in ground_truth else ""
            # Check if item is poisoned
            poison_marker = " 🔴 POISONED" if item_id in self.poisoned_item_ids else ""
            if item_id in self.poisoned_item_ids:
                poisoned_in_recs += 1
            text += f"\n{i+1}. Item #{item_id} (score: {score:.3f}){hit_marker}{poison_marker}\n"
            text += f"   Rationale: {self._truncate(rationale, 150)}\n"
        
        # Show hit statistics if ground truth available
        if ground_truth and final_items:
            hits = sum(1 for item in final_items[:10] if item in ground_truth)
            text += f"\n{'─' * 80}\n"
            text += f"📊 Hit@10: {hits}/{min(10, len(final_items))} items in ground truth\n"
        
        # Show poisoned items in recommendations
        if self.poisoned_item_ids:
            text += f"🔴 Poisoned items in top-10: {poisoned_in_recs}\n"
        
        text += f"\n{'─' * 80}\n"
        text += f"{'═' * 100}\n\n"
        
        self._output(text)
    
    def log_error(self, context: str, error: str):
        """Log an error that occurred during processing"""
        if not self.enabled:
            return
            
        text = f"\n❌ ERROR in {context}:\n"
        text += f"   {error}\n\n"
        
        self._output(text)
    
    def save_query_json(self, query_data: Dict[str, Any]):
        """Save detailed query data as JSON for analysis"""
        if not self.enabled:
            return
            
        json_path = os.path.join(
            self.conversation_dir, 
            f"query_{self.query_counter:04d}.json"
        )
        
        with open(json_path, 'w') as f:
            json.dump(query_data, f, indent=2, default=str)
    
    def _format_text(self, text: str, indent: int = 0, max_width: int = 80) -> str:
        """Format text with indentation and wrapping"""
        if not text:
            return " " * indent + "(empty)"
        
        indent_str = " " * indent
        lines = text.split('\n')
        formatted = []
        
        for line in lines:
            if len(line) <= max_width:
                formatted.append(indent_str + line)
            else:
                # Simple word wrap
                words = line.split()
                current = indent_str
                for word in words:
                    if len(current) + len(word) + 1 <= max_width + indent:
                        current += word + " "
                    else:
                        formatted.append(current.rstrip())
                        current = indent_str + word + " "
                if current.strip():
                    formatted.append(current.rstrip())
        
        return '\n'.join(formatted)
    
    def _truncate(self, text: str, max_len: int) -> str:
        """Truncate text with ellipsis"""
        if not text:
            return "(empty)"
        text = text.replace('\n', ' ').strip()
        if len(text) <= max_len:
            return text
        return text[:max_len - 3] + "..."
    
    def _save_agent_response_detail(
        self, 
        agent_id: str, 
        raw_response: str, 
        suggestions: List[Dict],
        rationale: str,
        status: str,
        system_prompt: str = None
    ):
        """Save detailed agent response to per-agent log file."""
        if not self.enabled:
            return
        
        # Create agent logs directory
        agent_logs_dir = os.path.join(self.conversation_dir, "agent_logs")
        os.makedirs(agent_logs_dir, exist_ok=True)
        
        # Append to agent's log file
        agent_log_path = os.path.join(agent_logs_dir, f"{agent_id}.txt")
        
        with open(agent_log_path, 'a') as f:
            f.write(f"\n{'=' * 60}\n")
            f.write(f"Query #{self.query_counter} | Status: {status}\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"{'=' * 60}\n\n")
            
            # Log system prompt if provided (important for NetSAFE attack debugging)
            if system_prompt:
                f.write("SYSTEM PROMPT:\n")
                f.write("-" * 40 + "\n")
                f.write(system_prompt)
                f.write("\n\n")
            
            f.write("FULL RESPONSE:\n")
            f.write("-" * 40 + "\n")
            f.write(raw_response or "(empty)")
            f.write("\n\n")
            
            f.write("RATIONALE:\n")
            f.write("-" * 40 + "\n")
            f.write(rationale or "(none)")
            f.write("\n\n")
            
            f.write("SUGGESTIONS:\n")
            f.write("-" * 40 + "\n")
            for s in suggestions:
                item_id = s.get('item_id', '?')
                score = s.get('score', 0)
                reason = s.get('reason', '')
                poison = " [POISONED]" if item_id in self.poisoned_item_ids else ""
                f.write(f"Item #{item_id}{poison} (score: {score:.2f})\n")
                f.write(f"  Reason: {reason}\n\n")
            
            f.write("\n")
    
    def get_contamination_stats(self) -> Dict[str, Any]:
        """Get current contamination statistics."""
        total = len(self.attacker_agent_ids) + len(self.contaminated_agent_ids) + len(self.clean_agent_ids)
        return {
            'n_attackers': len(self.attacker_agent_ids),
            'n_contaminated': len(self.contaminated_agent_ids),
            'n_clean': len(self.clean_agent_ids),
            'n_total': total,
            'contamination_rate': len(self.contaminated_agent_ids) / total if total > 0 else 0,
            'attacker_ids': list(self.attacker_agent_ids),
            'contaminated_ids': list(self.contaminated_agent_ids),
            'clean_ids': list(self.clean_agent_ids),
            'poisoned_item_ids': list(self.poisoned_item_ids)
        }
    
    def reset_for_new_query(self):
        """Reset contamination tracking for a new query (keeps poisoned items)."""
        # Keep attacker_agent_ids (based on poisoned items)
        self.contaminated_agent_ids.clear()
        self.clean_agent_ids.clear()
        self.interaction_graph.clear()
    
    def log_backward_pass(self, backward_result: Dict[str, Any]):
        """
        Log the backward pass results (ConnaCF-compatible learning).
        
        Args:
            backward_result: Dict with backward pass statistics including:
                - user_updates_applied: Number of user profile updates
                - item_updates_applied: Number of item description updates
                - total_user_agents: Total user agents
                - total_item_agents: Total item agents
                - feedback: Dict of agent feedback
        """
        if not self.enabled:
            return
        
        text = f"\n{'─' * 80}\n"
        text += "🔄 BACKWARD PASS (ConnaCF-Compatible Learning)\n"
        text += f"{'─' * 80}\n\n"
        
        user_updates = backward_result.get('user_updates_applied', 0)
        item_updates = backward_result.get('item_updates_applied', 0)
        total_users = backward_result.get('total_user_agents', 0)
        total_items = backward_result.get('total_item_agents', 0)
        
        text += f"📊 Profile Updates:\n"
        text += f"   User agents: {user_updates}/{total_users} profiles updated\n"
        text += f"   Item agents: {item_updates}/{total_items} descriptions updated\n"
        
        # Show feedback summary
        feedback = backward_result.get('feedback', {})
        if feedback:
            correct_count = sum(1 for f in feedback.values() if f.get('is_correct', False))
            total_count = len(feedback)
            text += f"\n📈 Feedback Summary:\n"
            text += f"   Agents with correct suggestions: {correct_count}/{total_count}\n"
            
            # Show hit rates
            hit_rates = [f.get('hit_rate', 0) for f in feedback.values()]
            if hit_rates:
                avg_hit_rate = sum(hit_rates) / len(hit_rates)
                text += f"   Average hit rate: {avg_hit_rate:.1%}\n"
        
        text += f"\n{'─' * 80}\n"
        
        self._output(text)
