"""
Unit tests for MACF attack framework integration.

Tests verify:
- AttackHooks class functionality
- Message interception in orchestrator
- Adversarial agent injection
- Interaction logging
- System continues execution when attacks modify behavior

Requirements: 11.2, 11.4, 11.5, 11.6
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from typing import List, Dict, Any

from connacf.macf.attack_hooks import AttackHooks, MessageInterceptor
from connacf.macf.data_models import AgentResponse, ItemSuggestion, RankedList


class MockInterceptor(MessageInterceptor):
    """Mock interceptor for testing."""
    
    def __init__(self, prefix: str = "INTERCEPTED: "):
        self.prefix = prefix
        self.intercepted_messages: List[tuple] = []
    
    def intercept(self, agent_id: str, message: str) -> str:
        self.intercepted_messages.append((agent_id, message))
        return f"{self.prefix}{message}"


class UppercaseInterceptor(MessageInterceptor):
    """Interceptor that converts messages to uppercase."""
    
    def intercept(self, agent_id: str, message: str) -> str:
        return message.upper()


class FailingInterceptor(MessageInterceptor):
    """Interceptor that always raises an exception."""
    
    def intercept(self, agent_id: str, message: str) -> str:
        raise RuntimeError("Interceptor failure")


class TestAttackHooksBasic:
    """Test basic AttackHooks functionality."""
    
    def test_init_without_attacker(self):
        """Test initialization without an attacker."""
        hooks = AttackHooks()
        assert hooks.attacker is None
        assert hooks.interceptors == []
        assert hooks.adversarial_agents == []
        assert hooks.interaction_log == []
    
    def test_init_with_attacker(self):
        """Test initialization with an attacker."""
        mock_attacker = Mock()
        hooks = AttackHooks(attacker=mock_attacker)
        assert hooks.attacker is mock_attacker
    
    def test_repr(self):
        """Test string representation."""
        hooks = AttackHooks()
        repr_str = repr(hooks)
        assert "AttackHooks" in repr_str
        assert "interceptors=0" in repr_str
        assert "adversarial_agents=0" in repr_str


class TestMessageInterception:
    """Test message interception functionality."""
    
    def test_register_interceptor(self):
        """Test registering a message interceptor."""
        hooks = AttackHooks()
        interceptor = MockInterceptor()
        
        hooks.register_interceptor(interceptor)
        
        assert len(hooks.interceptors) == 1
        assert hooks.interceptors[0] is interceptor
    
    def test_register_multiple_interceptors(self):
        """Test registering multiple interceptors."""
        hooks = AttackHooks()
        interceptor1 = MockInterceptor("A: ")
        interceptor2 = MockInterceptor("B: ")
        
        hooks.register_interceptor(interceptor1)
        hooks.register_interceptor(interceptor2)
        
        assert len(hooks.interceptors) == 2
    
    def test_register_invalid_interceptor(self):
        """Test that registering non-interceptor raises TypeError."""
        hooks = AttackHooks()
        
        with pytest.raises(TypeError):
            hooks.register_interceptor("not an interceptor")
    
    def test_intercept_message_no_interceptors(self):
        """Test that message is unchanged when no interceptors registered."""
        hooks = AttackHooks()
        original = "Hello, world!"
        
        result = hooks.intercept_message("agent_1", original)
        
        assert result == original
    
    def test_intercept_message_single_interceptor(self):
        """Test message interception with single interceptor."""
        hooks = AttackHooks()
        interceptor = MockInterceptor("MODIFIED: ")
        hooks.register_interceptor(interceptor)
        
        result = hooks.intercept_message("agent_1", "Hello")
        
        assert result == "MODIFIED: Hello"
        assert len(interceptor.intercepted_messages) == 1
        assert interceptor.intercepted_messages[0] == ("agent_1", "Hello")
    
    def test_intercept_message_chained_interceptors(self):
        """Test that interceptors are applied in order (chained)."""
        hooks = AttackHooks()
        hooks.register_interceptor(MockInterceptor("A: "))
        hooks.register_interceptor(MockInterceptor("B: "))
        
        result = hooks.intercept_message("agent_1", "Hello")
        
        # First interceptor adds "A: ", second adds "B: " to that result
        assert result == "B: A: Hello"
    
    def test_intercept_message_failing_interceptor_continues(self):
        """
        Test that system continues when interceptor fails.
        
        **Validates: Requirements 11.5**
        """
        hooks = AttackHooks()
        hooks.register_interceptor(FailingInterceptor())
        
        # Should not raise, should return original message
        result = hooks.intercept_message("agent_1", "Hello")
        
        assert result == "Hello"


class TestAdversarialAgentInjection:
    """Test adversarial agent injection functionality."""
    
    def test_inject_adversarial_agent(self):
        """Test injecting an adversarial agent."""
        hooks = AttackHooks()
        mock_agent = Mock()
        mock_agent.agent_id = "adversarial_agent_1"
        
        hooks.inject_adversarial_agent(mock_agent)
        
        assert len(hooks.adversarial_agents) == 1
        assert hooks.adversarial_agents[0] is mock_agent
    
    def test_inject_multiple_adversarial_agents(self):
        """Test injecting multiple adversarial agents."""
        hooks = AttackHooks()
        agent1 = Mock()
        agent1.agent_id = "adversarial_1"
        agent2 = Mock()
        agent2.agent_id = "adversarial_2"
        
        hooks.inject_adversarial_agent(agent1)
        hooks.inject_adversarial_agent(agent2)
        
        assert len(hooks.adversarial_agents) == 2
    
    def test_get_adversarial_agents_returns_copy(self):
        """Test that get_adversarial_agents returns a copy."""
        hooks = AttackHooks()
        mock_agent = Mock()
        hooks.inject_adversarial_agent(mock_agent)
        
        agents = hooks.get_adversarial_agents()
        agents.clear()  # Modify the returned list
        
        # Original should be unchanged
        assert len(hooks.adversarial_agents) == 1
    
    def test_get_adversarial_agents_empty(self):
        """Test get_adversarial_agents when none injected."""
        hooks = AttackHooks()
        
        agents = hooks.get_adversarial_agents()
        
        assert agents == []


class TestInteractionLogging:
    """Test interaction logging functionality."""
    
    def test_log_interaction(self):
        """Test logging an interaction."""
        hooks = AttackHooks()
        interaction = {
            'agent_id': 'user_agent_123',
            'round_idx': 1,
            'message': 'I suggest item 42'
        }
        
        hooks.log_interaction(interaction)
        
        assert len(hooks.interaction_log) == 1
        assert hooks.interaction_log[0]['agent_id'] == 'user_agent_123'
        assert hooks.interaction_log[0]['seq'] == 0  # Sequence number added
    
    def test_log_multiple_interactions(self):
        """Test logging multiple interactions with sequence numbers."""
        hooks = AttackHooks()
        
        hooks.log_interaction({'agent_id': 'agent_1'})
        hooks.log_interaction({'agent_id': 'agent_2'})
        hooks.log_interaction({'agent_id': 'agent_3'})
        
        assert len(hooks.interaction_log) == 3
        assert hooks.interaction_log[0]['seq'] == 0
        assert hooks.interaction_log[1]['seq'] == 1
        assert hooks.interaction_log[2]['seq'] == 2
    
    def test_export_log(self, tmp_path):
        """Test exporting interaction log to file."""
        hooks = AttackHooks()
        hooks.log_interaction({'agent_id': 'agent_1', 'message': 'test'})
        
        log_path = tmp_path / "test_log.json"
        hooks.export_log(str(log_path))
        
        assert log_path.exists()
        
        import json
        with open(log_path) as f:
            exported = json.load(f)
        
        assert len(exported) == 1
        assert exported[0]['agent_id'] == 'agent_1'
    
    def test_export_log_creates_directory(self, tmp_path):
        """Test that export_log creates parent directories."""
        hooks = AttackHooks()
        hooks.log_interaction({'test': 'data'})
        
        log_path = tmp_path / "subdir" / "nested" / "log.json"
        hooks.export_log(str(log_path))
        
        assert log_path.exists()


class TestAttackHooksReset:
    """Test reset functionality."""
    
    def test_clear_log(self):
        """Test clearing the interaction log."""
        hooks = AttackHooks()
        hooks.log_interaction({'test': 'data'})
        hooks.log_interaction({'test': 'data2'})
        
        hooks.clear_log()
        
        assert len(hooks.interaction_log) == 0
    
    def test_clear_adversarial_agents(self):
        """Test clearing adversarial agents."""
        hooks = AttackHooks()
        hooks.inject_adversarial_agent(Mock())
        hooks.inject_adversarial_agent(Mock())
        
        hooks.clear_adversarial_agents()
        
        assert len(hooks.adversarial_agents) == 0
    
    def test_clear_interceptors(self):
        """Test clearing interceptors."""
        hooks = AttackHooks()
        hooks.register_interceptor(MockInterceptor())
        hooks.register_interceptor(MockInterceptor())
        
        hooks.clear_interceptors()
        
        assert len(hooks.interceptors) == 0
    
    def test_reset_clears_all(self):
        """Test that reset clears everything."""
        hooks = AttackHooks()
        hooks.register_interceptor(MockInterceptor())
        hooks.inject_adversarial_agent(Mock())
        hooks.log_interaction({'test': 'data'})
        
        hooks.reset()
        
        assert len(hooks.interceptors) == 0
        assert len(hooks.adversarial_agents) == 0
        assert len(hooks.interaction_log) == 0


class TestOrchestratorAttackIntegration:
    """Test attack hooks integration with MACFOrchestrator."""
    
    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM."""
        llm = Mock()
        llm.generate_response = Mock(return_value=Mock(content="test response"))
        return llm
    
    @pytest.fixture
    def mock_toolkit(self):
        """Create a mock toolkit."""
        toolkit = Mock()
        toolkit.get_similar_users = Mock(return_value=[1, 2, 3])
        toolkit.get_relevant_items = Mock(return_value=[101, 102])
        toolkit.index_manager = Mock()
        toolkit.index_manager.get_user_history = Mock(return_value=[101, 102, 103])
        return toolkit
    
    @pytest.fixture
    def mock_config(self):
        """Create a mock config."""
        from connacf.macf.config import MACFConfig
        return MACFConfig(
            neighbor_count=3,
            history_item_count=2,
            max_rounds=2,
            top_k_recommendation=5
        )
    
    def test_orchestrator_accepts_attack_hooks(self, mock_llm, mock_toolkit, mock_config):
        """Test that orchestrator accepts attack_hooks parameter."""
        from connacf.macf.orchestrator import MACFOrchestrator
        
        hooks = AttackHooks()
        orchestrator = MACFOrchestrator(
            llm=mock_llm,
            toolkit=mock_toolkit,
            config=mock_config,
            attack_hooks=hooks
        )
        
        assert orchestrator.attack_hooks is hooks
    
    def test_orchestrator_without_attack_hooks(self, mock_llm, mock_toolkit, mock_config):
        """Test that orchestrator works without attack_hooks."""
        from connacf.macf.orchestrator import MACFOrchestrator
        
        orchestrator = MACFOrchestrator(
            llm=mock_llm,
            toolkit=mock_toolkit,
            config=mock_config
        )
        
        assert orchestrator.attack_hooks is None
    
    def test_recruit_agents_includes_adversarial(self, mock_llm, mock_toolkit, mock_config):
        """
        Test that recruit_agents includes adversarial agents.
        
        **Validates: Requirements 11.6**
        """
        from connacf.macf.orchestrator import MACFOrchestrator
        from connacf.macf.agents import UserAgent
        
        hooks = AttackHooks()
        
        # Create a mock adversarial agent
        adversarial_agent = Mock(spec=UserAgent)
        adversarial_agent.agent_id = "adversarial_user_999"
        hooks.inject_adversarial_agent(adversarial_agent)
        
        orchestrator = MACFOrchestrator(
            llm=mock_llm,
            toolkit=mock_toolkit,
            config=mock_config,
            attack_hooks=hooks
        )
        
        # Mock the agent creation to avoid LLM calls
        with patch.object(orchestrator, '_create_user_agent') as mock_create_user:
            with patch.object(orchestrator, '_create_item_agent') as mock_create_item:
                mock_create_user.return_value = Mock()
                mock_create_item.return_value = Mock()
                
                user_agents, item_agents = orchestrator.recruit_agents(
                    target_user_id=1,
                    query="test query"
                )
        
        # The adversarial agent should be included
        assert adversarial_agent in user_agents


class TestAttackHooksProperty20:
    """
    Property-based tests for attack message interception.
    
    Property 20: Attack Message Interception
    For any registered message interceptor and agent message, the interceptor's
    intercept() method shall be called and its return value shall replace
    the original message in the discussion.
    
    **Validates: Requirements 11.2, 11.6**
    """
    
    def test_interceptor_return_value_replaces_message(self):
        """Test that interceptor return value replaces original message."""
        hooks = AttackHooks()
        
        class ReplacingInterceptor(MessageInterceptor):
            def intercept(self, agent_id: str, message: str) -> str:
                return "REPLACED"
        
        hooks.register_interceptor(ReplacingInterceptor())
        
        result = hooks.intercept_message("agent_1", "original message")
        
        assert result == "REPLACED"
        assert result != "original message"
    
    def test_all_interceptors_called_in_order(self):
        """Test that all interceptors are called in registration order."""
        hooks = AttackHooks()
        call_order = []
        
        class OrderTrackingInterceptor(MessageInterceptor):
            def __init__(self, name: str):
                self.name = name
            
            def intercept(self, agent_id: str, message: str) -> str:
                call_order.append(self.name)
                return f"{self.name}:{message}"
        
        hooks.register_interceptor(OrderTrackingInterceptor("first"))
        hooks.register_interceptor(OrderTrackingInterceptor("second"))
        hooks.register_interceptor(OrderTrackingInterceptor("third"))
        
        result = hooks.intercept_message("agent_1", "msg")
        
        assert call_order == ["first", "second", "third"]
        assert result == "third:second:first:msg"


class TestAttackHooksProperty21:
    """
    Property-based tests for attack logging and continuation.
    
    Property 21: Attack Logging and Continuation
    For any attack that modifies agent behavior, the system shall log
    the original and modified messages and continue execution to completion.
    
    **Validates: Requirements 11.4, 11.5**
    """
    
    def test_system_continues_after_interceptor_failure(self):
        """Test that system continues execution when interceptor fails."""
        hooks = AttackHooks()
        
        # Add a failing interceptor
        hooks.register_interceptor(FailingInterceptor())
        
        # Should not raise, should return original
        result = hooks.intercept_message("agent_1", "test message")
        
        assert result == "test message"
    
    def test_logging_records_interactions(self):
        """Test that interactions are logged for analysis."""
        hooks = AttackHooks()
        
        # Log multiple interactions
        for i in range(5):
            hooks.log_interaction({
                'round_idx': i,
                'agent_id': f'agent_{i}',
                'message': f'message_{i}'
            })
        
        # All interactions should be logged
        assert len(hooks.interaction_log) == 5
        
        # Each should have a sequence number
        for i, log_entry in enumerate(hooks.interaction_log):
            assert log_entry['seq'] == i
            assert log_entry['round_idx'] == i
