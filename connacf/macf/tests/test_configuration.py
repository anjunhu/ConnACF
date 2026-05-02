"""
Unit and property-based tests for MACF configuration management.

Tests verify:
- Loading from YAML files
- Loading from dictionaries
- Default value handling
- Parameter validation
- Property 16: Configuration defaults applied for missing parameters
"""

import pytest
import tempfile
import os
from pathlib import Path
from hypothesis import given, strategies as st, settings

from connacf.macf.config import MACFConfig


class TestMACFConfigFromYAML:
    """Test loading configuration from YAML files."""
    
    def test_load_from_valid_yaml(self):
        """Test loading configuration from a valid YAML file."""
        yaml_content = """
neighbor_count: 10
history_item_count: 8
retrieval_k: 20
max_rounds: 3
top_k_recommendation: 15
api_batch: 30
chat_api_batch: 15
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name
        
        try:
            config = MACFConfig.from_yaml(temp_path)
            assert config.neighbor_count == 10
            assert config.history_item_count == 8
            assert config.retrieval_k == 20
            assert config.max_rounds == 3
            assert config.top_k_recommendation == 15
            assert config.api_batch == 30
            assert config.chat_api_batch == 15
        finally:
            os.unlink(temp_path)
    
    def test_load_from_partial_yaml(self):
        """Test loading with some parameters missing (should use defaults)."""
        yaml_content = """
neighbor_count: 10
max_rounds: 3
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name
        
        try:
            config = MACFConfig.from_yaml(temp_path)
            assert config.neighbor_count == 10
            assert config.max_rounds == 3
            # These should use defaults
            assert config.history_item_count == 5
            assert config.retrieval_k == 15
            assert config.top_k_recommendation == 10
        finally:
            os.unlink(temp_path)
    
    def test_load_from_empty_yaml(self):
        """Test loading from empty YAML file (should use all defaults)."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("")
            temp_path = f.name
        
        try:
            config = MACFConfig.from_yaml(temp_path)
            # All should be defaults
            assert config.neighbor_count == 5
            assert config.history_item_count == 5
            assert config.retrieval_k == 15
            assert config.max_rounds == 5
            assert config.top_k_recommendation == 10
            assert config.api_batch == 25
            assert config.chat_api_batch == 10
        finally:
            os.unlink(temp_path)
    
    def test_load_from_nonexistent_file(self):
        """Test that loading from non-existent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            MACFConfig.from_yaml('/nonexistent/path/config.yaml')
    
    def test_load_from_malformed_yaml(self):
        """Test that malformed YAML raises appropriate error."""
        yaml_content = """
neighbor_count: 10
  invalid_indentation: true
max_rounds: [unclosed list
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name
        
        try:
            with pytest.raises(Exception):  # yaml.YAMLError or similar
                MACFConfig.from_yaml(temp_path)
        finally:
            os.unlink(temp_path)


class TestMACFConfigFromDict:
    """Test loading configuration from dictionaries."""
    
    def test_load_from_complete_dict(self):
        """Test loading from dictionary with all parameters."""
        config_dict = {
            'neighbor_count': 10,
            'history_item_count': 8,
            'retrieval_k': 20,
            'max_rounds': 3,
            'top_k_recommendation': 15,
            'api_batch': 30,
            'chat_api_batch': 15
        }
        config = MACFConfig.from_dict(config_dict)
        assert config.neighbor_count == 10
        assert config.history_item_count == 8
        assert config.retrieval_k == 20
        assert config.max_rounds == 3
        assert config.top_k_recommendation == 15
        assert config.api_batch == 30
        assert config.chat_api_batch == 15
    
    def test_load_from_partial_dict(self):
        """Test loading from dictionary with some parameters missing."""
        config_dict = {
            'neighbor_count': 10,
            'max_rounds': 3
        }
        config = MACFConfig.from_dict(config_dict)
        assert config.neighbor_count == 10
        assert config.max_rounds == 3
        # These should use defaults
        assert config.history_item_count == 5
        assert config.retrieval_k == 15
        assert config.top_k_recommendation == 10
    
    def test_load_from_empty_dict(self):
        """Test loading from empty dictionary (should use all defaults)."""
        config = MACFConfig.from_dict({})
        assert config.neighbor_count == 5
        assert config.history_item_count == 5
        assert config.retrieval_k == 15
        assert config.max_rounds == 5
        assert config.top_k_recommendation == 10
        assert config.api_batch == 25
        assert config.chat_api_batch == 10
    
    def test_load_with_extra_parameters(self):
        """Test that extra parameters in dict are ignored."""
        config_dict = {
            'neighbor_count': 10,
            'unknown_param': 'should_be_ignored',
            'another_unknown': 42
        }
        config = MACFConfig.from_dict(config_dict)
        assert config.neighbor_count == 10
        assert not hasattr(config, 'unknown_param')
        assert not hasattr(config, 'another_unknown')


class TestMACFConfigDefaults:
    """Test default value handling."""
    
    def test_default_values(self):
        """Test that default constructor uses correct default values."""
        config = MACFConfig()
        assert config.neighbor_count == 5
        assert config.history_item_count == 5
        assert config.retrieval_k == 15
        assert config.max_rounds == 5
        assert config.top_k_recommendation == 10
        assert config.api_batch == 25
        assert config.chat_api_batch == 10


class TestMACFConfigValidation:
    """Test configuration validation."""
    
    def test_valid_config(self):
        """Test that valid configuration passes validation."""
        config = MACFConfig()
        assert config.validate() is True
    
    def test_invalid_neighbor_count(self):
        """Test that negative neighbor_count fails validation."""
        config = MACFConfig(neighbor_count=-1)
        assert config.validate() is False
    
    def test_invalid_history_item_count(self):
        """Test that zero history_item_count fails validation."""
        config = MACFConfig(history_item_count=0)
        assert config.validate() is False
    
    def test_invalid_retrieval_k(self):
        """Test that negative retrieval_k fails validation."""
        config = MACFConfig(retrieval_k=-5)
        assert config.validate() is False
    
    def test_invalid_max_rounds(self):
        """Test that zero max_rounds fails validation."""
        config = MACFConfig(max_rounds=0)
        assert config.validate() is False
    
    def test_invalid_top_k(self):
        """Test that negative top_k_recommendation fails validation."""
        config = MACFConfig(top_k_recommendation=-10)
        assert config.validate() is False


class TestMACFConfigPropertyBased:
    """Property-based tests for configuration."""
    
    @settings(max_examples=100)
    @given(
        neighbor_count=st.integers(min_value=-10, max_value=100),
        history_item_count=st.integers(min_value=-10, max_value=100),
        retrieval_k=st.integers(min_value=-10, max_value=100),
        max_rounds=st.integers(min_value=-10, max_value=100),
        top_k_recommendation=st.integers(min_value=-10, max_value=100)
    )
    def test_property_16_defaults_applied_for_missing_params(
        self,
        neighbor_count,
        history_item_count,
        retrieval_k,
        max_rounds,
        top_k_recommendation
    ):
        """
        Feature: connacf-macf-integration, Property 16: Configuration Defaults Applied for Missing Parameters
        
        **Validates: Requirements 7.2, 7.3, 7.4, 7.5, 7.6, 7.7**
        
        For any MACF configuration where a parameter is not specified,
        the system shall use the documented default value.
        """
        # Test with partial dictionary (some params missing)
        config_dict = {}
        
        # Randomly include some parameters
        if neighbor_count > 0:
            config_dict['neighbor_count'] = neighbor_count
        if history_item_count > 0:
            config_dict['history_item_count'] = history_item_count
        if retrieval_k > 0:
            config_dict['retrieval_k'] = retrieval_k
        if max_rounds > 0:
            config_dict['max_rounds'] = max_rounds
        if top_k_recommendation > 0:
            config_dict['top_k_recommendation'] = top_k_recommendation
        
        config = MACFConfig.from_dict(config_dict)
        
        # Verify that specified parameters are used
        if 'neighbor_count' in config_dict:
            assert config.neighbor_count == neighbor_count
        else:
            # Default should be used
            assert config.neighbor_count == 5
        
        if 'history_item_count' in config_dict:
            assert config.history_item_count == history_item_count
        else:
            assert config.history_item_count == 5
        
        if 'retrieval_k' in config_dict:
            assert config.retrieval_k == retrieval_k
        else:
            assert config.retrieval_k == 15
        
        if 'max_rounds' in config_dict:
            assert config.max_rounds == max_rounds
        else:
            assert config.max_rounds == 5
        
        if 'top_k_recommendation' in config_dict:
            assert config.top_k_recommendation == top_k_recommendation
        else:
            assert config.top_k_recommendation == 10
        
        # api_batch and chat_api_batch should always use defaults when not specified
        assert config.api_batch == 25
        assert config.chat_api_batch == 10
    
    @settings(max_examples=50)
    @given(
        config_dict=st.dictionaries(
            keys=st.sampled_from([
                'neighbor_count', 'history_item_count', 'retrieval_k',
                'max_rounds', 'top_k_recommendation', 'api_batch', 'chat_api_batch'
            ]),
            values=st.integers(min_value=1, max_value=100),
            min_size=0,
            max_size=7
        )
    )
    def test_from_dict_never_fails_with_valid_params(self, config_dict):
        """Test that from_dict never raises with valid parameter values."""
        config = MACFConfig.from_dict(config_dict)
        
        # Verify all attributes exist
        assert hasattr(config, 'neighbor_count')
        assert hasattr(config, 'history_item_count')
        assert hasattr(config, 'retrieval_k')
        assert hasattr(config, 'max_rounds')
        assert hasattr(config, 'top_k_recommendation')
        assert hasattr(config, 'api_batch')
        assert hasattr(config, 'chat_api_batch')
        
        # Verify all values are positive (either from dict or defaults)
        assert config.neighbor_count > 0
        assert config.history_item_count > 0
        assert config.retrieval_k > 0
        assert config.max_rounds > 0
        assert config.top_k_recommendation > 0
        assert config.api_batch > 0
        assert config.chat_api_batch > 0


class TestMACFConfigStringRepresentation:
    """Test string representation of configuration."""
    
    def test_str_representation(self):
        """Test that __str__ produces readable output."""
        config = MACFConfig(neighbor_count=10, max_rounds=3)
        str_repr = str(config)
        
        # Should contain all parameter names and values
        assert 'neighbor_count=10' in str_repr
        assert 'max_rounds=3' in str_repr
        assert 'history_item_count=5' in str_repr  # default
        assert 'MACFConfig' in str_repr
