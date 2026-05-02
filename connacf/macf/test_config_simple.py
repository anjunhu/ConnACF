"""
Simple test script to verify MACFConfig functionality without pytest.
"""

import sys
import tempfile
import os

from connacf.macf.config import MACFConfig


def test_default_config():
    """Test default configuration values."""
    print("Testing default configuration...")
    config = MACFConfig()
    assert config.neighbor_count == 5, f"Expected 5, got {config.neighbor_count}"
    assert config.history_item_count == 5, f"Expected 5, got {config.history_item_count}"
    assert config.retrieval_k == 15, f"Expected 15, got {config.retrieval_k}"
    assert config.max_rounds == 5, f"Expected 5, got {config.max_rounds}"
    assert config.top_k_recommendation == 10, f"Expected 10, got {config.top_k_recommendation}"
    assert config.api_batch == 25, f"Expected 25, got {config.api_batch}"
    assert config.chat_api_batch == 10, f"Expected 10, got {config.chat_api_batch}"
    print("✓ Default configuration test passed")


def test_from_dict():
    """Test loading from dictionary."""
    print("\nTesting from_dict...")
    config_dict = {
        'neighbor_count': 10,
        'max_rounds': 3,
        'unknown_param': 'should_be_ignored'
    }
    config = MACFConfig.from_dict(config_dict)
    assert config.neighbor_count == 10, f"Expected 10, got {config.neighbor_count}"
    assert config.max_rounds == 3, f"Expected 3, got {config.max_rounds}"
    # Should use defaults for missing params
    assert config.history_item_count == 5, f"Expected 5, got {config.history_item_count}"
    assert config.retrieval_k == 15, f"Expected 15, got {config.retrieval_k}"
    print("✓ from_dict test passed")


def test_from_yaml():
    """Test loading from YAML file."""
    print("\nTesting from_yaml...")
    yaml_content = """
neighbor_count: 10
history_item_count: 8
retrieval_k: 20
max_rounds: 3
top_k_recommendation: 15
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(yaml_content)
        temp_path = f.name
    
    try:
        config = MACFConfig.from_yaml(temp_path)
        assert config.neighbor_count == 10, f"Expected 10, got {config.neighbor_count}"
        assert config.history_item_count == 8, f"Expected 8, got {config.history_item_count}"
        assert config.retrieval_k == 20, f"Expected 20, got {config.retrieval_k}"
        assert config.max_rounds == 3, f"Expected 3, got {config.max_rounds}"
        assert config.top_k_recommendation == 15, f"Expected 15, got {config.top_k_recommendation}"
        print("✓ from_yaml test passed")
    finally:
        os.unlink(temp_path)


def test_from_yaml_partial():
    """Test loading from partial YAML (should use defaults for missing)."""
    print("\nTesting from_yaml with partial config...")
    yaml_content = """
neighbor_count: 10
max_rounds: 3
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(yaml_content)
        temp_path = f.name
    
    try:
        config = MACFConfig.from_yaml(temp_path)
        assert config.neighbor_count == 10, f"Expected 10, got {config.neighbor_count}"
        assert config.max_rounds == 3, f"Expected 3, got {config.max_rounds}"
        # Should use defaults
        assert config.history_item_count == 5, f"Expected 5, got {config.history_item_count}"
        assert config.retrieval_k == 15, f"Expected 15, got {config.retrieval_k}"
        print("✓ from_yaml partial test passed")
    finally:
        os.unlink(temp_path)


def test_validation():
    """Test configuration validation."""
    print("\nTesting validation...")
    
    # Valid config
    config = MACFConfig()
    assert config.validate() is True, "Valid config should pass validation"
    
    # Invalid configs
    invalid_config = MACFConfig(neighbor_count=-1)
    assert invalid_config.validate() is False, "Negative neighbor_count should fail"
    
    invalid_config = MACFConfig(max_rounds=0)
    assert invalid_config.validate() is False, "Zero max_rounds should fail"
    
    print("✓ Validation test passed")


def test_load_actual_yaml():
    """Test loading the actual MACF.yaml file."""
    print("\nTesting loading actual MACF.yaml...")
    try:
        config = MACFConfig.from_yaml('connacf/props/MACF.yaml')
        assert config.neighbor_count == 5
        assert config.history_item_count == 5
        assert config.retrieval_k == 15
        assert config.max_rounds == 5
        assert config.top_k_recommendation == 10
        print("✓ Actual MACF.yaml loaded successfully")
        print(f"\nLoaded configuration:\n{config}")
    except Exception as e:
        print(f"✗ Failed to load MACF.yaml: {e}")
        raise


def main():
    """Run all tests."""
    print("=" * 60)
    print("MACF Configuration Tests")
    print("=" * 60)
    
    try:
        test_default_config()
        test_from_dict()
        test_from_yaml()
        test_from_yaml_partial()
        test_validation()
        test_load_actual_yaml()
        
        print("\n" + "=" * 60)
        print("All tests passed! ✓")
        print("=" * 60)
        return 0
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        return 1
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
