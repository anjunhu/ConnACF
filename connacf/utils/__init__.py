"""Utility modules for ConnaCF."""

# Re-export from utils.py (the file in parent directory)
import importlib.util
import os

_utils_py = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'connacf', 'utils.py')
if not os.path.exists(_utils_py):
    _utils_py = os.path.join(os.path.dirname(__file__), '..', 'utils.py')

_spec = importlib.util.spec_from_file_location("_utils_file", _utils_py)
_utils = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_utils)

get_model = _utils.get_model
check_path = _utils.check_path
dispatch_bedrock_requests = _utils.dispatch_bedrock_requests
dispatch_single_bedrock_request = _utils.dispatch_single_bedrock_request

# Token utilities from this package
from .token_utils import count_tokens, truncate_to_tokens
from .gpu_utils import free_device

__all__ = [
    'get_model', 
    'check_path',
    'dispatch_bedrock_requests',
    'dispatch_single_bedrock_request',
    'count_tokens', 
    'truncate_to_tokens',
    'free_device',
]
