"""
connacf.attack.tools
====================
Dummy tool definitions for adversarial attack research.

These tools are *never executed* — they exist as declared capabilities
injected into agent system prompts so that LLMs can recognise and format
Action/Action Input blocks when poisoned content commands them to.

Usage
-----
    from connacf.attack.tools import get_tool_spec_block, TOOL_REGISTRY

    # All tools
    spec = get_tool_spec_block()

    # Subset
    spec = get_tool_spec_block(["BankManagerPayBill", "GmailSendEmail"])

    # Individual tool metadata
    tool = TOOL_REGISTRY["WikipediaSearch"]
"""

from connacf.attack.tools.dummy_tools import (
    TOOL_REGISTRY,
    TOOL_NAMES,
    get_tool_spec_block,
    get_react_scratchpad_prefix,
)

__all__ = [
    "TOOL_REGISTRY",
    "TOOL_NAMES",
    "get_tool_spec_block",
    "get_react_scratchpad_prefix",
]
