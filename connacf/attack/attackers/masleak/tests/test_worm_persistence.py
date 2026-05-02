"""
Property-based tests for worm persistence across update_memory assignments.

Property 9: Worm persistence under update_memory for both agent types
  For any MASLeakUserAgent or MASLeakItemAgent, and for any sequence of
  update_memory assignments with arbitrary new profile strings, the first
  element of update_memory should always contain the current _worm_payload
  as a substring.
  Validates: Requirements 5.1, 5.3, 5.4

Property 10: Base profile preserved after update_worm_for_round
  For any MASLeakUserAgent with a non-empty base profile, calling
  update_worm_for_round(round_idx) for any round index should produce a
  update_memory[0] that still contains the base profile as a substring.
  Validates: Requirements 5.5
"""

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from connacf.attack.attackers.masleak.attacker_agents import (
    MASLeakUserAgent,
    MASLeakItemAgent,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BASE_CONFIG = {
    'use_data_template': True,
    'targets': ['system_prompt'],
    'domain_config': {'domain': 'movies'},
}


def make_user_agent(original_profile: str = "I am a movie fan.") -> MASLeakUserAgent:
    return MASLeakUserAgent(
        user_id=1,
        config=_BASE_CONFIG,
        original_profile=original_profile,
    )


def make_item_agent(original_description: str = "A great film.") -> MASLeakItemAgent:
    return MASLeakItemAgent(
        item_id=1,
        config=_BASE_CONFIG,
        original_description=original_description,
    )


# ---------------------------------------------------------------------------
# Property 9: Worm persistence under update_memory — MASLeakUserAgent
# Validates: Requirements 5.1, 5.3, 5.4
# ---------------------------------------------------------------------------

@given(
    profiles=st.lists(
        st.text(min_size=1, max_size=100, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd'), blacklist_characters=' \t\n\r\xa0')),
        min_size=1,
        max_size=10,
    )
)
@settings(max_examples=100)
def test_property9_user_agent_worm_persists_across_update_memory(profiles):
    """
    Property 9 (user agent): After any sequence of update_memory assignments,
    update_memory[0] always contains _worm_payload as a substring.

    **Validates: Requirements 5.1, 5.3, 5.4**
    """
    agent = make_user_agent()
    worm = agent._worm_payload
    assume(worm)  # worm must be non-empty for the property to be meaningful

    for profile in profiles:
        agent.update_memory = [profile]
        assert worm in agent.update_memory[0], (
            f"Worm payload lost after update_memory assignment.\n"
            f"  profile: {profile!r}\n"
            f"  worm:    {worm!r}\n"
            f"  result:  {agent.update_memory[0]!r}"
        )


# ---------------------------------------------------------------------------
# Property 9: Worm persistence under update_memory — MASLeakItemAgent
# Validates: Requirements 5.1, 5.3, 5.4
# ---------------------------------------------------------------------------

@given(
    descriptions=st.lists(
        st.text(min_size=1, max_size=100, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd'), blacklist_characters=' \t\n\r\xa0')),
        min_size=1,
        max_size=10,
    )
)
@settings(max_examples=100)
def test_property9_item_agent_worm_persists_across_update_memory(descriptions):
    """
    Property 9 (item agent): After any sequence of update_memory assignments,
    update_memory[0] always contains _worm_payload as a substring.

    **Validates: Requirements 5.1, 5.3, 5.4**
    """
    agent = make_item_agent()
    worm = agent._worm_payload
    assume(worm)

    for desc in descriptions:
        agent.update_memory = [desc]
        assert worm in agent.update_memory[0], (
            f"Worm payload lost after update_memory assignment.\n"
            f"  description: {desc!r}\n"
            f"  worm:        {worm!r}\n"
            f"  result:      {agent.update_memory[0]!r}"
        )


# ---------------------------------------------------------------------------
# Property 10: Base profile preserved after update_worm_for_round
# Validates: Requirements 5.5
# ---------------------------------------------------------------------------

@given(
    base_profile=st.text(
        min_size=1,
        max_size=80,
        alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd'), blacklist_characters=' \t\n\r\xa0'),
    ),
    round_idx=st.integers(min_value=0, max_value=20),
)
@settings(max_examples=100)
def test_property10_base_profile_preserved_after_update_worm_for_round(base_profile, round_idx):
    """
    Property 10: After update_worm_for_round(round_idx), update_memory[0]
    still contains the base profile as a substring.

    **Validates: Requirements 5.5**
    """
    agent = make_user_agent(original_profile=base_profile)
    # Simulate ConnaCF writing a new profile (sets _base_profile)
    agent.update_memory = [base_profile]

    agent.update_worm_for_round(round_idx)

    assert base_profile in agent.update_memory[0], (
        f"Base profile lost after update_worm_for_round({round_idx}).\n"
        f"  base_profile: {base_profile!r}\n"
        f"  result:       {agent.update_memory[0]!r}"
    )


# ---------------------------------------------------------------------------
# Property 10 (item agent variant): same guarantee for MASLeakItemAgent
# Validates: Requirements 5.5
# ---------------------------------------------------------------------------

@given(
    base_desc=st.text(
        min_size=1,
        max_size=80,
        alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd'), blacklist_characters=' \t\n\r\xa0'),
    ),
    round_idx=st.integers(min_value=0, max_value=20),
)
@settings(max_examples=100)
def test_property10_item_agent_base_profile_preserved_after_update_worm_for_round(base_desc, round_idx):
    """
    Property 10 (item agent): After update_worm_for_round(round_idx),
    update_memory[0] still contains the base description as a substring.

    **Validates: Requirements 5.5**
    """
    agent = make_item_agent(original_description=base_desc)
    agent.update_memory = [base_desc]

    agent.update_worm_for_round(round_idx)

    assert base_desc in agent.update_memory[0], (
        f"Base description lost after update_worm_for_round({round_idx}).\n"
        f"  base_desc: {base_desc!r}\n"
        f"  result:    {agent.update_memory[0]!r}"
    )
