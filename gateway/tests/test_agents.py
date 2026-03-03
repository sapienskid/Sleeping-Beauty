"""Unit tests for agent routing / intent detection."""

from __future__ import annotations

import pytest

from gateway.services.agents import detect_intent, get_agent, strip_norag_prefix


class TestDetectIntent:
    def test_norag_prefix(self):
        assert detect_intent("!norag What is 2+2?") == "passthrough"

    def test_norag_uppercase(self):
        assert detect_intent("!NORAG hello") == "passthrough"

    def test_remember(self):
        assert detect_intent("remember that I like Python") == "remember"

    def test_note_that(self):
        assert detect_intent("note that my name is Alice") == "remember"

    def test_keep_in_mind(self):
        assert detect_intent("keep in mind that deadlines are Fridays") == "remember"

    def test_from_now_on(self):
        assert detect_intent("from now on use formal English") == "remember"

    def test_draft_email(self):
        assert detect_intent("draft an email to the team") == "draft"

    def test_write_me(self):
        assert detect_intent("write me a cover letter") == "draft"

    def test_compose(self):
        assert detect_intent("compose a blog post about AI") == "draft"

    def test_default_search(self):
        assert detect_intent("How do I install Arch Linux?") == "search"

    def test_empty_string(self):
        assert detect_intent("") == "search"

    def test_ambiguous_defaults_to_search(self):
        assert detect_intent("What is the weather today?") == "search"


class TestStripNoragPrefix:
    def test_strips_prefix(self):
        assert strip_norag_prefix("!norag What is 2+2?") == "What is 2+2?"

    def test_no_prefix_unchanged(self):
        assert strip_norag_prefix("Hello world") == "Hello world"

    def test_preserves_content_after_strip(self):
        result = strip_norag_prefix("!norag   Explain quantum physics")
        assert result == "Explain quantum physics"


class TestGetAgent:
    def test_search_agent(self):
        agent = get_agent("search")
        assert agent.name == "search"
        assert agent.confidence_threshold == 0.35

    def test_draft_agent(self):
        agent = get_agent("draft")
        assert agent.name == "draft"
        assert agent.confidence_threshold == 0.25
        assert "writing-style" in agent.memory_types

    def test_remember_agent(self):
        agent = get_agent("remember")
        assert agent.name == "remember"
        assert agent.collections == []

    def test_passthrough_agent(self):
        agent = get_agent("passthrough")
        assert agent.name == "passthrough"

    def test_unknown_intent_falls_back_to_search(self):
        agent = get_agent("unknown_intent")
        assert agent.name == "search"
