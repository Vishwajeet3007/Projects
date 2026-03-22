"""Tests for agent JSON parsing."""

from app.agents.factory import AgentFactory



def test_parse_json_content_accepts_wrapped_json() -> None:
    factory = AgentFactory.__new__(AgentFactory)
    parsed = factory._parse_json_content(
        "Here is the result:\n```json\n{\n  \"bugs\": [],\n  \"final_summary\": \"ok\",\n}\n```"
    )

    assert parsed["final_summary"] == "ok"



def test_parse_json_content_accepts_plain_embedded_json() -> None:
    factory = AgentFactory.__new__(AgentFactory)
    parsed = factory._parse_json_content(
        "Result follows {\"bugs\": [], \"final_summary\": \"ok\"} Thanks"
    )

    assert parsed["bugs"] == []
