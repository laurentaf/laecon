"""Tests for LAECON MCP server health check.

M1 target: expand to cover all 9 tools.
For BASIC, we just verify health and list_supported_operations work.
"""
from laecon.mcp.server import health, list_supported_operations


def test_health_returns_ok():
    """Health check should return ok status with capability metadata."""
    result = health()
    assert result["status"] == "ok"
    assert result["capability"] == "laecon"
    assert "version" in result
    assert "0.1.0-BASIC" == result["version"]
    assert "status_detail" in result
    assert "BASIC" in result["status_detail"]


def test_list_supported_operations_includes_9_tools():
    """List should include exactly 9 tools (2 implemented + 7 stubs)."""
    result = list_supported_operations()
    assert "operations" in result
    assert len(result["operations"]) == 9

    tool_names = {op["name"] for op in result["operations"]}
    expected = {
        "health", "list_supported_operations",
        "train_regression", "train_classifier",
        "validate_assumptions", "interpret_model",
        "evaluate_model", "predict",
        "export_diagnostic_report",
    }
    assert tool_names == expected


def test_list_supported_operations_marks_2_implemented():
    """In BASIC, exactly 2 tools should be marked as implemented."""
    result = list_supported_operations()
    implemented = [op for op in result["operations"] if op["status"] == "implemented"]
    assert len(implemented) == 2
    assert {op["name"] for op in implemented} == {"health", "list_supported_operations"}


def test_list_supported_operations_includes_evolution_plan():
    """Catalog should include evolution plan with M1, M2, M3 milestones."""
    result = list_supported_operations()
    assert "evolution_plan" in result
    plan = result["evolution_plan"]
    assert "current" in plan
    assert "next_milestone" in plan
    assert "M1" in plan["next_milestone"]
    assert "milestones_post_stable" in plan
    assert "M2" in plan["milestones_post_stable"]
    assert "M3" in plan["milestones_post_stable"]


def test_list_supported_operations_documents_17_conditions():
    """Catalog should reference the 17 vinculating conditions from the Council."""
    result = list_supported_operations()
    assert "vinculating_conditions" in result
    vc = result["vinculating_conditions"]
    assert vc["total"] == 17
    assert vc["blocking_stable"] is True
