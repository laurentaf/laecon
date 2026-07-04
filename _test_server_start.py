"""Test that the MCP server module initializes correctly."""
import sys
sys.path.insert(0, ".")

# This exercises all imports
from laecon.mcp.server import mcp, health, list_supported_operations
from laecon.mcp.server import (
    train_regression, train_classifier, validate_assumptions,
    interpret_model, evaluate_model, predict, export_diagnostic_report,
)

h = health()
assert h["status"] == "ok", f"Health check failed: {h}"
print(f"Health: {h['status']}, version: {h['version']}")

ops = list_supported_operations()
implemented = [o["name"] for o in ops["operations"] if o["status"] == "implemented"]
print(f"Implemented tools ({len(implemented)}):")
for name in implemented:
    print(f"  - {name}")

assert "train_regression" in implemented
assert "train_classifier" in implemented
assert "validate_assumptions" in implemented
assert "interpret_model" in implemented
assert "evaluate_model" in implemented
assert "predict" in implemented
assert "export_diagnostic_report" in implemented

print()
print("=== ALL 7 MCP TOOLS REGISTERED AND HEALTH CHECK PASSED ===")
