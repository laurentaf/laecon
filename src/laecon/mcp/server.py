import asyncio
from mcp.server import Server, stdio
from mcp.types import Tool, TextContent

server = Server("laecon")

SUPPORTED = [
    "health",
    "list_supported_operations",
    "train_regression",
    "train_classifier",
    "validate_assumptions",
    "interpret_model",
    "evaluate_model",
    "predict",
    "export_diagnostic_report",
]

@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="health",
            description="Check server health status",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="list_supported_operations",
            description="List all supported operations",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="train_regression",
            description="Train a regression model (OLS, GLM, etc.)",
            inputSchema={
                "type": "object",
                "properties": {
                    "formula": {"type": "string"},
                    "data": {"type": "object"},
                    "model_type": {
                        "type": "string",
                        "enum": ["ols", "glm", "logit", "probit"],
                        "default": "ols",
                    },
                },
                "required": ["formula", "data"],
            },
        ),
        Tool(
            name="train_classifier",
            description="Train a classification model",
            inputSchema={
                "type": "object",
                "properties": {
                    "features": {"type": "array", "items": {"type": "string"}},
                    "target": {"type": "string"},
                    "data": {"type": "object"},
                },
                "required": ["features", "target", "data"],
            },
        ),
        Tool(
            name="validate_assumptions",
            description="Validate econometric assumptions (normality, homoscedasticity, etc.)",
            inputSchema={
                "type": "object",
                "properties": {
                    "model_id": {"type": "string"},
                },
                "required": ["model_id"],
            },
        ),
        Tool(
            name="interpret_model",
            description="Interpret model coefficients and feature importance",
            inputSchema={
                "type": "object",
                "properties": {
                    "model_id": {"type": "string"},
                    "method": {
                        "type": "string",
                        "enum": ["coefficients", "shap", "pvalues"],
                        "default": "coefficients",
                    },
                },
                "required": ["model_id"],
            },
        ),
        Tool(
            name="evaluate_model",
            description="Evaluate model performance (R2, AIC, BIC, confusion matrix)",
            inputSchema={
                "type": "object",
                "properties": {
                    "model_id": {"type": "string"},
                    "test_data": {"type": "object"},
                },
                "required": ["model_id"],
            },
        ),
        Tool(
            name="predict",
            description="Generate predictions from a trained model",
            inputSchema={
                "type": "object",
                "properties": {
                    "model_id": {"type": "string"},
                    "new_data": {"type": "object"},
                },
                "required": ["model_id", "new_data"],
            },
        ),
        Tool(
            name="export_diagnostic_report",
            description="Export a full diagnostic report for a model",
            inputSchema={
                "type": "object",
                "properties": {
                    "model_id": {"type": "string"},
                    "format": {
                        "type": "string",
                        "enum": ["json", "markdown"],
                        "default": "json",
                    },
                },
                "required": ["model_id"],
            },
        ),
    ]

@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    if name == "health":
        return [TextContent(type="text", text='{"status": "ok", "server": "laecon"}')]
    elif name == "list_supported_operations":
        return [TextContent(type="text", text=str(SUPPORTED))]
    return [TextContent(type="text", text=f'{{"error": "unknown tool: {name}"}}')]

async def run_mcp():
    async with stdio.stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())

def main():
    asyncio.run(run_mcp())

if __name__ == "__main__":
    main()
