# tools.py
from mcp_tools import McpToolSpec
from pistachio_mcp_agent import pistachio_query

pistachio_tool = McpToolSpec(
    name="pistachio_agent",
    description="Answer questions about pistachio growing (local RAG + LLM).",
    func=pistachio_query,
    args_schema={"query": str}
)
