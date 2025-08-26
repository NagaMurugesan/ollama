from mcp.client import MCPClient
from tools import pistachio_tool
import asyncio

client = MCPClient(host="http://localhost:8000")
asyncio.run(client.register_tool(pistachio_tool))
