"""
Custom MCP client.

Wraps the official mcp ClientSession over a stdio transport, and pipes every
resource read / tool call through the shared Tracer so calls made *by any
worker agent* still show up in the single trace log.
"""

from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class MCPServerClient:
    def __init__(self, server_script: str, tracer=None, agent_name: str = "mcp-client"):
        self.server_script = server_script
        self.tracer = tracer
        self.agent_name = agent_name
        self._stack = AsyncExitStack()
        self.session: ClientSession | None = None

    async def connect(self):
        params = StdioServerParameters(command="python3", args=[self.server_script])
        read, write = await self._stack.enter_async_context(stdio_client(params))
        self.session = await self._stack.enter_async_context(ClientSession(read, write))
        await self.session.initialize()
        if self.tracer:
            self.tracer.log(self.agent_name, "connected", self.server_script, {})

    async def list_capabilities(self):
        tools = await self.session.list_tools()
        resources = await self.session.list_resources()
        return (
            [t.name for t in tools.tools],
            [str(r.uri) for r in resources.resources],
        )

    async def read_resource(self, uri: str) -> str:
        if self.tracer:
            self.tracer.log(self.agent_name, "resource_read_start", uri, {})
        result = await self.session.read_resource(uri)
        text = result.contents[0].text if result.contents else ""
        if self.tracer:
            self.tracer.log(self.agent_name, "resource_read_end", uri, {"result": text[:200]})
        return text

    async def call_tool(self, name: str, arguments: dict, caller: str | None = None) -> str:
        agent = caller or self.agent_name
        if self.tracer:
            self.tracer.log(agent, "tool_call_start", name, arguments)
        result = await self.session.call_tool(name, arguments)
        text = result.content[0].text if result.content else ""
        if self.tracer:
            self.tracer.log(agent, "tool_call_end", name, {"result": text})
        return text

    async def close(self):
        await self._stack.aclose()
