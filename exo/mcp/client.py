"""MCP client implementation."""

import asyncio
from typing import Dict, Optional, Any, List
from .transports import StdioTransport, HTTPTransport, SSETransport, TransportError
from exo import DEBUG


class MCPClientError(Exception):
    """Exception raised by MCP client operations."""
    pass


class MCPClient:
    """MCP client that handles communication with MCP servers."""
    def __init__(self, name: str, transport):
        self.name = name
        self.transport = transport
        self.initialized = False
        self.tools: List[Dict[str, Any]] = []
        self.resources: List[Dict[str, Any]] = []
        
    async def initialize(self) -> None:
        """Initialize the MCP client connection."""
        try:
            await self.transport.connect()
            
            # Send initialize request
            init_response = await self.transport.send_request(
                "initialize",
                {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {
                        "tools": {}
                    },
                    "clientInfo": {
                        "name": "exo",
                        "version": "0.0.1"
                    }
                }
            )
            
            if DEBUG >= 2:
                print(f"[MCP {self.name}] Initialized: {init_response}")
            
            # Send initialized notification (no response expected)
            try:
                if hasattr(self.transport, 'send_notification'):
                    await self.transport.send_notification("initialized", {})
                else:
                    # Fallback for transports that don't support notifications
                    # Send as request but ignore response
                    try:
                        await self.transport.send_request("initialized", {})
                    except Exception:
                        # Some servers don't support initialized as a request, that's ok
                        if DEBUG >= 2:
                            print(f"[MCP {self.name}] initialized notification not supported")
            except Exception as e:
                if DEBUG >= 2:
                    print(f"[MCP {self.name}] Failed to send initialized notification: {e}")
            
            # List available tools (gracefully handle if not supported)
            try:
                await self.list_tools()
            except MCPClientError as e:
                if DEBUG >= 1:
                    print(f"[MCP {self.name}] Could not list tools (may not be supported): {e}")
                # Continue anyway - tools might be available later
            
            self.initialized = True
            
        except TransportError as e:
            raise MCPClientError(f"Failed to initialize MCP client {e}")
    
    async def list_tools(self) -> List[Dict[str, Any]]:
        """List available tools from the MCP server."""
        try:
            response = await self.transport.send_request("tools/list", {})
            self.tools = response.get("tools", [])
            
            if DEBUG >= 1:
                print(f"[MCP {self.name}] Discovered {len(self.tools)} tools")
            
            return self.tools
        except TransportError as e:
            if DEBUG >= 1:
                print(f"[MCP {self.name}] Failed to list tools: {e}")
            return []
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Call a tool on the MCP server."""
        if not self.initialized:
            raise MCPClientError(f"Client {self.name} not initialized")
        
        try:
            response = await self.transport.send_request(
                "tools/call",
                {
                    "name": tool_name,
                    "arguments": arguments
                }
            )
            return response
        except TransportError as e:
            raise MCPClientError(f"Failed to call tool {tool_name}: {e}")
    
    async def disconnect(self) -> None:
        """Disconnect from the MCP server."""
        try:
            await self.transport.disconnect()
            self.initialized = False
        except Exception as e:
            if DEBUG >= 1:
                print(f"[MCP {self.name}] Error during disconnect: {e}")

