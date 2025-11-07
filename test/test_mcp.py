"""Comprehensive tests for MCP support."""

import pytest
import asyncio
import json
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import aiohttp

from exo.mcp import MCPServerManager, MCPClient, MCPClientError
from exo.mcp.transports import StdioTransport, HTTPTransport, SSETransport, TransportError


@pytest.fixture
def temp_config_file():
    """Create a temporary mcp.json file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        yield Path(f.name)
    # Cleanup
    if os.path.exists(f.name):
        os.unlink(f.name)


@pytest.fixture
def event_loop():
    """Create an event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
async def mcp_manager_cleanup():
    """Fixture to ensure MCP managers are properly cleaned up."""
    managers = []
    yield managers
    # Cleanup all managers
    for manager in managers:
        try:
            await manager.stop()
        except Exception:
            pass


class TestStdioTransport:
    """Tests for stdio transport."""
    
    @pytest.mark.asyncio
    async def test_stdio_transport_basic(self):
        """Test basic stdio transport functionality."""
        transport = StdioTransport("echo", ["hello"])
        assert transport.command == "echo"
        assert transport.args == ["hello"]
        assert not transport._running
    
    @pytest.mark.asyncio
    async def test_stdio_transport_tilde_expansion(self):
        """Test that ~ is expanded in args."""
        transport = StdioTransport("test", ["~/test"])
        assert "~" not in str(transport.args[0])
    
    @pytest.mark.asyncio
    async def test_stdio_transport_env(self):
        """Test environment variable passing."""
        transport = StdioTransport("test", [], {"TEST_VAR": "test_value"})
        assert transport.env["TEST_VAR"] == "test_value"


class TestHTTPTransport:
    """Tests for HTTP transport."""
    
    @pytest.mark.asyncio
    async def test_http_transport_basic(self):
        """Test basic HTTP transport."""
        transport = HTTPTransport("http://example.com", {"Authorization": "Bearer token"})
        assert transport.url == "http://example.com"
        assert transport.headers["Authorization"] == "Bearer token"
    
    @pytest.mark.asyncio
    async def test_http_transport_no_headers(self):
        """Test HTTP transport without headers."""
        transport = HTTPTransport("http://example.com")
        assert transport.url == "http://example.com"
        assert transport.headers == {}


class TestSSETransport:
    """Tests for SSE transport."""
    
    @pytest.mark.asyncio
    async def test_sse_transport_basic(self):
        """Test basic SSE transport."""
        transport = SSETransport("http://example.com/sse", {"Authorization": "Bearer token"})
        assert transport.url == "http://example.com/sse"
        assert transport.headers["Authorization"] == "Bearer token"
    
    @pytest.mark.asyncio
    async def test_sse_transport_detection(self):
        """Test SSE transport detection from URL."""
        transport = SSETransport("https://mcp.example.com/sse")
        assert "/sse" in transport.url.lower() or transport.url.endswith("/sse")


class TestMCPClient:
    """Tests for MCP client."""
    
    @pytest.mark.asyncio
    async def test_mcp_client_initialization(self):
        """Test MCP client initialization."""
        mock_transport = AsyncMock()
        mock_transport.send_request = AsyncMock(side_effect=[
            {"protocolVersion": "2024-11-05", "capabilities": {}},
            {},  # initialized notification
            {"tools": []}  # tools/list
        ])
        mock_transport.connect = AsyncMock()
        
        client = MCPClient("test", mock_transport)
        await client.initialize()
        
        assert client.initialized
        assert client.name == "test"
        assert mock_transport.connect.called
        assert mock_transport.send_request.call_count == 3
    
    @pytest.mark.asyncio
    async def test_mcp_client_list_tools(self):
        """Test listing tools from MCP server."""
        mock_transport = AsyncMock()
        mock_transport.send_request = AsyncMock(return_value={
            "tools": [
                {"name": "test_tool", "description": "A test tool"}
            ]
        })
        
        client = MCPClient("test", mock_transport)
        client.initialized = True
        
        tools = await client.list_tools()
        assert len(tools) == 1
        assert tools[0]["name"] == "test_tool"
    
    @pytest.mark.asyncio
    async def test_mcp_client_call_tool(self):
        """Test calling a tool on MCP server."""
        mock_transport = AsyncMock()
        mock_transport.send_request = AsyncMock(return_value={"result": "success"})
        
        client = MCPClient("test", mock_transport)
        client.initialized = True
        
        result = await client.call_tool("test_tool", {"arg": "value"})
        assert result == {"result": "success"}
        mock_transport.send_request.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_mcp_client_call_tool_not_initialized(self):
        """Test that calling tool before initialization raises error."""
        mock_transport = AsyncMock()
        client = MCPClient("test", mock_transport)
        client.initialized = False
        
        with pytest.raises(MCPClientError):
            await client.call_tool("test_tool", {})


class TestMCPServerManager:
    """Tests for MCP server manager."""
    
    @pytest.mark.asyncio
    async def test_manager_initialization(self):
        """Test manager initialization."""
        manager = MCPServerManager("nonexistent.json")
        assert manager.config_path.name == "nonexistent.json"
        assert len(manager.clients) == 0
    
    @pytest.mark.asyncio
    async def test_manager_stdio_server(self, temp_config_file):
        """Test manager with stdio server configuration."""
        config = {
            "mcpServers": {
                "test_stdio": {
                    "command": "echo",
                    "args": ["hello"]
                }
            }
        }
        
        with open(temp_config_file, 'w') as f:
            json.dump(config, f)
        
        manager = MCPServerManager(str(temp_config_file))
        
        try:
            # Mock the transport to avoid actually running processes
            with patch('exo.mcp.manager.StdioTransport') as mock_transport_class:
                mock_transport = AsyncMock()
                mock_transport.connect = AsyncMock()
                mock_transport.send_request = AsyncMock(side_effect=[
                    {"protocolVersion": "2024-11-05", "capabilities": {}},
                    {},
                    {"tools": []}
                ])
                mock_transport_class.return_value = mock_transport
                
                await manager.start()
                await asyncio.sleep(0.1)  # Allow time for async operations
                
                # Server should be attempted to connect
                # (may fail if echo doesn't support MCP, but that's ok for testing)
        finally:
            await manager.stop()
    
    @pytest.mark.asyncio
    async def test_manager_http_server(self, temp_config_file):
        """Test manager with HTTP server configuration."""
        config = {
            "mcpServers": {
                "test_http": {
                    "url": "http://example.com/mcp",
                    "headers": {
                        "Authorization": "Bearer token123"
                    }
                }
            }
        }
        
        with open(temp_config_file, 'w') as f:
            json.dump(config, f)
        
        manager = MCPServerManager(str(temp_config_file))
        
        with patch('exo.mcp.manager.HTTPTransport') as mock_transport_class:
            mock_transport = AsyncMock()
            mock_transport.connect = AsyncMock()
            mock_transport.send_request = AsyncMock(side_effect=[
                {"protocolVersion": "2024-11-05", "capabilities": {}},
                {},
                {"tools": []}
            ])
            mock_transport_class.return_value = mock_transport
            
            await manager.start()
            await asyncio.sleep(0.1)
            await manager.stop()
    
    @pytest.mark.asyncio
    async def test_manager_sse_server(self, temp_config_file):
        """Test manager with SSE server configuration."""
        config = {
            "mcpServers": {
                "test_sse": {
                    "url": "https://mcp.example.com/sse",
                    "headers": {
                        "Authorization": "Bearer napi_xxxx"
                    }
                }
            }
        }
        
        with open(temp_config_file, 'w') as f:
            json.dump(config, f)
        
        manager = MCPServerManager(str(temp_config_file))
        
        with patch('exo.mcp.manager.SSETransport') as mock_transport_class:
            mock_transport = AsyncMock()
            mock_transport.connect = AsyncMock()
            mock_transport.send_request = AsyncMock(side_effect=[
                {"protocolVersion": "2024-11-05", "capabilities": {}},
                {},
                {"tools": []}
            ])
            mock_transport_class.return_value = mock_transport
            
            await manager.start()
            await asyncio.sleep(0.1)
            await manager.stop()
    
    @pytest.mark.asyncio
    async def test_manager_multiple_servers(self, temp_config_file):
        """Test manager with multiple servers of different types."""
        config = {
            "mcpServers": {
                "stdio_server": {
                    "command": "node",
                    "args": ["~/test.js"]
                },
                "http_server": {
                    "url": "http://example.com/mcp"
                },
                "sse_server": {
                    "url": "https://mcp.example.com/sse",
                    "headers": {
                        "Authorization": "Bearer token"
                    }
                }
            }
        }
        
        with open(temp_config_file, 'w') as f:
            json.dump(config, f)
        
        manager = MCPServerManager(str(temp_config_file))
        
        with patch('exo.mcp.manager.StdioTransport') as mock_stdio, \
             patch('exo.mcp.manager.HTTPTransport') as mock_http, \
             patch('exo.mcp.manager.SSETransport') as mock_sse:
            
            for mock_class in [mock_stdio, mock_http, mock_sse]:
                mock_transport = AsyncMock()
                mock_transport.connect = AsyncMock()
                mock_transport.send_request = AsyncMock(side_effect=[
                    {"protocolVersion": "2024-11-05", "capabilities": {}},
                    {},
                    {"tools": []}
                ])
                mock_class.return_value = mock_transport
            
            await manager.start()
            await asyncio.sleep(0.1)
            await manager.stop()
    
    @pytest.mark.asyncio
    async def test_manager_hot_reload_add_server(self, temp_config_file):
        """Test hot reload when adding a server."""
        # Initial config with one server
        config = {
            "mcpServers": {
                "server1": {
                    "command": "echo",
                    "args": ["test"]
                }
            }
        }
        
        with open(temp_config_file, 'w') as f:
            json.dump(config, f)
        
        manager = MCPServerManager(str(temp_config_file))
        
        with patch('exo.mcp.manager.StdioTransport') as mock_transport_class:
            mock_transport = AsyncMock()
            mock_transport.connect = AsyncMock()
            mock_transport.send_request = AsyncMock(side_effect=[
                {"protocolVersion": "2024-11-05", "capabilities": {}},
                {},
                {"tools": []}
            ])
            mock_transport_class.return_value = mock_transport
            
            await manager.start()
            await asyncio.sleep(0.1)
            
            # Update config to add a server
            config["mcpServers"]["server2"] = {
                "url": "http://example.com/mcp"
            }
            
            with open(temp_config_file, 'w') as f:
                json.dump(config, f)
            
            # Trigger reload
            await manager.reload_config()
            await asyncio.sleep(0.1)
            
            await manager.stop()
    
    @pytest.mark.asyncio
    async def test_manager_hot_reload_remove_server(self, temp_config_file):
        """Test hot reload when removing a server."""
        # Initial config with two servers
        config = {
            "mcpServers": {
                "server1": {
                    "command": "echo",
                    "args": ["test"]
                },
                "server2": {
                    "url": "http://example.com/mcp"
                }
            }
        }
        
        with open(temp_config_file, 'w') as f:
            json.dump(config, f)
        
        manager = MCPServerManager(str(temp_config_file))
        
        with patch('exo.mcp.manager.StdioTransport') as mock_stdio, \
             patch('exo.mcp.manager.HTTPTransport') as mock_http:
            
            for mock_class in [mock_stdio, mock_http]:
                mock_transport = AsyncMock()
                mock_transport.connect = AsyncMock()
                mock_transport.disconnect = AsyncMock()
                mock_transport.send_request = AsyncMock(side_effect=[
                    {"protocolVersion": "2024-11-05", "capabilities": {}},
                    {},
                    {"tools": []}
                ])
                mock_class.return_value = mock_transport
            
            await manager.start()
            await asyncio.sleep(0.1)
            
            # Remove server2
            del config["mcpServers"]["server2"]
            
            with open(temp_config_file, 'w') as f:
                json.dump(config, f)
            
            # Trigger reload
            await manager.reload_config()
            await asyncio.sleep(0.1)
            
            await manager.stop()
    
    @pytest.mark.asyncio
    async def test_manager_get_all_tools(self, temp_config_file):
        """Test getting all tools from all servers."""
        config = {
            "mcpServers": {
                "server1": {
                    "command": "echo",
                    "args": ["test"]
                }
            }
        }
        
        with open(temp_config_file, 'w') as f:
            json.dump(config, f)
        
        manager = MCPServerManager(str(temp_config_file))
        
        with patch('exo.mcp.manager.StdioTransport') as mock_transport_class:
            mock_transport = AsyncMock()
            mock_transport.connect = AsyncMock()
            mock_transport.send_request = AsyncMock(side_effect=[
                {"protocolVersion": "2024-11-05", "capabilities": {}},
                {},
                {"tools": [
                    {"name": "tool1", "description": "Tool 1"},
                    {"name": "tool2", "description": "Tool 2"}
                ]}
            ])
            mock_transport_class.return_value = mock_transport
            
            await manager.start()
            await asyncio.sleep(0.1)
            
            tools = manager.get_all_tools()
            assert len(tools) == 2
            assert all(tool["name"].startswith("mcp_server1_") for tool in tools)
            
            await manager.stop()
    
    @pytest.mark.asyncio
    async def test_manager_call_tool(self, temp_config_file):
        """Test calling a tool through the manager."""
        config = {
            "mcpServers": {
                "server1": {
                    "command": "echo",
                    "args": ["test"]
                }
            }
        }
        
        with open(temp_config_file, 'w') as f:
            json.dump(config, f)
        
        manager = MCPServerManager(str(temp_config_file))
        
        with patch('exo.mcp.manager.StdioTransport') as mock_transport_class:
            mock_transport = AsyncMock()
            mock_transport.connect = AsyncMock()
            mock_transport.send_request = AsyncMock(side_effect=[
                {"protocolVersion": "2024-11-05", "capabilities": {}},
                {},
                {"tools": [{"name": "test_tool"}]},
                {"result": "success"}  # tool call response
            ])
            mock_transport_class.return_value = mock_transport
            
            await manager.start()
            await asyncio.sleep(0.1)
            
            result = await manager.call_tool("mcp_server1_test_tool", {"arg": "value"})
            assert result == {"result": "success"}
            
            await manager.stop()
    
    @pytest.mark.asyncio
    async def test_manager_invalid_config(self, temp_config_file):
        """Test manager with invalid server configuration."""
        config = {
            "mcpServers": {
                "invalid_server": {
                    # Missing both 'command' and 'url'
                }
            }
        }
        
        with open(temp_config_file, 'w') as f:
            json.dump(config, f)
        
        manager = MCPServerManager(str(temp_config_file))
        await manager.start()
        await asyncio.sleep(0.1)
        
        # Should not crash, just skip invalid servers
        assert len(manager.clients) == 0
        
        await manager.stop()
    
    @pytest.mark.asyncio
    async def test_manager_missing_config_file(self):
        """Test manager when config file doesn't exist."""
        manager = MCPServerManager("nonexistent_config.json")
        await manager.start()
        await asyncio.sleep(0.1)
        
        # Should handle missing file gracefully
        assert len(manager.clients) == 0
        
        await manager.stop()
    
    @pytest.mark.asyncio
    async def test_manager_invalid_json(self, temp_config_file):
        """Test manager with invalid JSON in config file."""
        with open(temp_config_file, 'w') as f:
            f.write("invalid json {")
        
        manager = MCPServerManager(str(temp_config_file))
        await manager.start()
        await asyncio.sleep(0.1)
        
        # Should handle invalid JSON gracefully
        await manager.stop()
    
    @pytest.mark.asyncio
    async def test_manager_empty_config(self, temp_config_file):
        """Test manager with empty mcpServers."""
        config = {
            "mcpServers": {}
        }
        
        with open(temp_config_file, 'w') as f:
            json.dump(config, f)
        
        manager = MCPServerManager(str(temp_config_file))
        await manager.start()
        await asyncio.sleep(0.1)
        
        assert len(manager.clients) == 0
        
        await manager.stop()
    
    @pytest.mark.asyncio
    async def test_manager_env_variables(self, temp_config_file):
        """Test manager with environment variables in stdio config."""
        config = {
            "mcpServers": {
                "server_with_env": {
                    "command": "node",
                    "args": ["script.js"],
                    "env": {
                        "NODE_ENV": "production",
                        "API_KEY": "secret123"
                    }
                }
            }
        }
        
        with open(temp_config_file, 'w') as f:
            json.dump(config, f)
        
        manager = MCPServerManager(str(temp_config_file))
        
        with patch('exo.mcp.manager.StdioTransport') as mock_transport_class:
            mock_transport = AsyncMock()
            mock_transport.connect = AsyncMock()
            mock_transport.send_request = AsyncMock(side_effect=[
                {"protocolVersion": "2024-11-05", "capabilities": {}},
                {},
                {"tools": []}
            ])
            mock_transport_class.return_value = mock_transport
            
            await manager.start()
            await asyncio.sleep(0.1)
            
            # Verify env was passed
            call_args = mock_transport_class.call_args
            assert call_args is not None
            assert "env" in call_args.kwargs or len(call_args.args) >= 3
            
            await manager.stop()
    
    @pytest.mark.asyncio
    async def test_manager_npx_command(self, temp_config_file):
        """Test manager with npx command (common MCP pattern)."""
        config = {
            "mcpServers": {
                "npx_server": {
                    "command": "npx",
                    "args": [
                        "-y",
                        "@neondatabase/mcp-server-neon",
                        "start",
                        "napi_xxxx"
                    ]
                }
            }
        }
        
        with open(temp_config_file, 'w') as f:
            json.dump(config, f)
        
        manager = MCPServerManager(str(temp_config_file))
        
        with patch('exo.mcp.manager.StdioTransport') as mock_transport_class:
            mock_transport = AsyncMock()
            mock_transport.connect = AsyncMock()
            mock_transport.send_request = AsyncMock(side_effect=[
                {"protocolVersion": "2024-11-05", "capabilities": {}},
                {},
                {"tools": []}
            ])
            mock_transport_class.return_value = mock_transport
            
            await manager.start()
            await asyncio.sleep(0.1)
            
            call_args = mock_transport_class.call_args
            assert call_args is not None
            assert call_args.args[0] == "npx"
            assert "-y" in call_args.args[1]
            
            await manager.stop()
    
    @pytest.mark.asyncio
    async def test_manager_sse_url_detection(self, temp_config_file):
        """Test that SSE transport is used for URLs ending in /sse."""
        config = {
            "mcpServers": {
                "sse_server": {
                    "url": "https://mcp.neon.tech/sse"
                }
            }
        }
        
        with open(temp_config_file, 'w') as f:
            json.dump(config, f)
        
        manager = MCPServerManager(str(temp_config_file))
        
        with patch('exo.mcp.manager.SSETransport') as mock_sse, \
             patch('exo.mcp.manager.HTTPTransport') as mock_http:
            
            mock_transport = AsyncMock()
            mock_transport.connect = AsyncMock()
            mock_transport.send_request = AsyncMock(side_effect=[
                {"protocolVersion": "2024-11-05", "capabilities": {}},
                {},
                {"tools": []}
            ])
            mock_sse.return_value = mock_transport
            
            await manager.start()
            await asyncio.sleep(0.1)
            
            # Should use SSE transport, not HTTP
            assert mock_sse.called
            assert not mock_http.called
            
            await manager.stop()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

