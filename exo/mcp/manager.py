"""MCP server manager that watches mcp.json and manages server lifecycles."""

import asyncio
import json
import os
from pathlib import Path
from typing import Dict, Optional, Any, List
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from .client import MCPClient, MCPClientError
from .transports import StdioTransport, HTTPTransport, SSETransport
from exo import DEBUG


class MCPConfigWatcher(FileSystemEventHandler):
    """File system event handler for mcp.json changes."""
    
    def __init__(self, manager):
        self.manager = manager
        
    def on_modified(self, event):
        if event.src_path.endswith('mcp.json'):
            asyncio.create_task(self.manager.reload_config())
    
    def on_created(self, event):
        if event.src_path.endswith('mcp.json'):
            asyncio.create_task(self.manager.reload_config())


class MCPServerManager:
    """Manages MCP server lifecycles and watches for config changes."""
    
    def __init__(self, config_path: Optional[str] = None, topology_viz: Optional[Any] = None):
        self.config_path = Path(config_path or "mcp.json").expanduser().resolve()
        self.clients: Dict[str, MCPClient] = {}
        self.observer: Optional[Observer] = None
        self._reload_lock = asyncio.Lock()
        self.topology_viz = topology_viz
        self.server_status: Dict[str, Dict[str, Any]] = {}  # server_name -> {status, error, tools_count}
        self._failed_configs: Dict[str, Dict[str, Any]] = {}  # server_name -> config (for retry)
        self._retry_task: Optional[asyncio.Task] = None
        self._connection_tasks: Dict[str, asyncio.Task] = {}  # server_name -> in-progress connection task
        self._connection_lock = asyncio.Lock()  # Lock for protecting _connection_tasks dictionary
        
    async def start(self) -> None:
        """Start the MCP manager and watch for config changes."""
        # Start file watcher
        self.observer = Observer()
        # Set as daemon thread so it doesn't prevent process exit
        self.observer.daemon = True
        event_handler = MCPConfigWatcher(self)
        config_dir = self.config_path.parent
        config_dir.mkdir(parents=True, exist_ok=True)
        self.observer.schedule(event_handler, str(config_dir), recursive=False)
        self.observer.start()
        
        if DEBUG >= 1:
            print(f"[MCP] Watching config file: {self.config_path}")
        
        # Load initial config in background (non-blocking)
        asyncio.create_task(self.reload_config())
        
        # Start automatic retry task
        self._retry_task = asyncio.create_task(self._auto_retry_loop())
    
    async def stop(self) -> None:
        """Stop the MCP manager and disconnect all clients."""
        # Stop automatic retry task
        if self._retry_task:
            self._retry_task.cancel()
            try:
                await self._retry_task
            except asyncio.CancelledError:
                pass
            self._retry_task = None
        
        if self.observer:
            try:
                self.observer.stop()
                # Use timeout to prevent hanging
                # Check if thread is daemon before joining
                is_daemon = False
                if hasattr(self.observer, '_thread') and self.observer._thread:
                    is_daemon = self.observer._thread.daemon
                if not is_daemon:
                    self.observer.join(timeout=0.5)
            except Exception as e:
                if DEBUG >= 2:
                    print(f"[MCP] Error stopping observer: {e}")
            finally:
                self.observer = None
        
        # Disconnect all clients
        for client in list(self.clients.values()):
            try:
                await client.disconnect()
            except Exception as e:
                if DEBUG >= 1:
                    print(f"[MCP] Error disconnecting client {client.name}: {e}")
        
        self.clients.clear()
        self.server_status.clear()
    
    async def reload_config(self) -> None:
        """Reload MCP configuration from file."""
        async with self._reload_lock:
            try:
                if not self.config_path.exists():
                    if DEBUG >= 1:
                        print(f"[MCP] Config file not found: {self.config_path}")
                    # Disconnect all clients if config file is removed
                    await self._disconnect_all()
                    return
                
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                
                servers = config.get("mcpServers", {})
                
                if DEBUG >= 1:
                    print(f"[MCP] Reloading config with {len(servers)} servers")
                
                # Find servers to remove
                current_names = set(servers.keys())
                existing_names = set(self.clients.keys()) | set(self.server_status.keys())
                to_remove = existing_names - current_names
                
                # Disconnect removed servers
                for name in to_remove:
                    if DEBUG >= 1:
                        print(f"[MCP] Removing server: {name}")
                    # Cancel any in-progress connection
                    await self._cancel_connection(name)
                    await self._disconnect_client(name)
                    # Also remove from failed_configs if present
                    self._failed_configs.pop(name, None)
                
                # Set status entries immediately for all servers in config (before connecting)
                # This ensures the TUI panel shows up right away
                for name in servers.keys():
                    if name not in self.server_status:
                        self.server_status[name] = {"status": "connecting", "error": None, "tools_count": 0}
                
                # Update TUI immediately to show servers that are about to connect
                self._update_tui()
                
                # Add or update servers
                for name, server_config in servers.items():
                    if name in self.clients:
                        # Check if config changed
                        if self._config_changed(name, server_config):
                            if DEBUG >= 1:
                                print(f"[MCP] Updating server: {name}")
                            # Cancel any in-progress connection for this server
                            await self._cancel_connection(name)
                            await self._disconnect_client(name)
                            await self._connect_client(name, server_config)
                    else:
                        # Cancel any in-progress connection before starting new one
                        await self._cancel_connection(name)
                        await self._connect_client(name, server_config)
                        
            except json.JSONDecodeError as e:
                if DEBUG >= 1:
                    print(f"[MCP] Invalid JSON in config file: {e}")
            except Exception as e:
                if DEBUG >= 1:
                    print(f"[MCP] Error reloading config: {e}")
    
    def _config_changed(self, name: str, new_config: Dict[str, Any]) -> bool:
        """Check if server config has changed."""
        # Simple comparison - in production, might want more sophisticated diffing
        # For now, we'll just reconnect if the config exists
        return True  # Always reconnect for simplicity
    
    async def _cancel_connection(self, name: str) -> None:
        """Cancel any in-progress connection attempt for a server."""
        task = None
        async with self._connection_lock:
            if name in self._connection_tasks:
                task = self._connection_tasks.pop(name)
                if not task.done():
                    task.cancel()
        
        # Wait for cancellation outside the lock
        if task and not task.done():
            try:
                await task
            except asyncio.CancelledError:
                if DEBUG >= 2:
                    print(f"[MCP] Cancelled in-progress connection for server: {name}")
            except Exception as e:
                if DEBUG >= 2:
                    print(f"[MCP] Error cancelling connection for {name}: {e}")
    
    async def _connect_client(self, name: str, config: Dict[str, Any]) -> None:
        """Connect to an MCP server based on its configuration."""
        # Check if connection is already in progress - exit immediately if so
        async with self._connection_lock:
            # Check if connection is already in progress
            if name in self._connection_tasks:
                existing_task = self._connection_tasks[name]
                if not existing_task.done():
                    if DEBUG >= 2:
                        print(f"[MCP] Connection already in progress for {name}, exiting immediately")
                    return
                else:
                    # Task is done, clean it up
                    self._connection_tasks.pop(name, None)
            
            # Create connection task
            connection_task = asyncio.create_task(self._connect_client_impl(name, config))
            self._connection_tasks[name] = connection_task
        
        # Wait for connection to complete (or be cancelled) - outside the lock
        try:
            await connection_task
        except asyncio.CancelledError:
            if DEBUG >= 2:
                print(f"[MCP] Connection cancelled for {name}")
            # Clean up on cancellation
            async with self._connection_lock:
                if name in self._connection_tasks and self._connection_tasks[name] == connection_task:
                    self._connection_tasks.pop(name, None)
            raise
        except Exception:
            # Clean up on error
            async with self._connection_lock:
                if name in self._connection_tasks and self._connection_tasks[name] == connection_task:
                    self._connection_tasks.pop(name, None)
            raise
    
    async def _cleanup_transport(self, transport) -> None:
        """Clean up transport on error (unless it's mcp-remote waiting for OAuth)."""
        if transport and hasattr(transport, '_is_mcp_remote') and not transport._is_mcp_remote:
            try:
                await transport.disconnect()
            except Exception:
                pass
    
    def _set_error_status(self, name: str, error_msg: str, config: Dict[str, Any]) -> None:
        """Set error status for a server and store config for retry."""
        self.server_status[name] = {"status": "error", "error": error_msg, "tools_count": 0}
        self._failed_configs[name] = config
        self._update_tui()
    
    async def _connect_client_impl(self, name: str, config: Dict[str, Any]) -> None:
        """Internal implementation of client connection."""
        self.server_status[name] = {"status": "connecting", "error": None, "tools_count": 0}
        self._update_tui()
        
        transport = None
        try:
            # Check if we've been cancelled before starting
            if asyncio.current_task() and asyncio.current_task().cancelled():
                return
            
            # Determine transport type
            if "command" in config:
                transport = StdioTransport(config["command"], config.get("args", []), config.get("env", {}))
            elif "url" in config:
                url, headers = config["url"], config.get("headers", {})

                if "/sse" in url.lower() or url.endswith("/sse"):
                    transport = SSETransport(url, headers)
                else:
                    transport = HTTPTransport(url, headers)
            else:
                self._set_error_status(name, "Invalid server config: missing 'command' or 'url'", config)
                
                if DEBUG >= 1:
                    print(f"[MCP] Invalid server config for {name}: {error_msg}")
                return
            
            # Create and initialize client
            self.clients[name] = MCPClient(name, transport)
            await self.clients[name].initialize()

            # Update status
            self.server_status[name] = {"status": "connected", "error": None, "tools_count": len(self.clients[name].tools)}
            # Remove from failed configs on success
            self._failed_configs.pop(name, None)
            self._update_tui()
            
            if DEBUG >= 1:
                print(f"[MCP] Connected to server: {name}")
                
        except asyncio.CancelledError:
            await self._cleanup_transport(transport)
            if DEBUG >= 2:
                print(f"[MCP] Connection cancelled for {name}")
            raise
        except (MCPClientError, Exception) as e:
            await self._cleanup_transport(transport)
            error_msg = str(e) if isinstance(e, MCPClientError) else f"Unexpected error: {str(e)}"
            self._set_error_status(name, error_msg, config)
            if DEBUG >= 1:
                print(f"[MCP] Failed to connect to server {name}: {e}")
    
    async def _disconnect_client(self, name: str) -> None:
        """Disconnect an MCP client."""
        # Cancel any in-progress connection
        await self._cancel_connection(name)
        
        if name in self.clients:
            client = self.clients.pop(name)
            try:
                await client.disconnect()
            except Exception as e:
                if DEBUG >= 1:
                    print(f"[MCP] Error disconnecting {name}: {e}")
        
        # Remove status
        if name in self.server_status:
            del self.server_status[name]
            self._update_tui()
    
    async def _disconnect_all(self) -> None:
        """Disconnect all clients."""
        for name in list(self.clients.keys()):
            await self._disconnect_client(name)
    
    def get_all_tools(self) -> List[Dict[str, Any]]:
        """Get all tools from all connected MCP servers."""
        all_tools = []
        for client in self.clients.values():
            for tool in client.tools:
                # Prefix tool name with server name to avoid conflicts
                tool_copy = tool.copy()
                tool_copy["name"] = f"mcp_{client.name}_{tool['name']}"
                tool_copy["_mcp_server"] = client.name
                tool_copy["_mcp_tool_name"] = tool["name"]
                all_tools.append(tool_copy)
        return all_tools
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Call an MCP tool by its prefixed name."""
        # Parse prefixed tool name: mcp_{server}_{tool}
        if not tool_name.startswith("mcp_"):
            error_msg = f"Invalid MCP tool name: {tool_name}"
            self._update_tui_error(error_msg)
            raise MCPClientError(error_msg)
        
        parts = tool_name.split("_", 2)
        if len(parts) < 3:
            error_msg = f"Invalid MCP tool name format: {tool_name}"
            self._update_tui_error(error_msg)
            raise MCPClientError(error_msg)
        
        server_name = parts[1]
        actual_tool_name = parts[2]
        
        if server_name not in self.clients:
            error_msg = f"MCP server not found: {server_name}"
            self._update_tui_error(error_msg)
            raise MCPClientError(error_msg)
        
        client = self.clients[server_name]
        
        try:
            result = await client.call_tool(actual_tool_name, arguments)
            # Clear any previous errors for this server
            if server_name in self.server_status:
                if self.server_status[server_name]["status"] == "error":
                    self.server_status[server_name]["status"] = "connected"
                    self.server_status[server_name]["error"] = None
                    self._update_tui()
            return result
        except MCPClientError as e:
            error_msg = f"Tool call failed on {server_name}: {str(e)}"
            if server_name in self.server_status:
                self.server_status[server_name]["status"] = "error"
                self.server_status[server_name]["error"] = error_msg
            self._update_tui_error(error_msg)
            raise
        except Exception as e:
            error_msg = f"Unexpected error calling tool {actual_tool_name} on {server_name}: {str(e)}"
            if server_name in self.server_status:
                self.server_status[server_name]["status"] = "error"
                self.server_status[server_name]["error"] = error_msg
            self._update_tui_error(error_msg)
            raise MCPClientError(error_msg)
    
    def _update_tui(self) -> None:
        """Update TUI with MCP server status."""
        if self.topology_viz:
            try:
                self.topology_viz.update_mcp_status(self.server_status)
            except Exception as e:
                if DEBUG >= 2:
                    print(f"[MCP] Error updating TUI: {e}")
    
    def _update_tui_error(self, error_msg: str) -> None:
        """Update TUI with error message."""
        if self.topology_viz:
            try:
                self.topology_viz.update_mcp_error(error_msg)
            except Exception as e:
                if DEBUG >= 2:
                    print(f"[MCP] Error updating TUI with error: {e}")
    
    def _load_server_config(self, name: str) -> Optional[Dict[str, Any]]:
        """Load server config from failed_configs or config file."""
        if name in self._failed_configs:
            return self._failed_configs[name]
        if not self.config_path.exists():
            return None
        try:
            with open(self.config_path, 'r') as f:
                servers = json.load(f).get("mcpServers", {})
                return servers.get(name)
        except Exception:
            return None
    
    async def retry_server(self, server_name: Optional[str] = None) -> Dict[str, Any]:
        """Retry connection to a failed server or all failed servers."""
        results = {}
        
        if server_name:
            if server_name in self.clients:
                return {server_name: {"success": False, "error": "Server is already connected"}}
            
            config = self._load_server_config(server_name)
            if not config:
                return {server_name: {"success": False, "error": f"Server '{server_name}' not found"}}
            
            if DEBUG >= 1:
                print(f"[MCP] Retrying server: {server_name}")
            try:
                await self._connect_client(server_name, config)
                results[server_name] = {"success": True, "error": None}
            except Exception as e:
                results[server_name] = {"success": False, "error": str(e)}
        else:
            # Retry all failed servers
            failed_names = [
                name for name, status in self.server_status.items()
                if status.get("status") == "error" and name not in self.clients
            ]
            
            if not failed_names:
                return {"message": "No failed servers to retry"}
            
            if DEBUG >= 1:
                print(f"[MCP] Retrying {len(failed_names)} failed server(s)")
            
            for name in failed_names:
                config = self._load_server_config(name)
                if not config:
                    results[name] = {"success": False, "error": "Config not available"}
                    continue
                
                try:
                    await self._connect_client(name, config)
                    results[name] = {"success": True, "error": None}
                except Exception as e:
                    results[name] = {"success": False, "error": str(e)}
        
        return results
    
    async def _auto_retry_loop(self) -> None:
        """Background task that automatically retries failed servers every 30 seconds."""
        while self._retry_task and not self._retry_task.done():
            try:
                await asyncio.sleep(30.0)
                
                if not self._retry_task or self._retry_task.done():
                    break
                
                failed_names = [
                    name for name, status in self.server_status.items()
                    if status.get("status") == "error" and name not in self.clients
                    and (name not in self._connection_tasks or self._connection_tasks[name].done())
                ]
                
                if not failed_names:
                    continue
                
                if DEBUG >= 2:
                    print(f"[MCP] Auto-retrying {len(failed_names)} failed server(s)")
                
                for name in failed_names:
                    if not self._retry_task or self._retry_task.done():
                        break
                    
                    config = self._load_server_config(name)
                    if config:
                        try:
                            await self._connect_client(name, config)
                            if DEBUG >= 1:
                                print(f"[MCP] Auto-retry succeeded for server: {name}")
                        except Exception as e:
                            if DEBUG >= 2:
                                print(f"[MCP] Auto-retry failed for server {name}: {e}")
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                if DEBUG >= 2:
                    print(f"[MCP] Error in auto-retry loop: {e}")
                await asyncio.sleep(1.0)

