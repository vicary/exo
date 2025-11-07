"""MCP server manager that watches mcp.json and manages server lifecycles."""

import asyncio
import json
import os
import time
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
        self._last_reload_time = 0
        self._reload_debounce = 0.5  # Debounce rapid file changes
        
    def _should_reload(self, event_path: str) -> bool:
        """Check if we should reload based on the event path."""
        # Normalize paths for comparison
        event_path_normalized = str(Path(event_path).resolve())
        config_path_normalized = str(self.manager.config_path.resolve())
        
        # Check if this is the config file we're watching
        if event_path_normalized == config_path_normalized:
            # Debounce rapid changes (some editors trigger multiple events)
            current_time = time.time()
            if current_time - self._last_reload_time < self._reload_debounce:
                return False
            self._last_reload_time = current_time
            return True
        return False
    
    def _schedule_reload(self):
        """Schedule a config reload in the event loop."""
        try:
            # Use the stored event loop from the manager
            loop = self.manager._event_loop
            if loop and loop.is_running():
                # Schedule the coroutine in the running event loop (from file watcher thread)
                asyncio.run_coroutine_threadsafe(self.manager.reload_config(), loop)
            else:
                # Fallback: try to get current event loop
                try:
                    loop = asyncio.get_event_loop()
                    if loop and loop.is_running():
                        asyncio.run_coroutine_threadsafe(self.manager.reload_config(), loop)
                    else:
                        # Try to create task if we're in the same thread
                        asyncio.create_task(self.manager.reload_config())
                except RuntimeError:
                    if DEBUG >= 1:
                        print(f"[MCP] Could not schedule reload: no event loop available")
        except Exception as e:
            if DEBUG >= 1:
                print(f"[MCP] Error scheduling reload: {e}")
    
    def on_modified(self, event):
        if not event.is_directory and self._should_reload(event.src_path):
            if DEBUG >= 1:
                print(f"[MCP] Config file modified: {event.src_path}")
            self._schedule_reload()
    
    def on_created(self, event):
        if not event.is_directory and self._should_reload(event.src_path):
            if DEBUG >= 1:
                print(f"[MCP] Config file created: {event.src_path}")
            self._schedule_reload()
    
    def on_deleted(self, event):
        if not event.is_directory and self._should_reload(event.src_path):
            if DEBUG >= 1:
                print(f"[MCP] Config file deleted: {event.src_path}")
            self._schedule_reload()


class MCPServerManager:
    """Manages MCP server lifecycles and watches for config changes."""
    
    def __init__(self, config_path: Optional[str] = None, topology_viz: Optional[Any] = None, node: Optional[Any] = None):
        self.config_path = Path(config_path or "mcp.json").expanduser().resolve()
        self.clients: Dict[str, MCPClient] = {}
        self.observer: Optional[Observer] = None
        self._reload_lock = asyncio.Lock()
        self.topology_viz = topology_viz
        self.node = node  # Reference to Node for broadcasting availability
        # Unified server info: server_name -> {status, error, tools_count, config, active_config}
        # - status: "connecting", "connected", "error"
        # - error: error message if status is "error"
        # - tools_count: number of tools available
        # - config: latest config from file (for retry if failed)
        # - active_config: config currently running (for change detection)
        self._servers: Dict[str, Dict[str, Any]] = {}
        self._retry_task: Optional[asyncio.Task] = None
        self._connection_tasks: Dict[str, asyncio.Task] = {}  # server_name -> in-progress connection task
        self._connection_lock = asyncio.Lock()  # Lock for protecting _connection_tasks dictionary
        self._event_loop: Optional[asyncio.AbstractEventLoop] = None  # Store event loop for file watcher
        self._last_broadcasted_servers: Dict[str, Dict[str, Any]] = {}  # Track last broadcasted server states
        
    async def start(self) -> None:
        """Start the MCP manager and watch for config changes."""
        # Store event loop reference for file watcher callbacks
        self._event_loop = asyncio.get_running_loop()
        
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
        self._servers.clear()
    
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
                existing_names = set(self.clients.keys()) | set(self._servers.keys())
                to_remove = existing_names - current_names
                
                # Disconnect removed servers
                for name in to_remove:
                    if DEBUG >= 1:
                        print(f"[MCP] Removing server: {name}")
                    # Cancel any in-progress connection
                    await self._cancel_connection(name)
                    await self._disconnect_client(name)
                
                # Set status entries immediately for all servers in config (before connecting)
                # This ensures the TUI panel shows up right away
                for name in servers.keys():
                    if name not in self._servers:
                        self._servers[name] = {
                            "status": "connecting",
                            "error": None,
                            "tools_count": 0,
                            "config": None,
                            "active_config": None
                        }
                    # Update config with latest from file
                    self._servers[name]["config"] = servers[name].copy()
                
                # Update TUI immediately to show servers that are about to connect
                self._update_tui()
                
                # Add or update servers
                for name, server_config in servers.items():
                    if name in self.clients:
                        # Check if config changed
                        if self._config_changed(name, server_config):
                            if DEBUG >= 1:
                                print(f"[MCP] Config changed for server: {name}, reconnecting")
                            # Cancel any in-progress connection for this server
                            await self._cancel_connection(name)
                            await self._disconnect_client(name)
                            await self._connect_client(name, server_config)
                        else:
                            # Config unchanged, keep server running
                            if DEBUG >= 2:
                                print(f"[MCP] Config unchanged for server: {name}, keeping alive")
                    else:
                        # New server or not yet connected
                        # Cancel any in-progress connection before starting new one
                        await self._cancel_connection(name)
                        await self._connect_client(name, server_config)
                        
            except json.JSONDecodeError as e:
                if DEBUG >= 1:
                    print(f"[MCP] Invalid JSON in config file: {e}")
            except Exception as e:
                if DEBUG >= 1:
                    print(f"[MCP] Error reloading config: {e}")
            finally:
                # Always broadcast availability after reload attempt, even on error
                await self._broadcast_availability_if_changed()
    
    def _config_changed(self, name: str, new_config: Dict[str, Any]) -> bool:
        """Check if server config has changed by comparing relevant fields."""
        # Get the current active config for this server
        server_info = self._servers.get(name)
        if not server_info:
            return True
        old_config = server_info.get("active_config")
        
        # If no old config exists, consider it changed (new server)
        if old_config is None:
            return True
        
        # Compare relevant fields that would require a restart
        # For stdio transports: command, args, env
        # For HTTP/SSE transports: url, headers
        
        if "command" in new_config:
            # Stdio transport
            if old_config.get("command") != new_config.get("command"):
                return True
            if old_config.get("args", []) != new_config.get("args", []):
                return True
            # Compare env dicts
            old_env = old_config.get("env", {})
            new_env = new_config.get("env", {})
            if old_env != new_env:
                return True
        elif "url" in new_config:
            # HTTP/SSE transport
            if old_config.get("url") != new_config.get("url"):
                return True
            # Compare headers dicts
            old_headers = old_config.get("headers", {})
            new_headers = new_config.get("headers", {})
            if old_headers != new_headers:
                return True
        else:
            # Config structure changed (no command or url), consider it changed
            return True
        
        # No relevant changes detected, keep server running
        return False
    
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
        if name not in self._servers:
            self._servers[name] = {
                "status": "error",
                "error": error_msg,
                "tools_count": 0,
                "config": config.copy(),
                "active_config": None
            }
        else:
            self._servers[name]["status"] = "error"
            self._servers[name]["error"] = error_msg
            self._servers[name]["tools_count"] = 0
            self._servers[name]["config"] = config.copy()
        self._update_tui()
    
    async def _connect_client_impl(self, name: str, config: Dict[str, Any]) -> None:
        """Internal implementation of client connection."""
        if name not in self._servers:
            self._servers[name] = {
                "status": "connecting",
                "error": None,
                "tools_count": 0,
                "config": config.copy(),
                "active_config": None
            }
        else:
            self._servers[name]["status"] = "connecting"
            self._servers[name]["error"] = None
            self._servers[name]["config"] = config.copy()
        self._update_tui()
        
        transport = None
        try:
            # Check if we've been cancelled before starting
            if asyncio.current_task() and asyncio.current_task().cancelled():
                return
            
            # Determine transport type
            if "command" in config:
                transport = StdioTransport(
                    config["command"], 
                    config.get("args", []), 
                    config.get("env", {})
                )
            elif "url" in config:
                url, headers = config["url"], config.get("headers", {})

                if "/sse" in url.lower() or url.endswith("/sse"):
                    transport = SSETransport(url, headers)
                else:
                    transport = HTTPTransport(url, headers)
            else:
                error_msg = "Invalid server config: missing 'command' or 'url'"
                self._set_error_status(name, error_msg, config)
                
                if DEBUG >= 1:
                    print(f"[MCP] Invalid server config for {name}: {error_msg}")
                return
            
            # Create and initialize client
            self.clients[name] = MCPClient(name, transport)
            await self.clients[name].initialize()

            # Update status to connected
            if name not in self._servers:
                self._servers[name] = {
                    "status": "connected",
                    "error": None,
                    "tools_count": len(self.clients[name].tools),
                    "config": config.copy(),
                    "active_config": config.copy()
                }
            else:
                self._servers[name]["status"] = "connected"
                self._servers[name]["error"] = None
                self._servers[name]["tools_count"] = len(self.clients[name].tools)
                self._servers[name]["active_config"] = config.copy()
            self._update_tui()
            
            # Broadcast MCP availability if we have connected servers
            await self._broadcast_availability_if_changed()
            
            if DEBUG >= 1:
                print(f"[MCP] Connected to server: {name}")
                
        except asyncio.CancelledError:
            await self._cleanup_transport(transport)
            if DEBUG >= 2:
                print(f"[MCP] Connection cancelled for {name}")
            raise
        except (MCPClientError, Exception) as e:
            await self._cleanup_transport(transport)
            # Extract the actual error message, removing redundant prefixes
            if isinstance(e, MCPClientError):
                error_msg = str(e)
                # Remove redundant "Failed to initialize MCP client" prefix if present
                if error_msg.startswith("Failed to initialize MCP client "):
                    error_msg = error_msg[len("Failed to initialize MCP client "):]
            else:
                error_msg = f"Unexpected error: {str(e)}"
            
            # Add server-specific context to make errors more identifiable
            # Include command/url info if available
            if transport and hasattr(transport, 'command'):
                error_msg = f"{error_msg} (command: {transport.command})"
            elif transport and hasattr(transport, 'url'):
                error_msg = f"{error_msg} (url: {transport.url})"
            elif config:
                # Fallback to config info if transport wasn't created
                if "command" in config:
                    cmd_info = f"{config['command']} {' '.join(config.get('args', []))}"
                    error_msg = f"{error_msg} (command: {cmd_info})"
                elif "url" in config:
                    error_msg = f"{error_msg} (url: {config['url']})"
            
            self._set_error_status(name, error_msg, config)
            if DEBUG >= 1:
                print(f"[MCP] Failed to connect to server {name}: {error_msg}")
    
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
        
        # Remove server info
        self._servers.pop(name, None)
        self._update_tui()
        
        # Broadcast MCP availability if changed
        await self._broadcast_availability_if_changed()
    
    async def _disconnect_all(self) -> None:
        """Disconnect all clients."""
        for name in list(self.clients.keys()):
            await self._disconnect_client(name)
        # Broadcast that we no longer have MCP servers
        await self._broadcast_availability_if_changed()
    
    async def _broadcast_availability_if_changed(self) -> None:
        """Broadcast MCP server statuses if they have changed."""
        if not self.node:
            return
        
        # Build current server statuses
        current_servers = {}
        for name, info in self._servers.items():
            current_servers[name] = {
                "status": info.get("status", "unknown"),
                "error": info.get("error"),
                "tools_count": info.get("tools_count", 0)
            }
        
        # Check if status has changed
        if current_servers != self._last_broadcasted_servers:
            self._last_broadcasted_servers = current_servers.copy()
            try:
                await self.node.broadcast_mcp_status(current_servers)
                if DEBUG >= 1:
                    print(f"[MCP] Broadcasted MCP status for {len(current_servers)} server(s)")
            except Exception as e:
                if DEBUG >= 1:
                    print(f"[MCP] Error broadcasting MCP status: {e}")
    
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
            if server_name in self._servers:
                if self._servers[server_name]["status"] == "error":
                    self._servers[server_name]["status"] = "connected"
                    self._servers[server_name]["error"] = None
                    self._update_tui()
            return result
        except MCPClientError as e:
            error_msg = f"Tool call failed on {server_name}: {str(e)}"
            if server_name in self._servers:
                self._servers[server_name]["status"] = "error"
                self._servers[server_name]["error"] = error_msg
            self._update_tui_error(error_msg)
            raise
        except Exception as e:
            error_msg = f"Unexpected error calling tool {actual_tool_name} on {server_name}: {str(e)}"
            if server_name in self._servers:
                self._servers[server_name]["status"] = "error"
                self._servers[server_name]["error"] = error_msg
            self._update_tui_error(error_msg)
            raise MCPClientError(error_msg)
    
    def _update_tui(self) -> None:
        """Update TUI with MCP server status."""
        if self.topology_viz:
            try:
                # Extract status dict for TUI (only status, error, tools_count)
                status_dict = {
                    name: {
                        "status": info["status"],
                        "error": info.get("error"),
                        "tools_count": info.get("tools_count", 0)
                    }
                    for name, info in self._servers.items()
                }
                self.topology_viz.update_mcp_status(status_dict)
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
        """Load server config from _servers or config file."""
        # First check if we have it in _servers
        if name in self._servers and self._servers[name].get("config"):
            return self._servers[name]["config"]
        # Fallback to config file
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
                name for name, info in self._servers.items()
                if info.get("status") == "error" and name not in self.clients
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
                
                # Check for disconnected clients (process exited but client still exists)
                for name, client in list(self.clients.items()):
                    if hasattr(client, 'transport') and hasattr(client.transport, '_running'):
                        if not client.transport._running:
                            # Process exited, mark as failed and clean up
                            if DEBUG >= 1:
                                print(f"[MCP] Detected disconnected client for {name}, marking as failed")
                            # Clean up the client
                            self.clients.pop(name)
                            asyncio.create_task(client.disconnect())
                            # Mark as failed if we have config
                            if name in self._servers and self._servers[name].get("config"):
                                self._set_error_status(name, "Process exited unexpectedly", self._servers[name]["config"])
                
                failed_names = [
                    name for name, info in self._servers.items()
                    if info.get("status") == "error" and name not in self.clients
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

