"""MCP transport implementations for stdio, HTTP, and SSE."""

import asyncio
import json
import os
import subprocess
import sys
from typing import Dict, Optional, Any, AsyncIterator
from pathlib import Path
import aiohttp
from exo import DEBUG


class TransportError(Exception):
    """Base exception for transport errors."""
    pass


def _format_error(error: Any) -> str:
    """Extract error message from JSON-RPC error format."""
    if isinstance(error, dict):
        error_msg = error.get("message", str(error))
        error_code = error.get("code", "")
        if error_code:
            error_msg = f"[{error_code}] {error_msg}"
        return error_msg
    return str(error)


class StdioTransport:
    """MCP transport over stdio (subprocess)."""
    
    def __init__(self, command: str, args: list[str], env: Optional[Dict[str, str]] = None):
        self.command = command
        self.args = [arg.replace("~", str(Path.home())) for arg in args]  # Expand ~
        self.env = env or {}
        self.process: Optional[subprocess.Popen] = None
        self._read_task: Optional[asyncio.Task] = None
        self._message_queue: asyncio.Queue = asyncio.Queue()
        self._request_id = 0
        self._pending_requests: Dict[int, asyncio.Future] = {}
        self._running = False
        self._is_mcp_remote = self._detect_mcp_remote()
    
    def _detect_mcp_remote(self) -> bool:
        """Detect if this transport is running mcp-remote."""
        # Check if command or args contain mcp-remote
        if 'mcp-remote' in self.command.lower():
            return True
        for arg in self.args:
            if 'mcp-remote' in str(arg).lower():
                return True
        return False
        
    async def connect(self) -> None:
        """Start the subprocess and establish stdio communication."""
        if self._running:
            return
            
        try:
            # Merge with current environment
            full_env = {**os.environ, **self.env}
            
            self.process = await asyncio.create_subprocess_exec(
                self.command,
                *self.args,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=full_env
            )
            
            # MCP server processes should be long-running. If they exit, we'll detect
            # it in the process monitor and trigger a retry.
            self._running = True
            self._read_task = asyncio.create_task(self._read_loop())
            
            # Start message processing
            asyncio.create_task(self._process_messages())
            
            # Start stderr monitoring
            asyncio.create_task(self._monitor_stderr())
            
            # Start process exit monitoring
            asyncio.create_task(self._monitor_process())
            
        except Exception as e:
            self._running = False
            raise TransportError(f"{e}")
    
    async def _read_loop(self) -> None:
        """Read messages from stdout."""
        if not self.process or not self.process.stdout:
            return
            
        buffer = ""
        try:
            while self._running and self.process:
                chunk = await self.process.stdout.read(4096)
                if not chunk:
                    # EOF - process exited
                    break
                    
                buffer += chunk.decode('utf-8', errors='replace')
                
                # Process complete JSON-RPC messages
                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    line = line.strip()
                    if not line:
                        continue
                        
                    try:
                        message = json.loads(line)
                        await self._message_queue.put(message)
                    except json.JSONDecodeError:
                        if DEBUG >= 2:
                            print(f"[MCP] Failed to parse JSON: {line}")
            
            # Process exited (EOF detected), fail pending requests
            if self._running and self.process:
                self._running = False
                returncode = self.process.returncode if self.process.returncode is not None else await self.process.wait()
                error_msg = f"Process exited with code {returncode}"
                for request_id, future in list(self._pending_requests.items()):
                    if not future.done():
                        future.set_exception(TransportError(error_msg))
                        self._pending_requests.pop(request_id, None)
                
                            
        except Exception as e:
            if DEBUG >= 1:
                print(f"[MCP] Read loop error: {e}")
            self._running = False
            # Fail pending requests on error
            error_msg = f"Read loop error: {str(e)}"
            for request_id, future in list(self._pending_requests.items()):
                if not future.done():
                    future.set_exception(TransportError(error_msg))
                    self._pending_requests.pop(request_id, None)
    
    async def _monitor_stderr(self) -> None:
        """Monitor stderr for important messages."""
        if not self.process or not self.process.stderr:
            return
            
        try:
            while self._running and self.process:
                line = await self.process.stderr.readline()
                if not line:
                    break
                    
                line_str = line.decode('utf-8', errors='replace').strip()
                if line_str and DEBUG >= 2:
                    print(f"[MCP stderr] {line_str}")
        except Exception as e:
            if DEBUG >= 2:
                print(f"[MCP] Stderr monitor error: {e}")
    
    async def _monitor_process(self) -> None:
        """Monitor process exit and fail pending requests."""
        if not self.process:
            return
            
        try:
            # Wait for process to exit
            returncode = await self.process.wait()
            
            # Process exited, mark as not running
            self._running = False
            
            # Fail all pending requests
            error_msg = f"Process exited with code {returncode}"
            for request_id, future in list(self._pending_requests.items()):
                if not future.done():
                    future.set_exception(TransportError(error_msg))
                    self._pending_requests.pop(request_id, None)
            
            if DEBUG >= 1:
                print(f"[MCP] Process exited with code {returncode}")
                
        except Exception as e:
            if DEBUG >= 2:
                print(f"[MCP] Process monitor error: {e}")
    
    async def send_notification(self, method: str, params: Optional[Dict[str, Any]] = None) -> None:
        """Send a JSON-RPC notification (no response expected)."""
        if not self._running or not self.process or not self.process.stdin:
            raise TransportError("Transport not connected")
        
        request = {
            "jsonrpc": "2.0",
            "method": method,
        }
        if params:
            request["params"] = params
        
        try:
            request_json = json.dumps(request) + '\n'
            self.process.stdin.write(request_json.encode('utf-8'))
            await self.process.stdin.drain()
        except Exception as e:
            raise TransportError(f"Failed to send notification: {e}")
    
    async def send_request(self, method: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Send a JSON-RPC request and wait for response."""
        if not self._running or not self.process or not self.process.stdin:
            raise TransportError("Transport not connected")
        
        self._request_id += 1
        request_id = self._request_id
        
        request = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
        }
        if params:
            request["params"] = params
        
        future = asyncio.Future()
        self._pending_requests[request_id] = future
        
        try:
            request_json = json.dumps(request) + '\n'
            self.process.stdin.write(request_json.encode('utf-8'))
            await self.process.stdin.drain()
            
            # Wait for response with timeout
            # For mcp-remote, wait indefinitely for initialize requests (OAuth may take time)
            # Other requests and non-mcp-remote commands use 10s timeout for initialization
            if self._is_mcp_remote and method == "initialize":
                # Wait indefinitely for mcp-remote initialization
                response = await future
            else:
                timeout = 10.0 if method == "initialize" else 30.0
                response = await asyncio.wait_for(future, timeout=timeout)
            return response
        except asyncio.TimeoutError:
            self._pending_requests.pop(request_id, None)
            raise TransportError(f"Request timeout for {method}")
        except Exception as e:
            self._pending_requests.pop(request_id, None)
            raise TransportError(f"Failed to send request: {e}")
    
    async def _process_messages(self) -> None:
        """Process incoming messages from the queue."""
        while self._running:
            try:
                message = await asyncio.wait_for(self._message_queue.get(), timeout=1.0)
                
                # Handle responses
                if "id" in message and message["id"] in self._pending_requests:
                    future = self._pending_requests.pop(message["id"])
                    if "error" in message:
                        future.set_exception(TransportError(_format_error(message["error"])))
                    else:
                        future.set_result(message.get("result", {}))
                        
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                if DEBUG >= 2:
                    print(f"[MCP] Message processing error: {e}")
    
    async def disconnect(self) -> None:
        """Close the transport connection."""
        self._running = False
        
        if self._read_task:
            self._read_task.cancel()
            try:
                await self._read_task
            except asyncio.CancelledError:
                pass
        
        if self.process:
            try:
                self.process.terminate()
                await asyncio.wait_for(self.process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                self.process.kill()
                await self.process.wait()
            except Exception as e:
                if DEBUG >= 2:
                    print(f"[MCP] Error disconnecting stdio: {e}")
            finally:
                self.process = None
        
        # Cancel pending requests
        for future in self._pending_requests.values():
            if not future.done():
                future.set_exception(TransportError("Transport disconnected"))
        self._pending_requests.clear()


class HTTPTransport:
    """MCP transport over HTTP."""
    
    def __init__(self, url: str, headers: Optional[Dict[str, str]] = None):
        self.url = url
        self.headers = headers or {}
        self.session: Optional[aiohttp.ClientSession] = None
        
    async def connect(self) -> None:
        """Establish HTTP connection."""
        self.session = aiohttp.ClientSession()
    
    async def send_notification(self, method: str, params: Optional[Dict[str, Any]] = None) -> None:
        """Send a JSON-RPC notification (no response expected)."""
        if not self.session:
            raise TransportError("Transport not connected")
        
        request = {
            "jsonrpc": "2.0",
            "method": method,
        }
        if params:
            request["params"] = params
        
        try:
            async with self.session.post(
                self.url,
                json=request,
                headers=self.headers,
                timeout=aiohttp.ClientTimeout(total=5)
            ) as response:
                # Notifications don't require a response, but we check for errors
                if response.status != 200:
                    if DEBUG >= 2:
                        print(f"[MCP] Notification {method} returned status {response.status}")
        except aiohttp.ClientError as e:
            if DEBUG >= 2:
                print(f"[MCP] Notification {method} failed: {e}")
        
    async def send_request(self, method: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Send a JSON-RPC request over HTTP."""
        if not self.session:
            raise TransportError("Transport not connected")
        
        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": method,
        }
        if params:
            request["params"] = params
        
        try:
            # Use 10s timeout for initialize requests, 30s for others
            timeout = 10 if method == "initialize" else 30
            async with self.session.post(
                self.url,
                json=request,
                headers=self.headers,
                timeout=aiohttp.ClientTimeout(total=timeout)
            ) as response:
                if response.status != 200:
                    raise TransportError(f"HTTP error {response.status}: {await response.text()}")
                
                result = await response.json()
                if "error" in result:
                    raise TransportError(_format_error(result["error"]))
                return result.get("result", {})
        except aiohttp.ClientError as e:
            raise TransportError(f"HTTP request failed: {e}")
    
    async def disconnect(self) -> None:
        """Close the HTTP connection."""
        if self.session:
            await self.session.close()
            self.session = None


class SSETransport:
    """MCP transport over Server-Sent Events (SSE)."""
    
    def __init__(self, url: str, headers: Optional[Dict[str, str]] = None):
        self.url = url
        self.headers = headers or {}
        self.session: Optional[aiohttp.ClientSession] = None
        self._request_id = 0
        self._pending_requests: Dict[int, asyncio.Future] = {}
        self._sse_task: Optional[asyncio.Task] = None
        self._running = False
        
    async def connect(self) -> None:
        """Establish SSE connection."""
        self.session = aiohttp.ClientSession()
        self._running = True
        self._sse_task = asyncio.create_task(self._sse_loop())
    
    async def send_notification(self, method: str, params: Optional[Dict[str, Any]] = None) -> None:
        """Send a JSON-RPC notification (no response expected)."""
        if not self.session or not self._running:
            raise TransportError("Transport not connected")
        
        request = {
            "jsonrpc": "2.0",
            "method": method,
        }
        if params:
            request["params"] = params
        
        try:
            # For SSE, POST notifications to the same endpoint
            post_url = self.url.replace('/sse', '') if '/sse' in self.url else self.url
            
            async with self.session.post(
                post_url,
                json=request,
                headers=self.headers,
                timeout=aiohttp.ClientTimeout(total=5)
            ) as response:
                # Notifications don't require a response, but we check for errors
                if response.status != 200:
                    if DEBUG >= 2:
                        print(f"[MCP] Notification {method} returned status {response.status}")
        except aiohttp.ClientError as e:
            if DEBUG >= 2:
                print(f"[MCP] Notification {method} failed: {e}")
        
    async def _sse_loop(self) -> None:
        """Read SSE events."""
        if not self.session:
            return
            
        try:
            async with self.session.get(
                self.url,
                headers={**self.headers, "Accept": "text/event-stream"},
                timeout=aiohttp.ClientTimeout(total=None)
            ) as response:
                if response.status != 200:
                    raise TransportError(f"SSE connection error {response.status}")
                
                async for line in response.content:
                    line_str = line.decode('utf-8', errors='replace').strip()
                    if line_str.startswith('data: '):
                        data_str = line_str[6:]  # Remove 'data: ' prefix
                        try:
                            message = json.loads(data_str)
                            await self._handle_message(message)
                        except json.JSONDecodeError:
                            if DEBUG >= 2:
                                print(f"[MCP] Failed to parse SSE JSON: {data_str}")
        except Exception as e:
            if DEBUG >= 1:
                print(f"[MCP] SSE loop error: {e}")
            self._running = False
    
    async def _handle_message(self, message: Dict[str, Any]) -> None:
        """Handle incoming SSE message."""
        if "id" in message and message["id"] in self._pending_requests:
            future = self._pending_requests.pop(message["id"])
            if "error" in message:
                future.set_exception(TransportError(_format_error(message["error"])))
            else:
                future.set_result(message.get("result", {}))
    
    async def send_request(self, method: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Send a JSON-RPC request over SSE (via HTTP POST)."""
        if not self.session or not self._running:
            raise TransportError("Transport not connected")
        
        self._request_id += 1
        request_id = self._request_id
        
        request = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
        }
        if params:
            request["params"] = params
        
        future = asyncio.Future()
        self._pending_requests[request_id] = future
        
        try:
            # For SSE, we typically POST requests to a separate endpoint
            # Use the same URL but POST instead of GET
            post_url = self.url.replace('/sse', '') if '/sse' in self.url else self.url
            
            async with self.session.post(
                post_url,
                json=request,
                headers=self.headers,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status != 200:
                    raise TransportError(f"HTTP error {response.status}: {await response.text()}")
                
                result = await response.json()
                if "error" in result:
                    future.set_exception(TransportError(_format_error(result["error"])))
                else:
                    future.set_result(result.get("result", {}))
            
            # Wait for response
            # Use 10s timeout for initialize requests, 30s for others
            timeout = 10 if method == "initialize" else 30
            response = await asyncio.wait_for(future, timeout=timeout)
            return response
        except asyncio.TimeoutError:
            self._pending_requests.pop(request_id, None)
            raise TransportError(f"Request timeout for {method}")
        except Exception as e:
            self._pending_requests.pop(request_id, None)
            raise TransportError(f"Failed to send request: {e}")
    
    async def disconnect(self) -> None:
        """Close the SSE connection."""
        self._running = False
        
        if self._sse_task:
            self._sse_task.cancel()
            try:
                await self._sse_task
            except asyncio.CancelledError:
                pass
        
        if self.session:
            await self.session.close()
            self.session = None
        
        # Cancel pending requests
        for future in self._pending_requests.values():
            if not future.done():
                future.set_exception(TransportError("Transport disconnected"))
        self._pending_requests.clear()

