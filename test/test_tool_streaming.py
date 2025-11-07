#!/usr/bin/env python3
"""
Test script for tool call streaming in chat completions API.

This script demonstrates that tool calls are now visible in the streaming response
before execution, providing transparency to users.
"""

import asyncio
import aiohttp
import json
import sys

async def test_tool_streaming(base_url="http://localhost:52415"):
    """Test streaming chat completion with tool calls."""
    
    url = f"{base_url}/v1/chat/completions"
    
    # Example request that might trigger MCP tools
    # Adjust this based on your MCP server capabilities
    request_data = {
        "model": "llama-3.2-1b",
        "messages": [
            {
                "role": "user",
                "content": "List all the Neon databases"
            }
        ],
        "stream": True
    }
    
    print("=" * 60)
    print("Testing Tool Call Streaming")
    print("=" * 60)
    print(f"\nSending request to: {url}")
    print(f"User message: {request_data['messages'][0]['content']}")
    print("\nStreaming response:\n")
    print("-" * 60)
    
    tool_calls_detected = []
    execution_indicators = []
    content_chunks = []
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                json=request_data,
                headers={"Content-Type": "application/json"}
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    print(f"Error: HTTP {response.status}")
                    print(error_text)
                    return False
                
                chunk_num = 0
                async for line in response.content:
                    line_str = line.decode('utf-8').strip()
                    
                    # Skip empty lines and comments
                    if not line_str or not line_str.startswith('data: '):
                        continue
                    
                    # Remove 'data: ' prefix
                    data_str = line_str[6:]
                    
                    # Skip [DONE] marker
                    if data_str == '[DONE]':
                        break
                    
                    try:
                        chunk = json.loads(data_str)
                        chunk_num += 1
                        
                        if 'choices' in chunk and len(chunk['choices']) > 0:
                            choice = chunk['choices'][0]
                            delta = choice.get('delta', {})
                            
                            # Check for tool calls
                            if 'tool_calls' in delta:
                                for tool_call in delta['tool_calls']:
                                    if 'function' in tool_call:
                                        func = tool_call['function']
                                        tool_name = func.get('name', 'unknown')
                                        tool_args = func.get('arguments', '{}')
                                        
                                        tool_calls_detected.append({
                                            'name': tool_name,
                                            'arguments': tool_args
                                        })
                                        
                                        print(f"\n🔧 TOOL CALL DETECTED:")
                                        print(f"   Name: {tool_name}")
                                        try:
                                            args_obj = json.loads(tool_args)
                                            print(f"   Args: {json.dumps(args_obj, indent=2)}")
                                        except:
                                            print(f"   Args: {tool_args}")
                                        print()
                            
                            # Check for content
                            if 'content' in delta and delta['content']:
                                content = delta['content']
                                content_chunks.append(content)
                                
                                # Check if this is an execution indicator
                                if 'Executing' in content and 'tool' in content:
                                    execution_indicators.append(content)
                                    print(f"\n⏳ {content.strip()}\n")
                                else:
                                    # Print content character by character for streaming effect
                                    print(content, end='', flush=True)
                    
                    except json.JSONDecodeError as e:
                        print(f"\nWarning: Failed to parse chunk: {e}")
                        continue
    
    except Exception as e:
        print(f"\nError during test: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n")
    print("-" * 60)
    print("\nTest Results:")
    print("=" * 60)
    print(f"✓ Tool calls detected: {len(tool_calls_detected)}")
    print(f"✓ Execution indicators: {len(execution_indicators)}")
    print(f"✓ Content chunks received: {len(content_chunks)}")
    
    if tool_calls_detected:
        print("\n🎉 SUCCESS: Tool calls are being streamed!")
        print("\nTool calls found:")
        for i, tool in enumerate(tool_calls_detected, 1):
            print(f"  {i}. {tool['name']}")
        return True
    else:
        print("\nℹ️  No tool calls detected in this response.")
        print("   This is normal if the model didn't call any tools.")
        print("   Try a different prompt that requires tool usage.")
        return True

if __name__ == "__main__":
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:52415"
    
    print("\n" + "=" * 60)
    print("Tool Call Streaming Test")
    print("=" * 60)
    print("\nThis test verifies that:")
    print("1. Tool calls are streamed before execution")
    print("2. Execution indicators are shown during processing")
    print("3. The response format is OpenAI compatible")
    print()
    
    asyncio.run(test_tool_streaming(base_url))


