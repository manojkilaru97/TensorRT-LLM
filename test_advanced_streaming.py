#!/usr/bin/env python3
"""
Enhanced test script for streaming tool calls with DeepSeek R1.
Demonstrates multiple tool calls and tool call continuation scenarios
"""

import json
import requests
import time
from typing import Dict, List, Any
from dataclasses import dataclass


@dataclass
class ToolCallState:
    """Track state of individual tool calls"""
    id: str = None
    name: str = None
    arguments: str = ""
    completed: bool = False


class StreamingToolCallTester:
    """Enhanced tester for streaming tool calls"""
    
    def __init__(self, base_url: str = "http://127.0.0.1:8003"):
        self.base_url = base_url
        self.endpoint = f"{base_url}/v1/chat/completions"
    
    def simulate_tool_responses(self, tool_calls: List[Dict]) -> List[Dict]:
        """Simulate tool responses for demonstration"""
        responses = []
        
        for tool_call in tool_calls:
            func_name = tool_call["function"]["name"]
            try:
                args = json.loads(tool_call["function"]["arguments"])
                
                if func_name == "get_weather":
                    location = args.get("location", "Unknown")
                    response = {
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "content": json.dumps({
                            "location": location,
                            "temperature": "22°C",
                            "condition": "Sunny",
                            "humidity": "65%",
                            "wind": "5 mph"
                        })
                    }
                elif func_name == "calculate":
                    expression = args.get("expression", "0")
                    try:
                        # Simple eval for demo (don't use in production!)
                        result = eval(expression.replace("*", "*").replace("x", "*"))
                        response = {
                            "role": "tool",
                            "tool_call_id": tool_call["id"],
                            "content": json.dumps({
                                "expression": expression,
                                "result": result
                            })
                        }
                    except:
                        response = {
                            "role": "tool",
                            "tool_call_id": tool_call["id"],
                            "content": json.dumps({"error": "Invalid expression"})
                        }
                elif func_name == "search_web":
                    query = args.get("query", "")
                    response = {
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "content": json.dumps({
                            "query": query,
                            "results": [
                                {"title": f"Result 1 for {query}", "url": "https://example1.com"},
                                {"title": f"Result 2 for {query}", "url": "https://example2.com"}
                            ]
                        })
                    }
                else:
                    response = {
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "content": json.dumps({"error": "Unknown tool"})
                    }
                
                responses.append(response)
            except Exception as e:
                responses.append({
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": json.dumps({"error": str(e)})
                })
        
        return responses
    
    def test_streaming_with_conversation(self):
        """Test a full conversation with tool calls and responses"""
        
        print("🔄 FULL CONVERSATION TEST")
        print("=" * 60)
        
        # Initial request
        messages = [
            {"role": "user", "content": "Get weather for Tokyo and calculate 15 * 23"}
        ]
        
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get current weather information for a location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The city and state, e.g. San Francisco, CA"
                            },
                            "unit": {
                                "type": "string",
                                "enum": ["celsius", "fahrenheit"],
                                "description": "Temperature unit"
                            }
                        },
                        "required": ["location"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "calculate",
                    "description": "Perform mathematical calculations",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "expression": {
                                "type": "string",
                                "description": "Mathematical expression to evaluate"
                            }
                        },
                        "required": ["expression"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "search_web",
                    "description": "Search the web for information",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query"
                            },
                            "max_results": {
                                "type": "integer",
                                "description": "Maximum number of results to return",
                                "default": 5
                            }
                        },
                        "required": ["query"]
                    }
                }
            }
        ]
        
        conversation_round = 1
        
        while True:
            print(f"\n🔸 ROUND {conversation_round}")
            print("-" * 40)
            
            # Show current conversation
            print("💬 Current Conversation:")
            for i, msg in enumerate(messages[-3:], 1):  # Show last 3 messages
                role_emoji = {"user": "👤", "assistant": "🤖", "tool": "🔧"}
                print(f"  {role_emoji.get(msg['role'], '❓')} {msg['role'].title()}: {msg.get('content', 'N/A')[:100]}...")
                if msg.get('tool_calls'):
                    for tc in msg['tool_calls']:
                        print(f"    🛠️  {tc['function']['name']}({tc['function']['arguments']})")
            
            # Make request
            payload = {
                "model": "DeepSeek-R1-0528-FP4",
                "messages": messages,
                "temperature": 0.1,
                "stream": True,
                "tools": tools,
                "max_tokens": 4096,
                "add_generation_prompt": True
            }
            
            print(f"\n📡 Streaming response for round {conversation_round}...")
            
            assistant_response = self.stream_and_parse(payload)
            messages.append(assistant_response)
            
            # If tool calls were made, simulate responses
            if assistant_response.get("tool_calls"):
                print(f"\n🔧 Simulating {len(assistant_response['tool_calls'])} tool response(s)...")
                
                tool_responses = self.simulate_tool_responses(assistant_response["tool_calls"])
                for response in tool_responses:
                    messages.append(response)
                    print(f"  ✅ Tool {response['tool_call_id']}: {response['content'][:100]}...")
                
                conversation_round += 1
                
                # Continue conversation to get final response
                if conversation_round <= 3:  # Limit rounds
                    continue
            
            break
        
        print(f"\n🎯 FINAL CONVERSATION ({len(messages)} messages):")
        print("=" * 50)
        for i, msg in enumerate(messages, 1):
            role_emoji = {"user": "👤", "assistant": "🤖", "tool": "🔧"}
            print(f"{i}. {role_emoji.get(msg['role'], '❓')} {msg['role'].title()}:")
            if msg.get('content'):
                print(f"   💬 {msg['content']}")
            if msg.get('reasoning_content'):
                print(f"   🧠 Reasoning: {len(msg['reasoning_content'])} chars")
            if msg.get('tool_calls'):
                for tc in msg['tool_calls']:
                    print(f"   🛠️  {tc['function']['name']}({tc['function']['arguments']})")
            print()
    
    def stream_and_parse(self, payload: Dict) -> Dict:
        """Stream a request and parse the complete response"""
        
        combined_response = {
            "role": "assistant",
            "content": "",
            "reasoning_content": "",
            "tool_calls": []
        }
        
        current_tool_calls = {}
        chunk_count = 0
        
        try:
            response = requests.post(
                self.endpoint,
                headers={"Content-Type": "application/json"},
                json=payload,
                stream=True,
                timeout=60
            )
            response.raise_for_status()
            
            for line in response.iter_lines():
                if line:
                    line_str = line.decode('utf-8')
                    
                    if not line_str.strip() or line_str.strip() == "data: [DONE]":
                        continue
                    
                    if line_str.startswith("data: "):
                        chunk_count += 1
                        json_str = line_str[6:]
                        
                        try:
                            chunk = json.loads(json_str)
                            choice = chunk.get("choices", [{}])[0]
                            delta = choice.get("delta", {})
                            finish_reason = choice.get("finish_reason")
                            
                            # Handle role
                            if delta.get("role"):
                                combined_response["role"] = delta["role"]
                            
                            # Handle content
                            if delta.get("content") is not None:
                                combined_response["content"] += delta["content"]
                                if delta["content"]:  # Only print non-empty content
                                    print(f"{delta['content']}", end="", flush=True)
                            
                            # Handle reasoning content
                            if delta.get("reasoning_content") is not None:
                                combined_response["reasoning_content"] += delta["reasoning_content"]
                                if delta["reasoning_content"]:
                                    print(f"🧠", end="", flush=True)
                            
                            # Handle tool calls
                            if delta.get("tool_calls"):
                                for tool_call_delta in delta["tool_calls"]:
                                    # Get tool call ID to track across chunks
                                    tool_id = tool_call_delta.get("id")
                                    
                                    if tool_id and tool_id not in current_tool_calls:
                                        current_tool_calls[tool_id] = {
                                            "id": tool_id,
                                            "type": "function",
                                            "function": {"name": None, "arguments": ""}
                                        }
                                        print(f"\n   🔧 Tool Call {tool_id[:8]}... started")
                                    
                                    # Update tool call
                                    if tool_id and tool_id in current_tool_calls:
                                        if tool_call_delta.get("function"):
                                            func_delta = tool_call_delta["function"]
                                            if func_delta.get("name"):
                                                current_tool_calls[tool_id]["function"]["name"] = func_delta["name"]
                                                print(f"      📛 Name: {func_delta['name']}")
                                            if func_delta.get("arguments") is not None:
                                                current_tool_calls[tool_id]["function"]["arguments"] += func_delta["arguments"]
                                                if func_delta["arguments"]:  # Only print non-empty arguments
                                                    print(f"{func_delta['arguments']}", end="", flush=True)
                            
                            # Handle finish reason
                            if finish_reason:
                                print(f"\n   🏁 Finished: {finish_reason}")
                        
                        except json.JSONDecodeError:
                            continue
            
            # Finalize combined response
            combined_response["tool_calls"] = list(current_tool_calls.values())
            
            print(f"\n✅ Response completed: {len(combined_response['content'])} chars content, {len(combined_response['reasoning_content'])} chars reasoning, {len(combined_response['tool_calls'])} tool calls")
            
            return combined_response
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return combined_response
    
    def test_reasoning_with_tools(self):
        """Test reasoning mode with tool calls"""
        
        print("\n🧠 REASONING + TOOLS TEST")
        print("=" * 50)
        
        payload = {
            "model": "DeepSeek-R1-0528-FP4",
            "messages": [
                {"role": "user", "content": "I need to know the weather in Tokyo to plan my trip. Can you help?"}
            ],
            "temperature": 0.1,
            "stream": True,
            "max_tokens": 4096,
            "add_generation_prompt": True,
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get current weather information for a location",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "location": {
                                    "type": "string",
                                    "description": "The city and state, e.g. San Francisco, CA"
                                },
                                "unit": {
                                    "type": "string",
                                    "enum": ["celsius", "fahrenheit"],
                                    "description": "Temperature unit"
                                }
                            },
                            "required": ["location"]
                        }
                    }
                }
            ]
        }
        
        print("🤔 Testing reasoning with tools...")
        response = self.stream_and_parse(payload)
        
        print("\n📋 Final Response Summary:")
        print(f"   Content: {response['content'][:100]}...")
        print(f"   Reasoning: {len(response.get('reasoning_content', ''))} chars")
        print(f"   Tool Calls: {len(response.get('tool_calls', []))}")
        
        return response

    def test_complex_multi_tool_scenario(self):
        """Test complex scenario with multiple tools"""
        
        print("\n🎯 COMPLEX MULTI-TOOL SCENARIO")
        print("=" * 50)
        
        payload = {
            "model": "DeepSeek-R1-0528-FP4",
            "messages": [
                {"role": "user", "content": "Help me plan a trip: get weather for Paris, calculate travel budget (500 * 7 days), and search for 'best restaurants in Paris'"}
            ],
            "temperature": 0.1,
            "stream": True,
            "max_tokens": 4096,
            "add_generation_prompt": True,
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get current weather information for a location",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "location": {
                                    "type": "string",
                                    "description": "The city and state, e.g. San Francisco, CA"
                                },
                                "unit": {
                                    "type": "string",
                                    "enum": ["celsius", "fahrenheit"],
                                    "description": "Temperature unit"
                                }
                            },
                            "required": ["location"]
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "calculate",
                        "description": "Perform mathematical calculations",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "expression": {
                                    "type": "string",
                                    "description": "Mathematical expression to evaluate"
                                }
                            },
                            "required": ["expression"]
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "search_web",
                        "description": "Search the web for information",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "description": "Search query"
                                },
                                "max_results": {
                                    "type": "integer",
                                    "description": "Maximum number of results to return",
                                    "default": 5
                                }
                            },
                            "required": ["query"]
                        }
                    }
                }
            ]
        }
        
        print("🎪 Testing complex multi-tool scenario...")
        response = self.stream_and_parse(payload)
        
        print("\n📊 Complex Scenario Results:")
        print(f"   Content: {len(response['content'])} chars")
        print(f"   Reasoning: {len(response.get('reasoning_content', ''))} chars")
        print(f"   Tool Calls: {len(response.get('tool_calls', []))}")
        
        if response.get('tool_calls'):
            print("   🛠️  Tool Call Details:")
            for i, tc in enumerate(response['tool_calls'], 1):
                print(f"     {i}. {tc['function']['name']}: {tc['function']['arguments']}")
        
        return response


def main():
    print("🚀 Enhanced DeepSeek R1 Streaming Tool Call Test Suite")
    print("=" * 70)
    
    tester = StreamingToolCallTester()
    
    # Test health first
    try:
        health_response = requests.get("http://127.0.0.1:8003/health", timeout=5)
        print(f"🏥 Server Health: {health_response.status_code}")
    except Exception as e:
        print(f"❌ Server not available: {e}")
        return
    
    # Test 1: Reasoning with tools
    tester.test_reasoning_with_tools()
    
    time.sleep(2)
    
    # Test 2: Complex multi-tool scenario
    tester.test_complex_multi_tool_scenario()
    
    time.sleep(2)
    
    # Test 3: Full conversation flow
    tester.test_streaming_with_conversation()
    
    print("\n🎉 All enhanced tests completed!")


if __name__ == "__main__":
    main()
