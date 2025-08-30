#!/usr/bin/env python3
"""
Multi-turn conversation testing for DeepSeek R1 tool calling.

Tests the complete flow of tool calling in conversations including:
1. User request -> Tool call
2. Tool response -> Assistant response  
3. Follow-up questions
4. Mixed tool and non-tool responses
"""

import json
import requests
from typing import Dict, List, Any

SERVER = "http://0.0.0.0:8003"
CHAT_URL = f"{SERVER}/v1/chat/completions"

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather information for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "The city and state"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
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
                    "expression": {"type": "string", "description": "Mathematical expression"}
                },
                "required": ["expression"]
            }
        }
    }
]

def make_request(messages: List[Dict], tools: List[Dict] = None) -> Dict[str, Any]:
    """Make a non-streaming request (more reliable for multi-turn)."""
    payload = {
        "model": "DeepSeek-R1-0528-FP4",
        "messages": messages,
        "stream": False,
        "tools": tools,
        "max_tokens": 4096,
        "temperature": 0.1,
    }
    
    response = requests.post(CHAT_URL, json=payload)
    if response.status_code != 200:
        return {"error": f"HTTP {response.status_code}: {response.text}"}
    
    result = response.json()
    if 'choices' in result and len(result['choices']) > 0:
        message = result['choices'][0]['message']
        return {
            "content": message.get('content', ''),
            "reasoning_content": message.get('reasoning_content', ''),
            "tool_calls": message.get('tool_calls', []),
            "finish_reason": result['choices'][0].get('finish_reason')
        }
    
    return {"content": "", "reasoning_content": "", "tool_calls": []}

def test_complete_tool_flow():
    """Test complete tool calling flow with responses."""
    print("=== TESTING COMPLETE TOOL FLOW ===")
    
    # Start conversation
    messages = [{"role": "user", "content": "What's the weather in Paris? Use Celsius."}]
    
    print("1. Initial request:")
    print(f"   User: {messages[0]['content']}")
    
    # Get tool call
    result = make_request(messages, TOOLS)
    
    if result.get('tool_calls'):
        tool_call = result['tool_calls'][0]
        print(f"   Assistant calls: {tool_call['function']['name']}")
        print(f"   Arguments: {tool_call['function']['arguments']}")
        
        # Add assistant message with tool call
        messages.append({
            "role": "assistant",
            "content": result.get('content', ''),
            "tool_calls": result['tool_calls']
        })
        
        # Simulate tool response
        tool_response = {
            "location": "Paris, France",
            "temperature": 18,
            "unit": "celsius", 
            "description": "Partly cloudy",
            "humidity": 65
        }
        
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call['id'],
            "content": json.dumps(tool_response)
        })
        
        print("2. Tool response provided")
        
        # Get assistant's final response
        result2 = make_request(messages, TOOLS)
        print(f"   Assistant: {result2.get('content', '')[:100]}...")
        
        # Continue conversation
        messages.append({
            "role": "assistant", 
            "content": result2.get('content', '')
        })
        
        messages.append({
            "role": "user",
            "content": "That's nice! Can you also calculate what 18 degrees Celsius is in Fahrenheit?"
        })
        
        print("3. Follow-up request:")
        print(f"   User: {messages[-1]['content']}")
        
        result3 = make_request(messages, TOOLS)
        
        if result3.get('tool_calls'):
            calc_call = result3['tool_calls'][0]
            print(f"   Assistant calls: {calc_call['function']['name']}")
            print(f"   Arguments: {calc_call['function']['arguments']}")
            
            return True
        else:
            print("   ❌ No tool call made for calculation")
            return False
    else:
        print("   ❌ No initial tool call made")
        return False

def test_mixed_conversation():
    """Test conversation mixing tool calls and regular responses."""
    print("\n=== TESTING MIXED CONVERSATION ===")
    
    messages = [
        {"role": "user", "content": "Hi! Tell me about artificial intelligence."}
    ]
    
    # Regular response (no tools)
    result1 = make_request(messages, TOOLS)
    print(f"1. Regular response: {len(result1.get('content', ''))} chars")
    print(f"   Tool calls: {len(result1.get('tool_calls', []))}")
    
    messages.append({"role": "assistant", "content": result1.get('content', '')})
    messages.append({"role": "user", "content": "Now calculate 42 * 1.5"})
    
    # Tool call response
    result2 = make_request(messages, TOOLS)
    print(f"2. Tool call response: {len(result2.get('tool_calls', []))} calls")
    
    if result2.get('tool_calls'):
        # Simulate tool execution
        messages.append({
            "role": "assistant",
            "tool_calls": result2['tool_calls']
        })
        messages.append({
            "role": "tool", 
            "tool_call_id": result2['tool_calls'][0]['id'],
            "content": "63.0"
        })
        
        # Get final response
        result3 = make_request(messages, TOOLS)
        print(f"3. Final response: {len(result3.get('content', ''))} chars")
        
        return True
    
    return False

def test_error_scenarios():
    """Test error handling in multi-turn scenarios."""
    print("\n=== TESTING ERROR SCENARIOS ===")
    
    # Test with malformed tool response
    messages = [
        {"role": "user", "content": "Calculate 10 + 5"},
    ]
    
    result = make_request(messages, TOOLS)
    
    if result.get('tool_calls'):
        # Add malformed tool response
        messages.append({
            "role": "assistant",
            "tool_calls": result['tool_calls']
        })
        messages.append({
            "role": "tool",
            "tool_call_id": result['tool_calls'][0]['id'],
            "content": "ERROR: Division by zero"  # Error response
        })
        
        # See how assistant handles error
        result2 = make_request(messages, TOOLS)
        print(f"Response to error: {len(result2.get('content', ''))} chars")
        print(f"Contains 'error': {'error' in result2.get('content', '').lower()}")
        
        return True
    
    return False

def test_long_conversation():
    """Test longer multi-turn conversation."""
    print("\n=== TESTING LONG CONVERSATION ===")
    
    messages = []
    
    # Turn 1: Weather request
    messages.append({"role": "user", "content": "What's the weather in London?"})
    result1 = make_request(messages, TOOLS)
    
    if result1.get('tool_calls'):
        messages.append({"role": "assistant", "tool_calls": result1['tool_calls']})
        messages.append({
            "role": "tool",
            "tool_call_id": result1['tool_calls'][0]['id'],
            "content": json.dumps({"temperature": 15, "description": "Rainy"})
        })
        
        result1b = make_request(messages, TOOLS)
        messages.append({"role": "assistant", "content": result1b.get('content', '')})
    
    # Turn 2: Math question
    messages.append({"role": "user", "content": "If it's 15 degrees, what's that in Fahrenheit?"})
    result2 = make_request(messages, TOOLS)
    
    if result2.get('tool_calls'):
        messages.append({"role": "assistant", "tool_calls": result2['tool_calls']})
        messages.append({
            "role": "tool",
            "tool_call_id": result2['tool_calls'][0]['id'],
            "content": "59.0"
        })
        
        result2b = make_request(messages, TOOLS)
        messages.append({"role": "assistant", "content": result2b.get('content', '')})
    
    # Turn 3: Regular question
    messages.append({"role": "user", "content": "Should I bring an umbrella?"})
    result3 = make_request(messages, TOOLS)
    messages.append({"role": "assistant", "content": result3.get('content', '')})
    
    # Turn 4: Another calculation
    messages.append({"role": "user", "content": "What's 59 - 32?"})
    result4 = make_request(messages, TOOLS)
    
    print(f"Conversation length: {len(messages)} messages")
    print(f"Final response has tool calls: {len(result4.get('tool_calls', [])) > 0}")
    
    return len(messages) >= 8  # Should have at least 8 messages

def main():
    """Run multi-turn conversation tests."""
    print("DeepSeek R1 Multi-Turn Conversation Testing")
    print("=" * 50)
    
    tests = [
        ("Complete Tool Flow", test_complete_tool_flow),
        ("Mixed Conversation", test_mixed_conversation), 
        ("Error Scenarios", test_error_scenarios),
        ("Long Conversation", test_long_conversation)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
            status = "✅ PASS" if success else "❌ FAIL"
            print(f"{status} {test_name}")
        except Exception as e:
            results.append((test_name, False))
            print(f"❌ FAIL {test_name}: {e}")
    
    print("\n" + "=" * 50)
    print("MULTI-TURN TEST RESULTS")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    print(f"Passed: {passed}/{total} ({passed/total*100:.1f}%)")
    
    for test_name, success in results:
        status = "✅" if success else "❌"
        print(f"  {status} {test_name}")

if __name__ == "__main__":
    main()
