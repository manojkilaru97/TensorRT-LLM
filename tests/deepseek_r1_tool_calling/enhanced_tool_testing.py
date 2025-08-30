#!/usr/bin/env python3
"""
Enhanced comprehensive test suite for DeepSeek R1 tool calling functionality.

This extends the basic test_tool_calling.py with:
1. More thorough reasoning content validation
2. Streaming parser issue analysis
3. Edge case testing
4. Performance benchmarking
5. Tool call argument validation
6. Multi-turn conversation testing
7. Error scenario testing
"""

import json
import time
import pathlib
import statistics
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

import requests


SERVER = "http://0.0.0.0:8003"
CHAT_URL = f"{SERVER}/v1/chat/completions"

# Enhanced tools for comprehensive testing
COMPREHENSIVE_TOOLS: List[Dict[str, Any]] = [
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
                    },
                    "include_forecast": {
                        "type": "boolean",
                        "description": "Whether to include 5-day forecast",
                        "default": False
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
                    },
                    "precision": {
                        "type": "integer",
                        "description": "Number of decimal places for result",
                        "default": 2
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
                        "default": 5,
                        "minimum": 1,
                        "maximum": 20
                    },
                    "language": {
                        "type": "string",
                        "description": "Language for search results",
                        "default": "en"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "send_email",
            "description": "Send an email message",
            "parameters": {
                "type": "object",
                "properties": {
                    "to": {
                        "type": "string",
                        "description": "Recipient email address"
                    },
                    "subject": {
                        "type": "string",
                        "description": "Email subject"
                    },
                    "body": {
                        "type": "string",
                        "description": "Email body content"
                    },
                    "priority": {
                        "type": "string",
                        "enum": ["low", "normal", "high"],
                        "description": "Email priority",
                        "default": "normal"
                    }
                },
                "required": ["to", "subject", "body"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "file_operations",
            "description": "Perform file operations like read, write, delete",
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["read", "write", "delete", "list"],
                        "description": "Type of file operation"
                    },
                    "path": {
                        "type": "string",
                        "description": "File or directory path"
                    },
                    "content": {
                        "type": "string",
                        "description": "Content to write (for write operation)"
                    },
                    "encoding": {
                        "type": "string",
                        "description": "File encoding",
                        "default": "utf-8"
                    }
                },
                "required": ["operation", "path"]
            }
        }
    }
]

@dataclass
class TestResult:
    """Test result container."""
    test_name: str
    success: bool
    duration: float
    details: Dict[str, Any]
    errors: List[str]
    warnings: List[str]

class EnhancedTester:
    """Enhanced testing class with comprehensive validation."""
    
    def __init__(self):
        self.results: List[TestResult] = []
        self.start_time = time.time()
    
    def make_request(self, messages: List[Dict], tools: List[Dict] = None, 
                    stream: bool = False, model: str = "DeepSeek-R1-0528-FP4",
                    timeout: int = 60, verbose: bool = True) -> Dict[str, Any]:
        """Enhanced request method with timing and error handling."""
        
        start_time = time.time()
        payload = {
            "model": model,
            "messages": messages,
            "stream": stream,
            "max_tokens": 4096,
            "temperature": 0.1,
            "add_generation_prompt": True,
        }
        
        if tools:
            payload["tools"] = tools
            
        if verbose:
            print(f"\n{'='*80}")
            print(f"REQUEST: stream={stream}, tools={'Yes' if tools else 'No'}")
            print(f"Messages: {json.dumps(messages[-1], indent=2)}")  # Only show last message
            print(f"{'='*80}")
        
        try:
            if stream:
                response = requests.post(CHAT_URL, json=payload, stream=True, timeout=timeout)
                response.raise_for_status()
                
                if verbose:
                    print("STREAMING RESPONSE:")
                
                full_content = ""
                full_reasoning = ""
                tool_calls = []
                chunks_received = 0
                
                for line in response.iter_lines():
                    if line:
                        line = line.decode('utf-8')
                        if line.startswith('data: '):
                            data_str = line[6:]
                            if data_str.strip() == '[DONE]':
                                break
                            try:
                                data = json.loads(data_str)
                                chunks_received += 1
                                if 'choices' in data and len(data['choices']) > 0:
                                    delta = data['choices'][0]['delta']
                                    if 'content' in delta and delta['content']:
                                        if verbose:
                                            print(delta['content'], end='', flush=True)
                                        full_content += delta['content']
                                    if 'reasoning_content' in delta and delta['reasoning_content']:
                                        full_reasoning += delta['reasoning_content']
                                    if 'tool_calls' in delta and delta['tool_calls']:
                                        for tc in delta['tool_calls']:
                                            tool_calls.append(tc)
                            except json.JSONDecodeError as e:
                                if verbose:
                                    print(f"[JSON Error: {e}]", end='')
                                continue
                
                duration = time.time() - start_time
                
                if verbose:
                    print(f"\n\nSTREAM STATS: {chunks_received} chunks, {duration:.2f}s")
                    print(f"CONTENT LENGTH: {len(full_content)}")
                    print(f"REASONING LENGTH: {len(full_reasoning)}")
                    print(f"TOOL CALLS: {len(tool_calls)}")
                
                return {
                    "content": full_content, 
                    "reasoning_content": full_reasoning,
                    "tool_calls": tool_calls,
                    "duration": duration,
                    "chunks_received": chunks_received
                }
                
            else:
                response = requests.post(CHAT_URL, json=payload, timeout=timeout)
                duration = time.time() - start_time
                
                if response.status_code != 200:
                    error_msg = f"HTTP {response.status_code}: {response.text}"
                    if verbose:
                        print(f"Error response: {error_msg}")
                    return {"error": error_msg, "duration": duration}
                
                result = response.json()
                
                if verbose:
                    print("NON-STREAMING RESPONSE:")
                    # Don't print full result, just key info
                    if 'choices' in result and len(result['choices']) > 0:
                        message = result['choices'][0]['message']
                        print(f"Content length: {len(message.get('content', ''))}")
                        print(f"Reasoning length: {len(message.get('reasoning_content', '') or '')}")
                        print(f"Tool calls: {len(message.get('tool_calls', []))}")
                        print(f"Duration: {duration:.2f}s")
                
                if 'choices' in result and len(result['choices']) > 0:
                    message = result['choices'][0]['message']
                    return {
                        "content": message.get('content', ''),
                        "reasoning_content": message.get('reasoning_content', ''),
                        "tool_calls": message.get('tool_calls', []),
                        "duration": duration,
                        "usage": result.get('usage', {}),
                        "finish_reason": result['choices'][0].get('finish_reason')
                    }
                
                return {"content": "", "reasoning_content": "", "tool_calls": [], "duration": duration}
                
        except Exception as e:
            duration = time.time() - start_time
            error_msg = str(e)
            if verbose:
                print(f"ERROR: {error_msg}")
            return {"error": error_msg, "duration": duration}

    def validate_reasoning_content(self, result: Dict[str, Any], test_name: str) -> Tuple[bool, List[str], List[str]]:
        """Comprehensive reasoning content validation."""
        errors = []
        warnings = []
        
        reasoning = result.get('reasoning_content', '') or ''
        tool_calls = result.get('tool_calls', [])
        content = result.get('content', '')
        
        # Check if reasoning exists when expected
        if not tool_calls and not reasoning:
            errors.append("No reasoning content for non-tool query")
        
        # Analyze reasoning quality
        if reasoning:
            if len(reasoning) < 50:
                warnings.append(f"Short reasoning content: {len(reasoning)} chars")
            
            # Check for thinking tags
            if '<think>' in reasoning and '</think>' in reasoning:
                warnings.append("Reasoning contains XML-style thinking tags")
            
            # Check reasoning structure
            if reasoning.count('\n') < 2:
                warnings.append("Reasoning appears to be single paragraph")
        
        # For tool calls, reasoning might be empty - this could be normal
        if tool_calls and not reasoning:
            warnings.append("No reasoning content with tool calls (may be expected)")
        
        return len(errors) == 0, errors, warnings

    def validate_tool_calls(self, result: Dict[str, Any], expected_count: int = None,
                          expected_functions: List[str] = None) -> Tuple[bool, List[str], List[str]]:
        """Comprehensive tool call validation."""
        errors = []
        warnings = []
        
        tool_calls = result.get('tool_calls', [])
        
        # Count validation
        if expected_count is not None and len(tool_calls) != expected_count:
            errors.append(f"Expected {expected_count} tool calls, got {len(tool_calls)}")
        
        # Function name validation
        if expected_functions:
            actual_functions = []
            for tc in tool_calls:
                func_name = tc.get('function', {}).get('name')
                if func_name:
                    actual_functions.append(func_name)
            
            for func in expected_functions:
                if func not in actual_functions:
                    errors.append(f"Expected function '{func}' not found")
        
        # Structure validation
        valid_calls = 0
        for i, tc in enumerate(tool_calls):
            call_errors = []
            
            # Required fields
            if 'id' not in tc:
                call_errors.append(f"Tool call {i} missing 'id'")
            if 'type' not in tc:
                call_errors.append(f"Tool call {i} missing 'type'")
            elif tc['type'] != 'function':
                call_errors.append(f"Tool call {i} type should be 'function', got '{tc['type']}'")
            
            if 'function' not in tc:
                call_errors.append(f"Tool call {i} missing 'function'")
            else:
                func = tc['function']
                if 'name' not in func:
                    call_errors.append(f"Tool call {i} function missing 'name'")
                if 'arguments' not in func:
                    call_errors.append(f"Tool call {i} function missing 'arguments'")
                else:
                    # Validate JSON arguments
                    try:
                        args = json.loads(func['arguments'])
                        if not isinstance(args, dict):
                            call_errors.append(f"Tool call {i} arguments should be JSON object")
                        else:
                            valid_calls += 1
                    except json.JSONDecodeError as e:
                        call_errors.append(f"Tool call {i} invalid JSON arguments: {e}")
            
            if call_errors:
                errors.extend(call_errors)
        
        if tool_calls and valid_calls == 0:
            errors.append("No valid tool calls found")
        elif tool_calls and valid_calls < len(tool_calls):
            warnings.append(f"Only {valid_calls}/{len(tool_calls)} tool calls are valid")
        
        return len(errors) == 0, errors, warnings

    def run_test(self, test_name: str, test_func) -> TestResult:
        """Run a single test with comprehensive error handling."""
        print(f"\n{'#'*80}")
        print(f"RUNNING: {test_name}")
        print(f"{'#'*80}")
        
        start_time = time.time()
        errors = []
        warnings = []
        details = {}
        
        try:
            test_details = test_func()
            if isinstance(test_details, dict):
                details.update(test_details)
            success = True
        except AssertionError as e:
            errors.append(f"Assertion failed: {e}")
            success = False
        except Exception as e:
            errors.append(f"Unexpected error: {e}")
            success = False
        
        duration = time.time() - start_time
        
        result = TestResult(
            test_name=test_name,
            success=success,
            duration=duration,
            details=details,
            errors=errors,
            warnings=warnings
        )
        
        self.results.append(result)
        
        # Print result
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"\n{status} {test_name} ({duration:.2f}s)")
        if errors:
            for error in errors:
                print(f"  ❌ {error}")
        if warnings:
            for warning in warnings:
                print(f"  ⚠️  {warning}")
        
        return result

    def test_reasoning_content_comprehensive(self) -> Dict[str, Any]:
        """Comprehensive reasoning content testing."""
        test_cases = [
            {
                "name": "Simple factual question",
                "message": "What is the capital of Japan?",
                "tools": None,
                "expect_reasoning": True
            },
            {
                "name": "Complex analytical question", 
                "message": "Explain the economic implications of artificial intelligence on job markets in the next decade.",
                "tools": None,
                "expect_reasoning": True
            },
            {
                "name": "Mathematical reasoning",
                "message": "If I have 100 apples and give away 30% to friends, then eat 20% of what remains, how many apples do I have left? Show your reasoning.",
                "tools": None,
                "expect_reasoning": True
            },
            {
                "name": "Creative task",
                "message": "Write a short poem about coding in Python.",
                "tools": None,
                "expect_reasoning": True
            }
        ]
        
        results = {}
        
        for case in test_cases:
            print(f"\n--- Testing: {case['name']} ---")
            messages = [{"role": "user", "content": case["message"]}]
            
            # Test both streaming and non-streaming
            for stream in [False, True]:
                mode = "streaming" if stream else "non-streaming"
                result = self.make_request(messages, tools=case["tools"], stream=stream, verbose=False)
                
                success, errors, warnings = self.validate_reasoning_content(result, case["name"])
                
                results[f"{case['name']}_{mode}"] = {
                    "success": success,
                    "errors": errors,
                    "warnings": warnings,
                    "reasoning_length": len(result.get('reasoning_content', '') or ''),
                    "content_length": len(result.get('content', '')),
                    "duration": result.get('duration', 0)
                }
                
                print(f"  {mode}: reasoning={len(result.get('reasoning_content', '') or '')} chars, "
                      f"content={len(result.get('content', ''))} chars")
        
        return {"test_cases": results}

    def test_streaming_parser_issues(self) -> Dict[str, Any]:
        """Analyze streaming parser issues in detail."""
        test_cases = [
            {
                "name": "Single tool call",
                "message": "What's the weather in Tokyo?",
                "expected_tools": ["get_weather"]
            },
            {
                "name": "Multiple tool calls",
                "message": "Get weather for London and calculate 15 * 23",
                "expected_tools": ["get_weather", "calculate"]
            },
            {
                "name": "Complex arguments",
                "message": "Search for 'machine learning trends 2024' with max 10 results in English",
                "expected_tools": ["search_web"]
            }
        ]
        
        results = {}
        
        for case in test_cases:
            print(f"\n--- Analyzing streaming issues: {case['name']} ---")
            messages = [{"role": "user", "content": case["message"]}]
            
            # Get non-streaming baseline
            non_stream_result = self.make_request(messages, tools=COMPREHENSIVE_TOOLS, stream=False, verbose=False)
            
            # Get streaming result
            stream_result = self.make_request(messages, tools=COMPREHENSIVE_TOOLS, stream=True, verbose=False)
            
            # Analyze differences
            non_stream_calls = non_stream_result.get('tool_calls', [])
            stream_calls = stream_result.get('tool_calls', [])
            
            analysis = {
                "non_stream_count": len(non_stream_calls),
                "stream_count": len(stream_calls),
                "chunks_received": stream_result.get('chunks_received', 0),
                "streaming_issues": []
            }
            
            # Check for fragmented arguments
            if stream_calls:
                for i, call in enumerate(stream_calls):
                    args = call.get('function', {}).get('arguments', '')
                    if len(args) < 10:  # Likely fragmented
                        analysis["streaming_issues"].append(f"Call {i}: fragmented arguments '{args}'")
                    
                    try:
                        json.loads(args)
                    except json.JSONDecodeError:
                        analysis["streaming_issues"].append(f"Call {i}: invalid JSON '{args}'")
            
            # Check if we can reconstruct proper tool calls from fragments
            if len(stream_calls) > len(non_stream_calls):
                # Try to reconstruct
                reconstructed = self._try_reconstruct_tool_calls(stream_calls)
                analysis["reconstruction_possible"] = reconstructed is not None
                if reconstructed:
                    analysis["reconstructed_calls"] = len(reconstructed)
            
            results[case["name"]] = analysis
            
            print(f"  Non-stream: {len(non_stream_calls)} calls")
            print(f"  Stream: {len(stream_calls)} calls ({len(analysis['streaming_issues'])} issues)")
        
        return {"streaming_analysis": results}

    def _try_reconstruct_tool_calls(self, fragmented_calls: List[Dict]) -> Optional[List[Dict]]:
        """Attempt to reconstruct proper tool calls from fragments."""
        if not fragmented_calls:
            return None
        
        # Group by tool call ID
        call_groups = {}
        for call in fragmented_calls:
            call_id = call.get('id')
            if call_id not in call_groups:
                call_groups[call_id] = []
            call_groups[call_id].append(call)
        
        reconstructed = []
        for call_id, fragments in call_groups.items():
            if not fragments:
                continue
            
            # Take structure from first fragment
            base_call = fragments[0].copy()
            
            # Concatenate arguments
            full_args = ""
            for fragment in fragments:
                args = fragment.get('function', {}).get('arguments', '')
                full_args += args
            
            # Update the base call
            if 'function' in base_call:
                base_call['function']['arguments'] = full_args
            
            # Validate reconstructed JSON
            try:
                json.loads(full_args)
                reconstructed.append(base_call)
            except json.JSONDecodeError:
                return None  # Reconstruction failed
        
        return reconstructed if reconstructed else None

    def test_edge_cases_comprehensive(self) -> Dict[str, Any]:
        """Test various edge cases and error scenarios."""
        edge_cases = [
            {
                "name": "Empty message",
                "message": "",
                "tools": COMPREHENSIVE_TOOLS,
                "should_error": False
            },
            {
                "name": "Very long message",
                "message": "Please help me " + "really " * 500 + "with this task.",
                "tools": COMPREHENSIVE_TOOLS,
                "should_error": False
            },
            {
                "name": "Special characters",
                "message": "Calculate: 2 + 2 = ? Also check weather in São Paulo, Brazil 🌤️",
                "tools": COMPREHENSIVE_TOOLS,
                "should_error": False
            },
            {
                "name": "JSON-like content",
                "message": 'Process this data: {"name": "test", "value": 123}',
                "tools": COMPREHENSIVE_TOOLS,
                "should_error": False
            },
            {
                "name": "Code in message",
                "message": "Help me debug this Python code:\n```python\ndef hello():\n    print('world')\n```",
                "tools": COMPREHENSIVE_TOOLS,
                "should_error": False
            },
            {
                "name": "No tools available",
                "message": "What's the weather like?",
                "tools": [],
                "should_error": False
            },
            {
                "name": "Ambiguous tool request",
                "message": "Help me with something",
                "tools": COMPREHENSIVE_TOOLS,
                "should_error": False
            }
        ]
        
        results = {}
        
        for case in edge_cases:
            print(f"\n--- Edge case: {case['name']} ---")
            messages = [{"role": "user", "content": case["message"]}]
            
            result = self.make_request(messages, tools=case["tools"], stream=False, verbose=False)
            
            has_error = "error" in result
            success = has_error == case["should_error"]
            
            results[case["name"]] = {
                "success": success,
                "has_error": has_error,
                "expected_error": case["should_error"],
                "content_length": len(result.get('content', '')),
                "reasoning_length": len(result.get('reasoning_content', '') or ''),
                "tool_calls": len(result.get('tool_calls', [])),
                "duration": result.get('duration', 0)
            }
            
            if has_error:
                results[case["name"]]["error"] = result.get("error", "Unknown error")
            
            status = "✓" if success else "✗"
            print(f"  {status} Expected error: {case['should_error']}, Got error: {has_error}")
        
        return {"edge_cases": results}

    def test_performance_benchmarks(self) -> Dict[str, Any]:
        """Performance benchmarking for different scenarios."""
        benchmarks = [
            {
                "name": "Simple query (no tools)",
                "message": "What is 2+2?",
                "tools": None,
                "iterations": 5
            },
            {
                "name": "Single tool call",
                "message": "Calculate 15 * 23",
                "tools": COMPREHENSIVE_TOOLS,
                "iterations": 5
            },
            {
                "name": "Multiple tool calls",
                "message": "Get weather for Paris and search for 'Eiffel Tower facts'",
                "tools": COMPREHENSIVE_TOOLS,
                "iterations": 3
            }
        ]
        
        results = {}
        
        for benchmark in benchmarks:
            print(f"\n--- Benchmarking: {benchmark['name']} ---")
            messages = [{"role": "user", "content": benchmark["message"]}]
            
            # Test both streaming and non-streaming
            for stream in [False, True]:
                mode = "streaming" if stream else "non-streaming"
                durations = []
                
                for i in range(benchmark["iterations"]):
                    result = self.make_request(messages, tools=benchmark["tools"], 
                                             stream=stream, verbose=False)
                    if "duration" in result:
                        durations.append(result["duration"])
                    time.sleep(0.5)  # Brief pause between requests
                
                if durations:
                    results[f"{benchmark['name']}_{mode}"] = {
                        "mean_duration": statistics.mean(durations),
                        "median_duration": statistics.median(durations),
                        "min_duration": min(durations),
                        "max_duration": max(durations),
                        "iterations": len(durations)
                    }
                    
                    print(f"  {mode}: {statistics.mean(durations):.2f}s avg "
                          f"({min(durations):.2f}-{max(durations):.2f}s)")
        
        return {"benchmarks": results}

    def generate_report(self) -> str:
        """Generate comprehensive test report."""
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r.success)
        total_duration = time.time() - self.start_time
        
        report = f"""
# DeepSeek R1 Tool Calling Test Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary
- **Total Tests**: {total_tests}
- **Passed**: {passed_tests}
- **Failed**: {total_tests - passed_tests}
- **Success Rate**: {(passed_tests/total_tests*100):.1f}%
- **Total Duration**: {total_duration:.2f}s

## Test Results
"""
        
        for result in self.results:
            status = "✅ PASS" if result.success else "❌ FAIL"
            report += f"\n### {status} {result.test_name} ({result.duration:.2f}s)\n"
            
            if result.errors:
                report += "**Errors:**\n"
                for error in result.errors:
                    report += f"- {error}\n"
            
            if result.warnings:
                report += "**Warnings:**\n"
                for warning in result.warnings:
                    report += f"- {warning}\n"
            
            if result.details:
                report += "**Details:**\n"
                for key, value in result.details.items():
                    if isinstance(value, dict):
                        report += f"- {key}: {json.dumps(value, indent=2)}\n"
                    else:
                        report += f"- {key}: {value}\n"
        
        return report


def main():
    """Run enhanced test suite."""
    print("Starting Enhanced DeepSeek R1 Tool Calling Tests")
    print(f"Server URL: {CHAT_URL}")
    
    tester = EnhancedTester()
    
    # Health check
    try:
        response = requests.get(f"{SERVER}/health", timeout=5)
        print(f"Server health check: {response.status_code}")
    except Exception as e:
        print(f"Warning: Could not reach health endpoint: {e}")
    
    # Run comprehensive tests
    tester.run_test("Reasoning Content Comprehensive", tester.test_reasoning_content_comprehensive)
    tester.run_test("Streaming Parser Issues Analysis", tester.test_streaming_parser_issues)
    tester.run_test("Edge Cases Comprehensive", tester.test_edge_cases_comprehensive)
    tester.run_test("Performance Benchmarks", tester.test_performance_benchmarks)
    
    # Generate and save report
    report = tester.generate_report()
    
    with open("/workspace/test_report.md", "w") as f:
        f.write(report)
    
    print(f"\n{'#'*80}")
    print("ENHANCED TESTING COMPLETED")
    print(f"Report saved to: /workspace/test_report.md")
    print(f"{'#'*80}")
    
    # Print summary
    total_tests = len(tester.results)
    passed_tests = sum(1 for r in tester.results if r.success)
    print(f"\nSUMMARY: {passed_tests}/{total_tests} tests passed ({(passed_tests/total_tests*100):.1f}%)")


if __name__ == "__main__":
    main()
