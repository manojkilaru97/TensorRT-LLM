from dataclasses import dataclass, field
from typing import List, Literal, Optional, Tuple, Union
import re
import json
import uuid

from .._utils import nvtx_range_debug
from ..executor import (DetokenizedGenerationResultBase, GenerationResult,
                        GenerationResultBase)
from ..executor.postproc_worker import PostprocArgs
from ..llmapi.reasoning_parser import (BaseReasoningParser,
                                       ReasoningParserFactory)
from ..llmapi.tokenizer import TransformersTokenizer
# yapf: disable
from .openai_protocol import (ChatCompletionLogProbs,
                              ChatCompletionLogProbsContent,
                              ChatCompletionNamedToolChoiceParam,
                              ChatCompletionRequest, ChatCompletionResponse,
                              ChatCompletionResponseChoice,
                              ChatCompletionResponseStreamChoice,
                              ChatCompletionStreamResponse,
                              ChatCompletionToolsParam, ChatMessage,
                              CompletionRequest, CompletionResponse,
                              CompletionResponseChoice,
                              CompletionResponseStreamChoice,
                              CompletionStreamResponse, DeltaMessage,
                              FunctionCall, StreamOptions, ToolCall, UsageInfo,
                              to_disaggregated_params)

# yapf: enale

@dataclass(kw_only=True)
class ChatPostprocArgs(PostprocArgs):
    echo: bool = False
    role: str = None
    model: str = None
    num_choices: int = 1
    tools: Optional[List[ChatCompletionToolsParam]] = None
    tool_choice: Optional[Union[Literal["none"],
                                ChatCompletionNamedToolChoiceParam]] = "none"
    return_logprobs: bool = False
    stream_options: Optional[StreamOptions] = None
    last_message_content: Optional[str] = None
    reasoning_parser: Optional[str] = None
    reasoning_parser_dict: dict[int, BaseReasoningParser] = field(
        default_factory=dict)
    # Internal state for tool-call parsing per output index
    tool_call_parser_states: dict[int, "_DeepSeekV3ToolCallParser"] = field(
        default_factory=dict)

    @classmethod
    def from_request(cls, request: ChatCompletionRequest):
        return cls(
            echo=request.echo,
            role="assistant"
            if request.add_generation_prompt else request.messages[-1]["role"],
            model=request.model,
            num_choices=request.n if request.n else 1,
            tools=request.tools,
            tool_choice=request.tool_choice,
            stream_options=request.stream_options,
            return_logprobs=request.logprobs,
        )


def create_logprobs(token_ids: List[int],
                    tokenizer: TransformersTokenizer,
                    logprobs: List[float]) -> ChatCompletionLogProbs:
    assert len(token_ids) == len(logprobs), \
            "token_ids and logprobs have different lengths"
    content: List[ChatCompletionLogProbsContent] = []
    for token_id, logprob in zip(token_ids, logprobs):
        token = tokenizer.decode(token_id)
        # returning multiple logprobs is not supported
        first_logprob = ChatCompletionLogProbsContent(
            token=token,
            logprob=max(logprob, -9999.0),
            bytes=list(token.encode("utf-8", errors="replace")))
        content.append(first_logprob)
    chat_logprobs = ChatCompletionLogProbs(content=content)
    return chat_logprobs


def apply_reasoning_parser(args: ChatPostprocArgs, output_index: int, text: str, streaming: bool) -> Tuple[bool, str, str]:
    reasoning_parser = None
    if args.reasoning_parser is not None:
        if output_index not in args.reasoning_parser_dict:
            args.reasoning_parser_dict[output_index] = ReasoningParserFactory.create_reasoning_parser(
                args.reasoning_parser)
        reasoning_parser = args.reasoning_parser_dict[output_index]

    in_reasoning = False
    if reasoning_parser is not None:
        if not streaming:
            result = reasoning_parser.parse(text)
        else:
            result = reasoning_parser.parse_delta(text)
        in_reasoning, content, reasoning_content = result.in_reasoning, result.content, result.reasoning_content
    else:
        in_reasoning, content, reasoning_content = False, text, None

    return in_reasoning, content, reasoning_content


class _DeepSeekV3ToolCallParser:
    """
    Streaming and one-shot parser for DeepSeek V3-style tool call format used by DeepSeek-R1/DeepSeek-V3.

    Format example (multiple calls supported):
    <｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>get_current_weather\n```json\n{"location": "Tokyo"}\n```<｜tool▁call▁end｜>...<｜tool▁calls▁end｜>
    
    Also supports alternative format:
    <tool_call>
    {
      "name": "function_name",
      "arguments": {...}
    }
    </tool_call>
    """

    def __init__(self):
        self.bot_token = "<｜tool▁calls▁begin｜>"
        self.eot_token = "<｜tool▁calls▁end｜>"
        self._buffer: str = ""
        self._last_arguments: str = ""
        self.current_tool_id: int = -1
        self.current_tool_name_sent: bool = False
        self.current_tool_call_id: str = ""  # Track the ID for the current tool call
        self._accumulated_arguments: str = ""  # Track accumulated arguments for streaming
        self._last_sent_length: int = 0  # Track how much we've already sent

        # regexes for DeepSeek V3 format
        self.func_block_regex = r"<｜tool▁call▁begin｜>.*?<｜tool▁call▁end｜>"
        self.func_detail_regex = r"<｜tool▁call▁begin｜>(.*)<｜tool▁sep｜>(.*)\n```json\n([\s\S]*?)\n```<｜tool▁call▁end｜>"
        
        # regexes for alternative format
        self.alt_tool_call_regex = r"<tool_call>\s*\{[\s\S]*?\}\s*</tool_call>"
        self.alt_tool_detail_regex = r"<tool_call>\s*\{\s*\"name\"\s*:\s*\"([^\"]+)\"\s*,\s*\"arguments\"\s*:\s*(\{[\s\S]*?\})\s*\}\s*</tool_call>"

    @staticmethod
    def _is_complete_json(s: str) -> bool:
        """Check if string is valid and complete JSON."""
        if not s or not s.strip():
            return False
        try:
            json.loads(s)
            return True
        except (json.JSONDecodeError, ValueError, TypeError):
            return False
    
    @staticmethod
    def _validate_function_name(name: str) -> bool:
        """Validate function name follows expected patterns."""
        if not name or not isinstance(name, str):
            return False
        # Function names should be alphanumeric with underscores, reasonable length
        return bool(re.match(r'^[a-zA-Z_][a-zA-Z0-9_]{0,63}$', name))
    
    @staticmethod
    def _sanitize_arguments(args_str: str) -> str:
        """Sanitize and validate JSON arguments string."""
        if not args_str:
            return "{}"
        
        # Remove any potential injection attempts or malformed content
        args_str = args_str.strip()
        
        # Basic validation - should start with { and end with }
        if not (args_str.startswith('{') and args_str.endswith('}')):
            return "{}"
        
        try:
            # Validate it's proper JSON
            parsed = json.loads(args_str)
            if not isinstance(parsed, dict):
                return "{}"
            # Re-serialize to ensure clean JSON
            return json.dumps(parsed, separators=(',', ':'))
        except (json.JSONDecodeError, ValueError, TypeError):
            return "{}"

    def parse_stream(self, new_text: str) -> Tuple[list[ToolCall], str]:
        """Parse a streaming delta. Returns (tool_call_deltas, normal_text)."""
        if not new_text:
            return [], ""
        self._buffer += new_text
        current_text = self._buffer

        # Check for DeepSeek V3 format
        has_tool_call = (self.bot_token in current_text) or ("<｜tool▁call▁begin｜>" in current_text)
        
        # Check for alternative format
        has_alt_tool_call = "<tool_call>" in current_text
        
        if not has_tool_call and not has_alt_tool_call:
            # Not a tool call. Strip any stray special markers that might leak.
            cleaned = new_text
            for tok in (self.eot_token, "```", "<｜tool▁call▁end｜>", "</tool_call>"):
                cleaned = cleaned.replace(tok, "")
            self._buffer = ""  # do not accumulate non-tool text
            return [], cleaned

        # Try DeepSeek V3 format first
        if has_tool_call:
            # Try to partially match a tool call (without requiring closing markers)
            partial = re.search(
                pattern=r"<｜tool▁call▁begin｜>(.*)<｜tool▁sep｜>(.*)\n```json\n([\s\S]*)",
                string=current_text,
                flags=re.DOTALL,
            )

            deltas: list[ToolCall] = []
            if partial:
                func_name = partial.group(2).strip()
                func_args_raw = partial.group(3)
                
                # Validate function name
                if not self._validate_function_name(func_name):
                    # Invalid function name, skip this tool call
                    return [], ""
                
                # Limit argument length to prevent abuse
                if len(func_args_raw) > 10000:  # 10KB limit
                    func_args_raw = func_args_raw[:10000]

                # Initialize tracking for first tool call
                if self.current_tool_id == -1:
                    self.current_tool_id = 0
                    self.current_tool_name_sent = False
                    self._last_arguments = ""
                    self._accumulated_arguments = ""
                    self._last_sent_length = 0
                    # Generate a unique ID for this tool call that will be reused across all chunks
                    self.current_tool_call_id = f"chatcmpl-tool-{uuid.uuid4().hex}"

                if not self.current_tool_name_sent:
                    deltas.append(
                        ToolCall(id=self.current_tool_call_id, function=FunctionCall(name=func_name, arguments=""))
                    )
                    self.current_tool_name_sent = True
                    self._accumulated_arguments = func_args_raw
                    self._last_sent_length = 0
                else:
                    # Update accumulated arguments
                    self._accumulated_arguments = func_args_raw
                    
                    # Only send delta if we have new content and it's substantial
                    if len(self._accumulated_arguments) > self._last_sent_length:
                        # Send incremental chunks, but not character by character
                        # Wait for at least a few characters or complete tokens before sending
                        new_content = self._accumulated_arguments[self._last_sent_length:]
                        
                        # Only send if we have substantial new content (avoid single character fragments)
                        # or if we have a complete JSON structure
                        should_send = (
                            len(new_content) >= 3 or  # At least 3 characters
                            new_content.endswith('"') or  # Complete string value
                            new_content.endswith(',') or  # Complete field
                            new_content.endswith('}') or  # Complete object
                            self._is_complete_json(self._accumulated_arguments)  # Complete JSON
                        )
                        
                        if should_send:
                            try:
                                deltas.append(
                                    ToolCall(id=self.current_tool_call_id, function=FunctionCall(name=func_name, arguments=new_content))
                                )
                                self._last_sent_length = len(self._accumulated_arguments)
                            except Exception:
                                # If tool call creation fails, skip this delta
                                pass

                # If json is complete, send any remaining content and reset state for next tool
                if self._is_complete_json(func_args_raw):
                    # Send any remaining unsent content
                    if len(self._accumulated_arguments) > self._last_sent_length:
                        remaining_content = self._accumulated_arguments[self._last_sent_length:]
                        if remaining_content:
                            try:
                                deltas.append(
                                    ToolCall(id=self.current_tool_call_id, function=FunctionCall(name=func_name, arguments=remaining_content))
                                )
                            except Exception:
                                # If tool call creation fails, skip this delta
                                pass
                    
                    # Remove the first completed block from buffer
                    match = re.search(self.func_block_regex, current_text, re.DOTALL)
                    if match:
                        self._buffer = current_text[match.end():]
                    else:
                        self._buffer = ""
                    # reset tool state for the next tool
                    self.current_tool_id += 1
                    self._last_arguments = ""
                    self._accumulated_arguments = ""
                    self._last_sent_length = 0
                    self.current_tool_name_sent = False
                    # Generate new ID for the next tool call
                    self.current_tool_call_id = f"chatcmpl-tool-{uuid.uuid4().hex}"

                return deltas, ""

        # Try alternative format for streaming
        if has_alt_tool_call:
            # Look for complete tool calls in alternative format
            complete_matches = re.findall(self.alt_tool_call_regex, current_text, re.DOTALL)
            if complete_matches:
                deltas: list[ToolCall] = []
                for match in complete_matches:
                    detail_match = re.search(self.alt_tool_detail_regex, match, re.DOTALL)
                    if detail_match:
                        func_name = detail_match.group(1).strip()
                        func_args_str = detail_match.group(2).strip()
                        
                        # Validate function name and arguments
                        if not self._validate_function_name(func_name):
                            continue
                        
                        # Sanitize and validate arguments
                        clean_args = self._sanitize_arguments(func_args_str)
                        if clean_args != "{}":  # Only proceed if we have valid arguments
                            try:
                                deltas.append(ToolCall(function=FunctionCall(name=func_name, arguments=clean_args)))
                                # Remove processed tool call from buffer
                                self._buffer = self._buffer.replace(match, "")
                            except Exception:
                                continue
                return deltas, ""

        # Fallback
        return [], ""

    def detect_and_parse(self, text: str) -> Tuple[str, list[ToolCall]]:
        # First try DeepSeek V3 format
        idx = text.find(self.bot_token)
        if idx != -1:
            normal_text = text[:idx].strip()
            calls: list[ToolCall] = []
            try:
                for block in re.findall(self.func_block_regex, text, re.DOTALL):
                    m = re.search(self.func_detail_regex, block, re.DOTALL)
                    if not m:
                        continue
                    func_name = m.group(2).strip()
                    func_args = m.group(3)
                    
                    # Validate function name
                    if not self._validate_function_name(func_name):
                        continue
                    
                    # Sanitize and validate arguments
                    clean_args = self._sanitize_arguments(func_args)
                    if clean_args == "{}" and func_args.strip():  # Skip if sanitization failed on non-empty input
                        continue
                    
                    try:
                        calls.append(ToolCall(function=FunctionCall(name=func_name, arguments=clean_args)))
                    except Exception:
                        continue
                return normal_text, calls
            except Exception:
                pass
        
        # Try alternative format: <tool_call>{"name": "...", "arguments": {...}}</tool_call>
        alt_matches = re.findall(self.alt_tool_call_regex, text, re.DOTALL)
        if alt_matches:
            calls: list[ToolCall] = []
            normal_text = text
            try:
                for match in alt_matches:
                    # Remove the tool call from normal text
                    normal_text = normal_text.replace(match, "").strip()
                    
                    # Parse the tool call
                    detail_match = re.search(self.alt_tool_detail_regex, match, re.DOTALL)
                    if detail_match:
                        func_name = detail_match.group(1).strip()
                        func_args_str = detail_match.group(2).strip()
                        
                        # Validate function name
                        if not self._validate_function_name(func_name):
                            continue
                        
                        # Sanitize and validate arguments
                        clean_args = self._sanitize_arguments(func_args_str)
                        if clean_args != "{}":  # Only proceed if we have valid arguments
                            try:
                                calls.append(ToolCall(function=FunctionCall(name=func_name, arguments=clean_args)))
                            except Exception:
                                continue
                return normal_text, calls
            except Exception:
                pass
        
        # No tool calls found
        return text, []


@nvtx_range_debug("chat_stream_post_processor")
def chat_stream_post_processor(rsp: GenerationResultBase, args: ChatPostprocArgs) -> List[str]:

    def yield_first_chat(num_tokens: int,
                         idx: int,
                         role: str = None,
                         content: str = None):
        choice_data = ChatCompletionResponseStreamChoice(index=idx,
                                                         delta=DeltaMessage(
                                                             role=role,
                                                             content=content),
                                                         finish_reason=None)
        chunk = ChatCompletionStreamResponse(choices=[choice_data],
                                             model=args.model)
        if include_continuous_usage:
            chunk.usage = UsageInfo(prompt_tokens=num_tokens,
                                    total_tokens=num_tokens,
                                    completion_tokens=0)
        data = chunk.model_dump_json(exclude_none=True)
        return data

    res: List[str] = []
    finish_reason_sent = [False] * args.num_choices
    prompt_tokens = args.num_prompt_tokens
    if stream_option := args.stream_options:
        include_usage = stream_option.include_usage
        include_continuous_usage = include_usage and stream_option.continuous_usage_stats
    else:
        include_usage = False
        include_continuous_usage = False
    if args.first_iteration:
        for i in range(args.num_choices):
            res.append(f"data: {yield_first_chat(prompt_tokens, i, role=args.role)} \n\n")
            if args.echo and args.last_message_content:
                res.append(f"data: {yield_first_chat(prompt_tokens, i, content=args.last_message_content)} \n\n")
        args.first_iteration = False

    for output in rsp.outputs:
        i = output.index

        if finish_reason_sent[i]:
            continue

        delta_text = output.text_diff

        in_reasoning, delta_text, reasoning_delta_text = apply_reasoning_parser(
            args, i, delta_text, True)

        # DeepSeek R1/V3 tool-call streaming detection (enabled when reasoning_parser indicates DeepSeek and tools are provided)
        tool_delta_message: Optional[DeltaMessage] = None
        # Only attempt tool-call parsing when not in reasoning and we have actual text
        if (args.tools and args.reasoning_parser == "deepseek-r1" and not in_reasoning
                and delta_text):
            if i not in args.tool_call_parser_states:
                args.tool_call_parser_states[i] = _DeepSeekV3ToolCallParser()
            parser = args.tool_call_parser_states[i]
            tool_deltas, residual_text = parser.parse_stream(delta_text)
            if tool_deltas:
                tool_delta_message = DeltaMessage(tool_calls=tool_deltas)
            else:
                # overwrite delta_text with residual if any
                delta_text = residual_text

        if tool_delta_message is not None:
            delta_message = tool_delta_message
        else:
            if args.tool_choice and type(
                    args.tool_choice) is ChatCompletionNamedToolChoiceParam:
                delta_message = DeltaMessage(tool_calls=[
                    ToolCall(function=FunctionCall(
                        name=args.tool_choice.function.name, arguments=delta_text))
                ])
            else:
                if in_reasoning:
                    delta_message = DeltaMessage(
                        reasoning_content=reasoning_delta_text)
                else:
                    delta_message = DeltaMessage(
                        content=delta_text, reasoning_content=reasoning_delta_text)

        choice = ChatCompletionResponseStreamChoice(index=i,
                                                    delta=delta_message,
                                                    finish_reason=None,
                                                    avg_decoded_tokens_per_iter=getattr(rsp, 'avg_decoded_tokens_per_iter', None))
        if args.return_logprobs:
            logprobs = output.logprobs_diff
            token_ids = output.token_ids_diff
            choice.logprobs = create_logprobs(token_ids, args.tokenizer, logprobs)
        if output.finish_reason is not None:
            choice.finish_reason = output.finish_reason
            choice.stop_reason = output.stop_reason
            finish_reason_sent[i] = True
        chunk = ChatCompletionStreamResponse(choices=[choice], model=args.model)
        if include_continuous_usage:
            chunk.usage = UsageInfo(prompt_tokens=prompt_tokens,
                                    completion_tokens=output.length,
                                    total_tokens=output.length + prompt_tokens)
        data = chunk.model_dump_json(exclude_none=True)
        res.append(f"data: {data}\n\n")

    if include_usage and rsp._done:
        completion_tokens = sum(output.length
                                for output in rsp.outputs)
        final_usage = UsageInfo(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )

        final_usage_chunk = ChatCompletionStreamResponse(
            choices=[], model=args.model, usage=final_usage)
        final_usage_data = final_usage_chunk.model_dump_json()
        res.append(f"data: {final_usage_data}\n\n")
    return res


@nvtx_range_debug("chat_response_post_processor")
def chat_response_post_processor(rsp: GenerationResultBase, args: ChatPostprocArgs) -> ChatCompletionResponse:
    choices: List[ChatCompletionResponseChoice] = []
    role = args.role
    for output in rsp.outputs:
        _, text, reasoning_text = apply_reasoning_parser(
            args, output.index, output.text, False)

        # Try DeepSeek tool-call detection for non-streaming paths
        tool_calls_detected: list[ToolCall] = []
        if args.tools and args.reasoning_parser == "deepseek-r1":
            parser = args.tool_call_parser_states.get(output.index)
            if parser is None:
                parser = _DeepSeekV3ToolCallParser()
            normal_text, tool_calls_detected = parser.detect_and_parse(output.text)
            if tool_calls_detected:
                text = normal_text

        if tool_calls_detected:
            message = ChatMessage(
                role=role,
                content="",
                tool_calls=tool_calls_detected)
        else:
            if args.tool_choice and isinstance(
                    args.tool_choice,
                    ChatCompletionNamedToolChoiceParam):
                message = ChatMessage(
                    role=role,
                    content="",
                    tool_calls=[
                        ToolCall(function=FunctionCall(
                            name=args.tool_choice.function.name,
                            arguments=text))
                    ])
            else:
                if text is None:
                    text = ""
                message = ChatMessage(
                    role=role, content=text, reasoning_content=reasoning_text)
        disaggregated_params = to_disaggregated_params(output.disaggregated_params)
        choice = ChatCompletionResponseChoice(
            index=output.index,
            message=message,
            finish_reason=output.finish_reason,
            stop_reason=output.stop_reason,
            disaggregated_params=disaggregated_params,
            avg_decoded_tokens_per_iter=getattr(rsp, 'avg_decoded_tokens_per_iter', None),
        )

        # If tool calls are detected in full response, set finish_reason accordingly
        if tool_calls_detected:
            choice.finish_reason = "tool_calls"

        if args.return_logprobs:
            choice.logprobs = create_logprobs(output.token_ids, args.tokenizer, output.logprobs)
        choices.append(choice)

    if args.echo and args.last_message_content:
        for choice in choices:
            full_message = args.last_message_content + choice.message.content
            choice.message.content = full_message

    num_prompt_tokens = args.num_prompt_tokens
    num_generated_tokens = sum(
        len(output.token_ids) for output in rsp.outputs)
    usage = UsageInfo(
        prompt_tokens=num_prompt_tokens,
        completion_tokens=num_generated_tokens,
        total_tokens=num_prompt_tokens + num_generated_tokens,
    )
    response = ChatCompletionResponse(
        model=args.model,
        choices=choices,
        usage=usage,
    )
    return response


@dataclass(kw_only=True)
class CompletionPostprocArgs(PostprocArgs):
    echo: bool = False
    model: str = None
    num_choices: int = 1
    prompt_idx: int = 0
    detokenize: bool = True
    prompt: Optional[str] = None
    stream_options: Optional[StreamOptions] = None

    @classmethod
    def from_request(cls, request: CompletionRequest):
        return cls(
            echo=request.echo,
            model=request.model,
            num_choices=request.n if request.n else 1,
            stream_options=request.stream_options,
            detokenize=request.detokenize,
        )


@nvtx_range_debug("completion_stream_post_processor")
def completion_stream_post_processor(rsp: DetokenizedGenerationResultBase, args: CompletionPostprocArgs) -> List[str]:
    res: List[str] = []
    prompt_tokens = args.num_prompt_tokens
    if stream_option := args.stream_options:
        include_usage = stream_option.include_usage
        include_continuous_usage = include_usage and stream_option.continuous_usage_stats
    else:
        include_usage = False
        include_continuous_usage = False

    for output in rsp.outputs:
        delta_text = output.text_diff
        if args.echo and args.first_iteration:
            delta_text = args.prompt + delta_text
        choice = CompletionResponseStreamChoice(
            index=args.prompt_idx * args.num_choices + output.index,
            text=delta_text if args.detokenize else "",
            token_ids=None if args.detokenize else output.token_ids_diff,
            finish_reason = output.finish_reason,
            stop_reason = output.stop_reason,
            avg_decoded_tokens_per_iter=getattr(rsp, 'avg_decoded_tokens_per_iter', None),
        )
        chunk = CompletionStreamResponse(model=args.model, choices=[choice])
        if include_continuous_usage:
            chunk.usage = UsageInfo(prompt_tokens=prompt_tokens,
                                    completion_tokens=output.length,
                                    total_tokens=output.length + prompt_tokens)
        data = chunk.model_dump_json(exclude_unset=False)
        res.append(f"data: {data}\n\n")

    if include_usage and rsp._done:
        completion_tokens = sum(output.length
                                for output in rsp.outputs)
        final_usage = UsageInfo(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )

        final_usage_chunk = ChatCompletionStreamResponse(
            choices=[], model=args.model, usage=final_usage)
        final_usage_data = final_usage_chunk.model_dump_json()
        res.append(f"data: {final_usage_data}\n\n")
    args.first_iteration = False
    return res


@nvtx_range_debug("completion_response_post_processor")
def completion_response_post_processor(rsp: GenerationResult, args: CompletionPostprocArgs) -> CompletionResponse:
    prompt_tokens = args.num_prompt_tokens
    completion_tokens = 0
    choices = []
    for output in rsp.outputs:
        text = output.text
        if args.echo:
            text = args.prompt + text
        disaggregated_params = to_disaggregated_params(output.disaggregated_params)
        choice = CompletionResponseChoice(
            text=text if args.detokenize else "",
            token_ids=None if args.detokenize else output.token_ids,
            index=args.prompt_idx * args.num_choices + output.index,
            disaggregated_params=disaggregated_params,
            context_logits=None if rsp.context_logits is None else rsp.context_logits.tolist(),
            stop_reason=output.stop_reason,
            finish_reason=output.finish_reason,
            avg_decoded_tokens_per_iter=getattr(rsp, 'avg_decoded_tokens_per_iter', None),
        )

        completion_tokens += output.length
        choices.append(choice)

    usage = UsageInfo(prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=completion_tokens + prompt_tokens)
    response = CompletionResponse(choices=choices, model=args.model, usage=usage)
    return response
