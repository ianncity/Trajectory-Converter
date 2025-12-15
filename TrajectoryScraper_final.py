import json
import tiktoken
import numpy as np
import sys
from pathlib import Path
from typing import Dict, List, Any
import math
import mimetypes
from datetime import datetime
import chardet

# Initialize tokenizer
tokenizer = tiktoken.get_encoding("cl100k_base")

def count_tokens(text: str) -> int:
    return len(tokenizer.encode(str(text)))

def normalize_role(role: str) -> str:
    """Normalize role names to standard format: system, user, assistant"""
    if not role:
        return "user"
    
    role_lower = role.lower().strip()
    
    # Map various role representations to standard format
    role_mappings = {
        # System variations
        "system": "system",
        "sys": "system", 
        "s": "system",
        "context": "system",
        "instruction": "system",
        
        # User variations  
        "user": "user",
        "human": "user",
        "human_user": "user",
        "u": "user",
        "customer": "user",
        "client": "user",
        "question": "user",
        "query": "user",
        
        # Assistant variations
        "assistant": "assistant",
        "ai": "assistant",
        "bot": "assistant", 
        "a": "assistant",
        "agent": "assistant",
        "model": "assistant",
        "response": "assistant",
        "answer": "assistant",
        "completion": "assistant"
    }
    
    # Direct mapping
    if role_lower in role_mappings:
        return role_mappings[role_lower]
    
    # Fuzzy matching for partial matches
    if "system" in role_lower or "context" in role_lower or "instruction" in role_lower:
        return "system"
    elif "user" in role_lower or "human" in role_lower or "customer" in role_lower:
        return "user"  
    elif "assistant" in role_lower or "ai" in role_lower or "bot" in role_lower or "agent" in role_lower:
        return "assistant"
    
    # Default fallback
    return "user"

def extract_messages(data: Any) -> List[Dict[str, str]]:
    """Extract messages from the file."""
    if isinstance(data, dict) and 'messages' in data:
        messages_field = data['messages']
        
        # Handle case where messages is a JSON string that needs to be parsed
        if isinstance(messages_field, str):
            try:
                messages_array = json.loads(messages_field)
                if isinstance(messages_array, list):
                    normalized_messages = []
                    for msg in messages_array:
                        if isinstance(msg, dict) and 'role' in msg and 'content' in msg:
                            # Normalize the role and ensure content exists
                            normalized_msg = {
                                'role': normalize_role(msg['role']),
                                'content': msg.get('content', '')
                            }
                            normalized_messages.append(normalized_msg)
                    return normalized_messages
            except json.JSONDecodeError:
                pass
        elif isinstance(messages_field, list):
            normalized_messages = []
            for msg in messages_field:
                if isinstance(msg, dict) and 'role' in msg and 'content' in msg:
                    # Normalize the role and ensure content exists
                    normalized_msg = {
                        'role': normalize_role(msg['role']),
                        'content': msg.get('content', '')
                    }
                    normalized_messages.append(normalized_msg)
            return normalized_messages
    
    return []

def load_trajectory_file(file_path: Path) -> tuple[List[Dict[str, str]], str]:
    """
    Load trajectory file with dual JSON/JSONL support for LLM training data.
    
    Automatically detects and processes both JSON and JSONL formats:
    - JSON: Single object or array with messages field
    - JSONL: One JSON object per line (supports multiple trajectories)
    
    Returns:
        Tuple of (messages_list, file_format_detected)
    """
    all_messages = []
    file_format = None
    
    try:
        # Detect file format and process accordingly
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                # Read first few lines to detect format
                first_lines = []
                for i, line in enumerate(f):
                    if i < 5:  # Check first 5 lines
                        first_lines.append(line.strip())
                    else:
                        break
            
            # JSONL detection: multiple lines, each is a complete JSON object
            jsonl_indicators = []
            for line in first_lines:
                if line and line.startswith('{') and line.endswith('}'):
                    try:
                        json.loads(line)
                        jsonl_indicators.append(True)
                    except json.JSONDecodeError:
                        jsonl_indicators.append(False)
            
            # If most lines are valid JSON objects, treat as JSONL
            if len(jsonl_indicators) > 0 and sum(jsonl_indicators) >= len(jsonl_indicators) * 0.6:
                file_format = "JSONL"
                return load_jsonl_format(file_path, all_messages), file_format
            else:
                file_format = "JSON"
                return load_json_format(file_path, all_messages), file_format
                
        except Exception as e:
            # Fallback: try JSON format first, then JSONL
            try:
                file_format = "JSON (fallback)"
                return load_json_format(file_path, all_messages), file_format
            except Exception:
                file_format = "JSONL (fallback)"
                return load_jsonl_format(file_path, all_messages), file_format
    
    except Exception as e:
        raise Exception(f"Failed to load trajectory file ({file_format}): {str(e)}")

def load_json_format(file_path: Path, all_messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Load single JSON file (object or array)."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        messages = extract_messages(data)
        all_messages.extend(messages)
    return all_messages

def load_jsonl_format(file_path: Path, all_messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Load JSONL file (one JSON object per line)."""
    line_count = 0
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            line_count += 1
            if line:  # Skip empty lines
                try:
                    data = json.loads(line)
                    messages = extract_messages(data)
                    if messages:
                        all_messages.extend(messages)
                except json.JSONDecodeError as e:
                    print(f"    Warning: JSON decode error on line {line_num}: {e}")
                    continue
    
    print(f"    Processed {line_count} lines from JSONL file")
    return all_messages

def is_substantial_content(content: str) -> bool:
    """Check if content is substantial enough to include (filters boilerplate)."""
    if not content or len(content.strip()) < 50:
        return False
    
    # Skip obvious boilerplate
    boilerplate_phrases = [
        "thank you", "thanks", "you're welcome", "my pleasure",
        "glad to help", "hope this helps", "let me know if",
        "is there anything else", "feel free to ask",
        "good question", "excellent question", "that's a great question"
    ]
    
    content_lower = content.lower()
    boilerplate_count = sum(1 for phrase in boilerplate_phrases if phrase in content_lower)
    
    # If more than 30% of content is boilerplate, skip it
    if boilerplate_count > 0 and len(content) < 200:
        return False
    
    return True

def create_focused_chain_of_thought_segments(messages: List[Dict[str, str]], 
                                           target_tokens: int = 6144,
                                           target_preservation: float = 70.0) -> List[Dict[str, Any]]:
    """
    Create focused segments optimized for chain of thought learning.
    Filters out boilerplate and focuses on substantial reasoning content.
    """
    if not messages:
        return []
    
    segments = []
    assistant_indices = [i for i, msg in enumerate(messages) if msg.get('role') == 'assistant']
    
    if not assistant_indices:
        return []
    
    print(f"  Found {len(assistant_indices)} assistant messages")
    
    # Calculate target total tokens based on preservation rate
    original_messages_text = "\n".join([f"{m.get('role', 'unknown')}: {m.get('content', '')}" for m in messages])
    original_tokens = count_tokens(original_messages_text)
    target_total_tokens = int(original_tokens * (target_preservation / 100.0))
    
    print(f"  Original tokens: {original_tokens}")
    print(f"  Target preservation: {target_preservation}%")
    print(f"  Target total tokens: {target_total_tokens}")
    print(f"  Max tokens per segment: {target_tokens}")
    
    # Strategy: Create focused segments by selecting substantial content
    used_indices = set()
    total_generated_tokens = 0
    tokens_remaining = target_total_tokens
    
    # Filter for substantial assistant messages
    substantial_assistants = []
    for idx in assistant_indices:
        msg = messages[idx]
        if is_substantial_content(msg.get('content', '')):
            substantial_assistants.append(idx)
    
    print(f"  Found {len(substantial_assistants)} substantial assistant responses")
    
    i = 0
    while i < len(substantial_assistants) and tokens_remaining > 1000:
        assistant_idx = substantial_assistants[i]
        
        if assistant_idx in used_indices:
            i += 1
            continue
        
        # Find optimal segment size for this assistant
        best_segment = None
        best_tokens = 0
        best_start = max(0, assistant_idx - 3)  # More context before
        best_end = assistant_idx + 1
        
        # Try different segment sizes
        for context_before in range(1, min(6, assistant_idx + 1)):
            for context_after in range(1, 4):
                start_idx = max(0, assistant_idx - context_before)
                end_idx = min(len(messages), assistant_idx + context_after + 1)
                
                # Skip if any indices are already used
                range_indices = set(range(start_idx, end_idx))
                if used_indices.intersection(range_indices):
                    continue
                
                segment_msgs = messages[start_idx:end_idx]
                
                # Ensure we have context and the target assistant
                context_msgs = [msg for msg in segment_msgs if msg.get('role') != 'assistant']
                assistant_msgs = [msg for msg in segment_msgs if msg.get('role') == 'assistant']
                
                if not context_msgs or assistant_idx not in [messages.index(msg) for msg in assistant_msgs]:
                    continue
                
                segment_text = "\n".join([f"{m['role']}: {m['content']}" for m in segment_msgs])
                segment_tokens = count_tokens(segment_text)
                
                # Prefer segments that use tokens efficiently and fit remaining budget
                if (segment_tokens <= min(target_tokens, tokens_remaining) and 
                    segment_tokens > best_tokens and 
                    segment_tokens >= 800):  # Minimum substantial segment
                    
                    best_segment = segment_msgs
                    best_tokens = segment_tokens
                    best_start = start_idx
                    best_end = end_idx
        
        # Create the best segment found
        if best_segment and best_tokens >= 800:
            target_msg = messages[assistant_idx]
            context = [msg for msg in best_segment if msg.get('role') != 'assistant']
            
            segments.append({
                'input_context': context,
                'target_action': target_msg['content'],
                'metadata': {
                    'segment_tokens': best_tokens,
                    'start_idx': best_start,
                    'end_idx': best_end,
                    'assistant_index': assistant_idx,
                    'segment_type': 'focused_coT',
                    'context_length': len(context),
                    'target_length': len(target_msg['content']) if target_msg['content'] else 0,
                    'num_assistants_in_segment': 1,
                    'quality_filter': 'substantial_content',
                    'role_normalization': True
                }
            })
            
            total_generated_tokens += best_tokens
            tokens_remaining -= best_tokens
            used_indices.update(range(best_start, best_end))
            
            print(f"    Created focused CoT segment: {best_tokens} tokens, {len(context)} context messages")
            i += 1
        else:
            # Try with minimal context if no good segment found
            context_start = max(0, assistant_idx - 2)
            context_end = min(len(messages), assistant_idx + 2)
            
            context_msgs = messages[context_start:assistant_idx]
            segment_msgs = messages[context_start:context_end]
            
            # Check if this creates a substantial segment
            target_msg = messages[assistant_idx]
            if (len(context_msgs) > 0 and 
                not used_indices.intersection(range(context_start, context_end)) and
                is_substantial_content(target_msg.get('content', ''))):
                
                segment_text = "\n".join([f"{m['role']}: {m['content']}" for m in segment_msgs])
                segment_tokens = count_tokens(segment_text)
                
                if segment_tokens >= 500 and segment_tokens <= tokens_remaining:
                    segments.append({
                        'input_context': context_msgs,
                        'target_action': target_msg['content'],
                        'metadata': {
                            'segment_tokens': segment_tokens,
                            'start_idx': context_start,
                            'end_idx': context_end,
                            'assistant_index': assistant_idx,
                            'segment_type': 'minimal_focused',
                            'context_length': len(context_msgs),
                            'target_length': len(target_msg['content']) if target_msg['content'] else 0,
                            'num_assistants_in_segment': 1,
                            'quality_filter': 'minimal_substantial',
                            'role_normalization': True
                        }
                    })
                    
                    total_generated_tokens += segment_tokens
                    tokens_remaining -= segment_tokens
                    used_indices.update(range(context_start, context_end))
                    
                    print(f"    Created minimal focused segment: {segment_tokens} tokens")
                    i += 1
                else:
                    i += 1
            else:
                i += 1
    
    # Sort segments by start_idx
    segments.sort(key=lambda x: x['metadata']['start_idx'])
    
    print(f"  Total generated tokens: {total_generated_tokens}")
    print(f"  Actual preservation: {(total_generated_tokens / original_tokens * 100):.1f}%")
    print(f"  Number of focused CoT segments: {len(segments)}")
    
    return segments

def calculate_preservation_rate(segments: List[Dict[str, Any]], original_tokens: int) -> float:
    """Calculate the actual preservation rate."""
    total_tokens = 0
    for segment in segments:
        total_tokens += segment['metadata']['segment_tokens']
    
    return (total_tokens / original_tokens * 100) if original_tokens > 0 else 0

def process_trajectory_file(file_path: Path, output_dir: Path) -> Dict[str, Any]:
    """
    Process a single trajectory file with dual JSON/JSONL support.
    
    Automatically detects JSON vs JSONL format and applies role normalization
    for system/user/assistant conversion.
    """
    result = {
        'file': file_path.name,
        'success': False,
        'error': None,
        'segments': 0,
        'preservation': 0.0,
        'file_format': None
    }
    
    try:
        # Load file with dual JSON/JSONL support and format detection
        messages, file_format = load_trajectory_file(file_path)
        
        if not messages:
            result['error'] = "No messages found"
            return result
        
        print(f"  Detected format: {file_format}")
        print(f"  Messages: {len(messages)}, Assistant: {sum(1 for m in messages if m.get('role') == 'assistant')}")
        
        # Show role distribution for verification (auto-conversion to system/user/assistant)
        role_counts = {}
        for msg in messages:
            role = msg.get('role', 'unknown')
            role_counts[role] = role_counts.get(role, 0) + 1
        print(f"  Role distribution: {role_counts}")
        print(f"  Auto-role conversion: system/user/assistant format applied")
        
        # Create focused segments optimized for chain of thought learning
        segments = create_focused_chain_of_thought_segments(messages)
        
        if not segments:
            result['error'] = "Could not create segments"
            return result
        
        # Save to file in JSONL format for LLM training
        output_file = output_dir / f"{file_path.stem}_final.jsonl"
        with open(output_file, 'w', encoding='utf-8') as f:
            for segment in segments:
                f.write(json.dumps(segment, ensure_ascii=False) + '\n')
        
        # Calculate preservation rate
        original_messages_text = "\n".join([f"{m.get('role', 'unknown')}: {m.get('content', '')}" for m in messages])
        original_tokens = count_tokens(original_messages_text)
        preservation = calculate_preservation_rate(segments, original_tokens)
        
        result['success'] = True
        result['segments'] = len(segments)
        result['preservation'] = preservation
        result['output_file'] = str(output_file)
        result['original_tokens'] = original_tokens
        result['file_format'] = file_format
        
        print(f"  Created {len(segments)} final focused CoT segments, {preservation:.1f}% preservation")
        print(f"  Role normalization: Applied to all messages")
        print(f"  Output format: JSONL for LLM training")
        
        # Show segment types and token stats
        segment_types = {}
        token_counts = []
        context_lengths = []
        target_lengths = []
        assistant_counts = []
        
        for s in segments:
            seg_type = s['metadata']['segment_type']
            segment_types[seg_type] = segment_types.get(seg_type, 0) + 1
            token_counts.append(s['metadata']['segment_tokens'])
            context_lengths.append(s['metadata']['context_length'])
            target_lengths.append(s['metadata']['target_length'])
            assistant_counts.append(s['metadata'].get('num_assistants_in_segment', 1))
        
        print(f"  Segment types: {segment_types}")
        if token_counts:
            print(f"  Token stats - Min: {min(token_counts)}, Max: {max(token_counts)}, Avg: {np.mean(token_counts):.0f}")
            print(f"  Token utilization - Min: {min(token_counts)/6144*100:.1f}%, Max: {max(token_counts)/6144*100:.1f}%, Avg: {np.mean(token_counts)/6144*100:.1f}%")
            print(f"  Context length - Avg: {np.mean(context_lengths):.1f} messages")
            print(f"  Target length - Avg: {np.mean(target_lengths):.0f} characters")
            print(f"  Assistants per segment - Avg: {np.mean(assistant_counts):.1f}")
        
    except Exception as e:
        result['error'] = str(e)
    
    return result

# =============================================================================
# WORKSPACE EXPORT FUNCTIONALITY
# =============================================================================

def detect_encoding(file_path: Path) -> str:
    """Detect file encoding using chardet."""
    try:
        with open(file_path, 'rb') as f:
            raw_data = f.read(10000)  # Read first 10KB for detection
            result = chardet.detect(raw_data)
            return result.get('encoding', 'utf-8')
    except Exception:
        return 'utf-8'  # Fallback to utf-8

def get_mime_type(file_path: Path) -> str:
    """Get MIME type for a file."""
    mime_type, _ = mimetypes.guess_type(str(file_path))
    return mime_type or 'application/octet-stream'

def read_file_content(file_path: Path, encoding: str) -> tuple[str, bool, int]:
    """Read file content and return content, is_text, and line count."""
    try:
        with open(file_path, 'r', encoding=encoding, errors='ignore') as f:
            content = f.read()
            lines = content.split('\n')
            return content, True, len(lines)
    except Exception:
        # If text reading fails, try binary reading
        try:
            with open(file_path, 'rb') as f:
                content = f.read()
                # Try to decode as UTF-8, replace errors
                content_str = content.decode('utf-8', errors='replace')
                lines = content_str.split('\n')
                return content_str, False, len(lines)
        except Exception as e:
            return f"[Error reading file: {str(e)}]", False, 0

def get_file_info(file_path: Path) -> Dict[str, Any]:
    """Get comprehensive information about a file."""
    try:
        stat = file_path.stat()
        
        # Detect encoding and MIME type
        encoding = detect_encoding(file_path)
        mime_type = get_mime_type(file_path)
        
        # Read content
        content, is_text, line_count = read_file_content(file_path, encoding)
        
        # Generate preview
        preview = content[:200] if isinstance(content, str) else str(content)[:200]
        
        return {
            "file_path": str(file_path.relative_to(Path.cwd())),
            "file_name": file_path.name,
            "file_extension": file_path.suffix[1:] if file_path.suffix else "",
            "file_size": stat.st_size,
            "mime_type": mime_type,
            "encoding": encoding,
            "is_text_file": is_text,
            "line_count": line_count,
            "content": content,
            "content_preview": preview,
            "last_modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "created_timestamp": datetime.fromtimestamp(stat.st_ctime).isoformat(),
            "access_timestamp": datetime.fromtimestamp(stat.st_atime).isoformat()
        }
    except Exception as e:
        return {
            "file_path": str(file_path.relative_to(Path.cwd())),
            "file_name": file_path.name,
            "error": f"Failed to process file: {str(e)}"
        }

def export_workspace(output_file: str = "workspace_export.jsonl") -> Dict[str, Any]:
    """Export all files in the workspace to a JSONL file."""
    
    print(f"Starting workspace export...")
    print(f"Current directory: {Path.cwd()}")
    
    # Find all files in the workspace
    workspace_path = Path.cwd()
    all_files = []
    
    # Walk through all directories and files
    for file_path in workspace_path.rglob('*'):
        if file_path.is_file() and not any(part.startswith('.') for part in file_path.parts):
            all_files.append(file_path)
    
    print(f"Found {len(all_files)} files to export")
    
    # Export to JSONL
    results = []
    success_count = 0
    error_count = 0
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for i, file_path in enumerate(all_files, 1):
            print(f"[{i}/{len(all_files)}] Processing: {file_path.name}")
            
            file_info = get_file_info(file_path)
            
            # Write to JSONL
            json_line = json.dumps(file_info, ensure_ascii=False, separators=(',', ':'))
            f.write(json_line + '\n')
            
            results.append(file_info)
            
            if 'error' not in file_info:
                success_count += 1
            else:
                error_count += 1
    
    summary = {
        "export_timestamp": datetime.now().isoformat(),
        "total_files": len(all_files),
        "successful": success_count,
        "failed": error_count,
        "output_file": output_file,
        "workspace_path": str(workspace_path),
        "file_summary": {
            "total_size_bytes": sum(r.get('file_size', 0) for r in results if 'file_size' in r),
            "text_files": sum(1 for r in results if r.get('is_text_file', False)),
            "binary_files": sum(1 for r in results if not r.get('is_text_file', True)),
            "extensions": list(set(r.get('file_extension', '') for r in results if r.get('file_extension')))
        }
    }
    
    print(f"\nExport completed!")
    print(f"Total files: {summary['total_files']}")
    print(f"Successful: {summary['successful']}")
    print(f"Failed: {summary['failed']}")
    print(f"Total size: {summary['file_summary']['total_size_bytes']:,} bytes")
    print(f"Text files: {summary['file_summary']['text_files']}")
    print(f"Binary files: {summary['file_summary']['binary_files']}")
    print(f"Output file: {output_file}")
    
    return summary

def main():
    # Check command line arguments
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        
        if mode == "export":
            # Export workspace to JSONL
            output_file = sys.argv[2] if len(sys.argv) > 2 else "workspace_export.jsonl"
            if not output_file.endswith('.jsonl'):
                output_file += '.jsonl'
            
            print("="*80)
            print("WORKSPACE EXPORT MODE")
            print("Exporting all files in workspace to JSONL format")
            print("="*80)
            
            try:
                summary = export_workspace(output_file)
                print(f"\nWorkspace export completed successfully!")
                return 0
            except Exception as e:
                print(f"Error during export: {str(e)}")
                return 1
        
        elif mode == "help" or mode == "-h" or mode == "--help":
            print("TrajectoryScraper - Focused Chain of Thought Version")
            print("="*60)
            print("Usage:")
            print("  python TrajectoryScraper_final.py              # Run trajectory segmentation")
            print("  python TrajectoryScraper_final.py export       # Export workspace to JSONL")
            print("  python TrajectoryScraper_final.py export [file] # Export to custom file")
            print("  python TrajectoryScraper_final.py help         # Show this help")
            print("\nModes:")
            print("  (default)  - Process trajectory files with focused CoT segmentation")
            print("  export     - Export all workspace files to JSONL format")
            print("\nFocused CoT Features:")
            print("  - Filters out boilerplate and back-and-forth conversation")
            print("  - Target preservation: 70% (focuses on substantial content)")
            print("  - Role normalization: Converts to system/user/assistant format")
            print("  - Supports both JSON and JSONL input formats")
            print("  - Optimized for high-quality chain of thought training")
            return 0
    
    # Default mode: trajectory segmentation
    print("="*80)
    print("FOCUSED CHAIN OF THOUGHT TRAJECTORY SEGMENTATION")
    print("Filters boilerplate and focuses on substantial reasoning content")
    print("Target preservation: 70% (selects best content for CoT learning)")
    print("Role normalization: Converts all roles to system/user/assistant format")
    print("="*80)
    
    # Find JSON files
    json_files = []
    for pattern in ['*.json', '*.jsonl']:
        json_files.extend(Path('.').rglob(pattern))
    
    # Filter out already processed files
    json_files = [f for f in json_files if not any(
        s in f.name for s in ['_segmented', '_compressed', '_reasoning', '_highpres', '_debug', '_final']
    )]
    
    print(f"Found {len(json_files)} JSON files to process.")
    
    # Create output directory
    output_dir = Path("segmented_trajectories")
    output_dir.mkdir(exist_ok=True)
    
    results = []
    
    for i, file_path in enumerate(json_files, 1):
        print(f"\n[{i}/{len(json_files)}] Processing: {file_path.name}")
        
        result = process_trajectory_file(file_path, output_dir)
        results.append(result)
    
    # Print summary
    print("\n" + "="*80)
    print("PROCESSING SUMMARY")
    print("="*80)
    
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]
    
    if successful:
        print(f"Successfully processed: {len(successful)}/{len(results)} files")
        
        total_segments = sum(r['segments'] for r in successful)
        avg_segments = np.mean([r['segments'] for r in successful])
        avg_preservation = np.mean([r['preservation'] for r in successful])
        
        print(f"\nTotal segments created: {total_segments}")
        print(f"Average segments per file: {avg_segments:.1f}")
        print(f"Average preservation: {avg_preservation:.1f}%")
        
        # Verify preservation target (70% for focused CoT learning)
        if 65.0 <= avg_preservation <= 75.0:
            print(f"OK Preservation rate WITHIN target range (65-75% for focused CoT)")
        else:
            print(f"! Preservation rate: {avg_preservation:.1f}% (target: 65-75% for focused CoT)")
        
        # Show file details
        print(f"\nFile details:")
        for result in successful:
            print(f"  {result['file']}:")
            print(f"    - Segments: {result['segments']}")
            print(f"    - Preservation: {result['preservation']:.1f}%")
            print(f"    - Output: {Path(result['output_file']).name}")
    
    if failed:
        print(f"\nFailed files ({len(failed)}):")
        for result in failed:
            print(f"  {result['file']}: {result['error']}")
    
    print(f"\nOutput directory: {output_dir}")
    print(f"Role normalization: Applied to all segments")

if __name__ == "__main__":
    exit(main())