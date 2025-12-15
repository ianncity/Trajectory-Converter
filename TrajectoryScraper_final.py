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

def extract_messages(data: Any) -> List[Dict[str, str]]:
    """Extract messages from the file."""
    if isinstance(data, dict) and 'messages' in data:
        messages = data['messages']
        if isinstance(messages, list):
            return messages
    return []

def create_chain_of_thought_segments(messages: List[Dict[str, str]], 
                                   target_tokens: int = 6144,
                                   target_preservation: float = 80.0) -> List[Dict[str, Any]]:
    """
    Create longer segments optimized for chain of thought learning.
    Combines multiple neighboring assistant responses up to 6144 tokens.
    """
    if not messages:
        return []
    
    segments = []
    assistant_indices = [i for i, msg in enumerate(messages) if msg.get('role') == 'assistant']
    
    if not assistant_indices:
        return []
    
    print(f"  Found {len(assistant_indices)} assistant messages")
    
    # Calculate target total tokens based on higher preservation rate for CoT learning
    original_messages_text = "\n".join([f"{m.get('role', 'unknown')}: {m.get('content', '')}" for m in messages])
    original_tokens = count_tokens(original_messages_text)
    target_total_tokens = int(original_tokens * (target_preservation / 100.0))
    
    print(f"  Original tokens: {original_tokens}")
    print(f"  Target preservation: {target_preservation}%")
    print(f"  Target total tokens: {target_total_tokens}")
    print(f"  Max tokens per segment: {target_tokens}")
    
    # Strategy: Create large segments by combining multiple assistant responses
    used_indices = set()
    total_generated_tokens = 0
    
    i = 0
    while i < len(assistant_indices):
        if assistant_indices[i] in used_indices:
            i += 1
            continue
        
        # Start a new segment from this assistant
        current_assistant_idx = assistant_indices[i]
        segment_start = max(0, current_assistant_idx - 2)  # Include some context before
        
        # Determine the end of this segment
        segment_end = current_assistant_idx + 1  # Include this assistant response
        
        # Try to expand the segment to include more assistant responses
        j = i + 1
        while j < len(assistant_indices):
            next_assistant_idx = assistant_indices[j]
            
            # Check if adding the next assistant response would exceed token limit
            test_end = next_assistant_idx + 1
            test_segment_msgs = messages[segment_start:test_end]
            test_text = "\n".join([f"{m['role']}: {m['content']}" for m in test_segment_msgs])
            test_tokens = count_tokens(test_text)
            
            if test_tokens <= target_tokens:
                segment_end = test_end
                j += 1
            else:
                break
        
        # Create the segment
        segment_msgs = messages[segment_start:segment_end]
        
        # Find the target assistant message (the last one in this segment)
        target_msg = None
        target_idx_in_segment = -1
        for k, msg in enumerate(segment_msgs):
            if msg.get('role') == 'assistant' and messages.index(msg) == assistant_indices[j-1]:
                target_msg = msg
                target_idx_in_segment = k
                break
        
        # If we didn't find the exact target, use the last assistant in the segment
        if not target_msg:
            for k in range(len(segment_msgs) - 1, -1, -1):
                if segment_msgs[k].get('role') == 'assistant':
                    target_msg = segment_msgs[k]
                    target_idx_in_segment = k
                    break
        
        if target_msg and target_idx_in_segment > 0:  # Need some context before assistant
            context = segment_msgs[:target_idx_in_segment]
            
            segment_text = "\n".join([f"{m['role']}: {m['content']}" for m in segment_msgs])
            segment_tokens = count_tokens(segment_text)
            
            # Only create segment if it has substantial content
            if segment_tokens >= 1000:  # Minimum 1000 tokens for meaningful CoT learning
                segments.append({
                    'input_context': context,
                    'target_action': target_msg['content'],
                    'metadata': {
                        'segment_tokens': segment_tokens,
                        'start_idx': segment_start,
                        'end_idx': segment_end,
                        'assistant_index': segment_start + target_idx_in_segment,
                        'segment_type': 'chain_of_thought',
                        'context_length': len(context),
                        'target_length': len(target_msg['content']) if target_msg['content'] else 0,
                        'num_assistants_in_segment': sum(1 for msg in segment_msgs if msg.get('role') == 'assistant')
                    }
                })
                
                total_generated_tokens += segment_tokens
                used_indices.update(range(segment_start, segment_end))
                
                print(f"    Created CoT segment: {segment_tokens} tokens, {len(context)} context messages, {sum(1 for msg in segment_msgs if msg.get('role') == 'assistant')} assistants")
                
                # Move to the next unprocessed assistant
                i = j
            else:
                # If segment is too small, create a minimal one
                context_start = max(0, current_assistant_idx - 3)
                context_end = min(len(messages), current_assistant_idx + 2)
                
                context_msgs = messages[context_start:current_assistant_idx]
                segment_msgs = messages[context_start:context_end]
                
                segment_text = "\n".join([f"{m['role']}: {m['content']}" for m in segment_msgs])
                segment_tokens = count_tokens(segment_text)
                
                if segment_tokens >= 500:  # Lower threshold for small segments
                    segments.append({
                        'input_context': context_msgs,
                        'target_action': target_msg['content'],
                        'metadata': {
                            'segment_tokens': segment_tokens,
                            'start_idx': context_start,
                            'end_idx': context_end,
                            'assistant_index': current_assistant_idx,
                            'segment_type': 'minimal_coT',
                            'context_length': len(context_msgs),
                            'target_length': len(target_msg['content']) if target_msg['content'] else 0,
                            'num_assistants_in_segment': 1
                        }
                    })
                    
                    total_generated_tokens += segment_tokens
                    used_indices.update(range(context_start, context_end))
                    
                    print(f"    Created minimal CoT segment: {segment_tokens} tokens, {len(context_msgs)} context messages")
                    i += 1
                else:
                    i += 1
        else:
            i += 1
    
    # Sort segments by start_idx
    segments.sort(key=lambda x: x['metadata']['start_idx'])
    
    print(f"  Total generated tokens: {total_generated_tokens}")
    print(f"  Actual preservation: {(total_generated_tokens / original_tokens * 100):.1f}%")
    print(f"  Number of CoT segments: {len(segments)}")
    
    return segments

def calculate_preservation_rate(segments: List[Dict[str, Any]], original_tokens: int) -> float:
    """Calculate the actual preservation rate."""
    total_tokens = 0
    for segment in segments:
        total_tokens += segment['metadata']['segment_tokens']
    
    return (total_tokens / original_tokens * 100) if original_tokens > 0 else 0

def process_trajectory_file(file_path: Path, output_dir: Path) -> Dict[str, Any]:
    """Process a single trajectory file."""
    result = {
        'file': file_path.name,
        'success': False,
        'error': None,
        'segments': 0,
        'preservation': 0.0
    }
    
    try:
        # Load file
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract messages
        messages = extract_messages(data)
        
        if not messages:
            result['error'] = "No messages found"
            return result
        
        print(f"  Messages: {len(messages)}, Assistant: {sum(1 for m in messages if m.get('role') == 'assistant')}")
        
        # Create segments optimized for chain of thought learning
        segments = create_chain_of_thought_segments(messages)
        
        if not segments:
            result['error'] = "Could not create segments"
            return result
        
        # Save to file
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
        
        print(f"  Created {len(segments)} final CoT segments, {preservation:.1f}% preservation")
        
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
            print("TrajectoryScraper - Chain of Thought Optimized Version")
            print("="*60)
            print("Usage:")
            print("  python TrajectoryScraper_final.py              # Run trajectory segmentation")
            print("  python TrajectoryScraper_final.py export       # Export workspace to JSONL")
            print("  python TrajectoryScraper_final.py export [file] # Export to custom file")
            print("  python TrajectoryScraper_final.py help         # Show this help")
            print("\nModes:")
            print("  (default)  - Process trajectory files with CoT-optimized segmentation")
            print("  export     - Export all workspace files to JSONL format")
            print("\nCoT Features:")
            print("  - Combines multiple neighboring segments up to 6144 tokens")
            print("  - Optimized for chain of thought learning")
            print("  - Higher preservation rate (80% vs 60%)")
            print("  - Groups multiple assistant responses for better coherence")
            return 0
    
    # Default mode: trajectory segmentation
    print("="*80)
    print("CHAIN OF THOUGHT OPTIMIZED TRAJECTORY SEGMENTATION")
    print("Combines neighboring segments up to 6144 tokens for better reasoning")
    print("Target preservation: 80% (increased from 60% for CoT learning)")
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
        
        # Verify preservation target (now 80% for CoT learning)
        if 70.0 <= avg_preservation <= 90.0:
            print(f"OK Preservation rate WITHIN target range (70-90% for CoT)")
        else:
            print(f"! Preservation rate: {avg_preservation:.1f}% (target: 70-90% for CoT)")
        
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

if __name__ == "__main__":
    exit(main())