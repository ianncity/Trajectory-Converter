import json
import tiktoken
import numpy as np
import sys
from pathlib import Path
from typing import Dict, List, Any
import math

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

def create_balanced_segments(messages: List[Dict[str, str]], 
                            target_tokens: int = 6144,
                            target_preservation: float = 60.0) -> List[Dict[str, Any]]:
    """Create segments that balance token utilization with target preservation rate."""
    if not messages:
        return []
    
    segments = []
    assistant_indices = [i for i, msg in enumerate(messages) if msg.get('role') == 'assistant']
    
    if not assistant_indices:
        return []
    
    print(f"  Found {len(assistant_indices)} assistant messages")
    
    # Calculate target total tokens based on desired preservation rate
    original_messages_text = "\n".join([f"{m.get('role', 'unknown')}: {m.get('content', '')}" for m in messages])
    original_tokens = count_tokens(original_messages_text)
    target_total_tokens = int(original_tokens * (target_preservation / 100.0))
    target_avg_tokens = target_total_tokens / len(assistant_indices)
    
    print(f"  Original tokens: {original_tokens}")
    print(f"  Target preservation: {target_preservation}%")
    print(f"  Target total tokens: {target_total_tokens}")
    print(f"  Target avg tokens per segment: {target_avg_tokens:.0f}")
    
    # Strategy: Create segments with optimal size based on target preservation
    used_indices = set()
    total_generated_tokens = 0
    
    for idx in assistant_indices:
        if idx in used_indices:
            continue
            
        # Find optimal segment size that gets us close to target preservation
        best_segment = None
        best_tokens = 0
        best_size = 0
        
        # Try different segment sizes, focusing on getting closer to target_avg_tokens
        for segment_size in range(min(20, len(messages) - idx), 2, -1):
            end_idx = min(len(messages), idx + segment_size)
            if end_idx > len(messages):
                continue
                
            # Check if any indices in this range are already used
            range_indices = set(range(idx, end_idx))
            if used_indices.intersection(range_indices):
                continue
                
            segment_msgs = messages[idx:end_idx]
            
            # Ensure this segment includes the assistant message
            has_assistant = any(msg.get('role') == 'assistant' for msg in segment_msgs)
            if not has_assistant:
                continue
            
            segment_text = "\n".join([f"{m['role']}: {m['content']}" for m in segment_msgs])
            segment_tokens = count_tokens(segment_text)
            
            # Prefer segments that get us close to target_avg_tokens
            if segment_tokens <= target_tokens and abs(segment_tokens - target_avg_tokens) < abs(best_tokens - target_avg_tokens):
                best_segment = segment_msgs
                best_tokens = segment_tokens
                best_size = segment_size
        
        # If we found a good segment, use it; otherwise create minimal segment
        if best_segment and best_tokens >= target_avg_tokens * 0.3:  # At least 30% of target
            # Find the assistant message in this segment
            target_msg = None
            target_idx_in_segment = -1
            for i, msg in enumerate(best_segment):
                if msg.get('role') == 'assistant':
                    target_msg = msg
                    target_idx_in_segment = i
                    break
            
            if target_msg and target_idx_in_segment > 0:
                context = best_segment[:target_idx_in_segment]
                
                segments.append({
                    'input_context': context,
                    'target_action': target_msg['content'],
                    'metadata': {
                        'segment_tokens': best_tokens,
                        'start_idx': idx,
                        'end_idx': idx + len(best_segment),
                        'assistant_index': idx + target_idx_in_segment,
                        'segment_type': 'optimized'
                    }
                })
                
                total_generated_tokens += best_tokens
                used_indices.update(range(idx, idx + len(best_segment)))
        
        else:
            # Create minimal segment for remaining assistants
            start_idx = max(0, idx - 1)
            end_idx = min(len(messages), idx + 2)
            
            # Check if this range overlaps with used indices
            range_indices = set(range(start_idx, end_idx))
            if used_indices.intersection(range_indices):
                continue
                
            segment_msgs = messages[start_idx:end_idx]
            
            # Ensure we have at least one context message and the assistant
            context_messages = [msg for msg in segment_msgs if msg.get('role') != 'assistant']
            assistant_messages = [msg for msg in segment_msgs if msg.get('role') == 'assistant']
            
            if context_messages and assistant_messages:
                target_msg = assistant_messages[0]
                context = context_messages
                
                segment_text = "\n".join([f"{m['role']}: {m['content']}" for m in segment_msgs])
                segment_tokens = count_tokens(segment_text)
                
                segments.append({
                    'input_context': context,
                    'target_action': target_msg['content'],
                    'metadata': {
                        'segment_tokens': segment_tokens,
                        'start_idx': start_idx,
                        'end_idx': end_idx,
                        'assistant_index': idx,
                        'segment_type': 'minimal'
                    }
                })
                
                total_generated_tokens += segment_tokens
                used_indices.update(range_indices)
    
    # Sort segments by start_idx
    segments.sort(key=lambda x: x['metadata']['start_idx'])
    
    print(f"  Total generated tokens: {total_generated_tokens}")
    print(f"  Actual preservation: {(total_generated_tokens / original_tokens * 100):.1f}%")
    
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
        
        # Create segments
        segments = create_balanced_segments(messages)
        
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
        
        print(f"  Created {len(segments)} final segments, {preservation:.1f}% preservation")
        
        # Show segment types and token stats
        segment_types = {}
        token_counts = []
        for s in segments:
            seg_type = s['metadata']['segment_type']
            segment_types[seg_type] = segment_types.get(seg_type, 0) + 1
            token_counts.append(s['metadata']['segment_tokens'])
        
        print(f"  Segment types: {segment_types}")
        if token_counts:
            print(f"  Token stats - Min: {min(token_counts)}, Max: {max(token_counts)}, Avg: {np.mean(token_counts):.0f}")
            print(f"  Token utilization - Min: {min(token_counts)/6144*100:.1f}%, Max: {max(token_counts)/6144*100:.1f}%, Avg: {np.mean(token_counts)/6144*100:.1f}%")
        
    except Exception as e:
        result['error'] = str(e)
    
    return result

def main():
    print("="*80)
    print("FINAL OPTIMIZED TRAJECTORY SEGMENTATION")
    print("Balanced approach: Target 50-70% preservation with optimal token utilization")
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
        
        # Verify preservation target
        if 50.0 <= avg_preservation <= 70.0:
            print(f"OK Preservation rate WITHIN target range (50-70%)")
        else:
            print(f"! Preservation rate: {avg_preservation:.1f}% (target: 50-70%)")
        
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
    main()