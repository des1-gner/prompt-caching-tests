import boto3
import json
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

bedrock_runtime = boto3.client('bedrock-runtime', region_name='us-east-1')
MODEL_ID = 'us.anthropic.claude-sonnet-4-5-20250929-v1:0'

def create_body(query, use_cache=True):
    static_text = "This is static context that will be cached. " * 300  # >1024 tokens
    
    content = [
        {"type": "text", "text": static_text}
    ]
    
    if use_cache:
        content.append({
            "type": "text",
            "text": "End of static content.",
            "cache_control": {"type": "ephemeral"}
        })
    
    content.append({"type": "text", "text": query})
    
    return {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 512,
        "messages": [{"role": "user", "content": content}]
    }

def invoke(req_id, query, use_cache=True):
    start = time.time()
    response = bedrock_runtime.invoke_model(
        modelId=MODEL_ID,
        body=json.dumps(create_body(query, use_cache))
    )
    latency = time.time() - start
    
    result = json.loads(response['body'].read())
    usage = result.get('usage', {})
    
    return {
        'id': req_id,
        'latency': latency,
        'input': usage.get('input_tokens', 0),
        'cache_read': usage.get('cache_read_input_tokens', 0),
        'cache_write': usage.get('cache_creation_input_tokens', 0),
        'output': usage.get('output_tokens', 0)
    }

def print_results(results, title):
    print(f"\n{'='*70}\n{title}\n{'='*70}")
    for r in results:
        print(f"Request {r['id']}: latency={r['latency']:.2f}s | "
              f"input={r['input']} | write={r['cache_write']} | read={r['cache_read']}")
    
    total_writes = sum(r['cache_write'] for r in results)
    total_reads = sum(r['cache_read'] for r in results)
    avg_latency = sum(r['latency'] for r in results) / len(results)
    
    print(f"\nTotal cache writes: {total_writes:,}")
    print(f"Total cache reads: {total_reads:,}")
    print(f"Average latency: {avg_latency:.2f}s")

if __name__ == "__main__":
    print(f"\n{datetime.now().isoformat()} | Model: {MODEL_ID}")
    
    # Test 1: Initial concurrent requests
    print("\n[TEST 1] Initial concurrent requests (expect multiple cache writes)")
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(invoke, i, f"Question {i}?") for i in range(1, 6)]
        test1_results = [f.result() for f in futures]
    print_results(test1_results, "TEST 1: Concurrent Initial")
    
    time.sleep(2)
    
    # Test 2: Sequential after cache exists
    print("\n[TEST 2] Sequential requests (expect cache reads)")
    test2_results = []
    for i in range(1, 6):
        test2_results.append(invoke(i, f"Question {i}?"))
        time.sleep(0.5)
    print_results(test2_results, "TEST 2: Sequential After Cache")
    
    # Test 3: Pre-warm then concurrent
    print("\n[TEST 3] Pre-warm strategy")
    print("Pre-warming...")
    prewarm = invoke(0, "Warmup question?")
    print(f"Pre-warm: write={prewarm['cache_write']}, read={prewarm['cache_read']}")
    
    time.sleep(2)
    
    print("Concurrent after pre-warm...")
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(invoke, i, f"Question {i}?") for i in range(1, 6)]
        test3_results = [f.result() for f in futures]
    print_results(test3_results, "TEST 3: Concurrent After Pre-warm")
    
    # Test 4: Without caching
    print("\n[TEST 4] Without caching (baseline)")
    test4_results = []
    for i in range(1, 4):
        test4_results.append(invoke(i, f"Question {i}?", use_cache=False))
        time.sleep(0.5)
    print_results(test4_results, "TEST 4: No Caching")
    
    # Test 5: Concurrent after pre-warm (repeated to verify consistency)
    print("\n[TEST 5] Repeat concurrent (verify cache behavior)")
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(invoke, i, f"Question {i}?") for i in range(1, 6)]
        test5_results = [f.result() for f in futures]
    print_results(test5_results, "TEST 5: Concurrent Repeat")
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("Test 1 (concurrent initial): Shows multiple cache writes")
    print("Test 2 (sequential): Shows cache reads work")
    print("Test 3 (pre-warm): Shows optimized pattern")
    print("Test 4 (no cache): Shows baseline cost/latency")
    print("Test 5 (concurrent repeat): Shows stable cache behavior")
