"""
simulation demonstrating the Paged K-V Cache Memory Manager.
"""

from block_manager import BlockManager
from sequence import Sequence




def print_status(manager: BlockManager, sequences: dict):
    """Print current memory status."""
    print(f"Free blocks: {manager.get_num_free_blocks()}/{manager.num_blocks}")
    if sequences:
        print("Active sequences:")
        for seq_id, seq in sequences.items():
            blocks_str = ", ".join(str(b) for b in seq.block_table.values())
            print(f"  - {seq_id}: {seq.token_count} tokens, "
                  f"{seq.get_num_blocks()} blocks (physical: [{blocks_str}])")
    print()


def test_case_1_simple_run():
    """
    Test Case 1: Simple Run
    
    Demonstrates basic allocation and shows logical-to-physical mapping.
    """

    print("TEST CASE 1: Simple Run\n")
    
    # Initialize manager
    print("Create BlockManager with 100 blocks, block_size=16")
    manager = BlockManager(
        num_blocks=100,
        block_size=16,
        num_layers=4,
        num_heads=8,
        head_size=64,
        device="cpu"
    )
    print(f"Total GPU memory allocated: "
          f"{manager.key_cache.element_size() * manager.key_cache.nelement() * 2 / (1024**2):.2f} MB")
    print()
    
    # Create sequence and allocate tokens
    print("Creating sequence 'seq_A'")
    seq_a = Sequence("seq_A")
    
    print("Appending 30 tokens to seq_A...")
    seq_a.append_tokens(manager, 30)
    
    print(f"seq_A allocated {seq_a.get_num_blocks()} blocks for {seq_a.token_count} tokens")
    print(f"Block table (logical → physical): {seq_a.block_table}")
    print(f"Free blocks: {manager.get_num_free_blocks()}/{manager.num_blocks}")
    print()
    
    # Show physical location of some tokens
    print("Physical locations:")
    for token_idx in [0, 15, 16, 29]:
        physical_block, offset = seq_a.get_physical_block(token_idx, manager)
        print(f"  Token {token_idx:2d} → Physical block {physical_block}, offset {offset}")
    print()
    
    # Clean up
    print("Freeing seq_A blocks...")
    seq_a.free_all_blocks(manager)
    print(f"Free blocks: {manager.get_num_free_blocks()}/{manager.num_blocks}")
    print()


def test_case_2_fragmentation_demo():
    """
    Test Case 2: Concurrency & Fragmentation - THE KEY DEMO
    
    Demonstrates how paged memory solves external fragmentation:
    1. Allocate blocks to seq_1 and seq_2 (80/100 blocks used)
    2. Try to allocate seq_3 (needs 30 blocks) - FAILS (only 20 free)
    3. seq_1 finishes and frees its blocks (now 60 free, but non-contiguous)
    4. Try to allocate seq_3 again - SUCCESS! (works despite non-contiguous blocks)
    """

    print("TEST CASE 2: Concurrency & Fragmentation (THE KEY DEMO)\n")
    
    
    # Initialize manager
    print("Create BlockManager with 100 blocks, block_size=16")
    manager = BlockManager(
        num_blocks=100,
        block_size=16,
        num_layers=4,
        num_heads=8,
        head_size=64,
        device="cpu"
    )
    print()
    
    # Step 1: Allocate seq_1
    print("Create seq_1 with 640 tokens (40 blocks)")
    seq_1 = Sequence("seq_1")
    seq_1.append_tokens(manager, 640)
    print(f"Allocated {seq_1.get_num_blocks()} blocks for seq_1")
    print(f"Physical blocks: {sorted(seq_1.block_table.values())[:5]}...{sorted(seq_1.block_table.values())[-5:]}")
    print_status(manager, {"seq_1": seq_1})
    
    # Step 2: Allocate seq_2
    print("Create seq_2 with 640 tokens (40 blocks)")
    seq_2 = Sequence("seq_2")
    seq_2.append_tokens(manager, 640)
    print(f"Allocated {seq_2.get_num_blocks()} blocks for seq_2")
    print(f"Physical blocks: {sorted(seq_2.block_table.values())[:5]}...{sorted(seq_2.block_table.values())[-5:]}")
    print_status(manager, {"seq_1": seq_1, "seq_2": seq_2})
    
    # Step 3: Try to allocate seq_3 (should fail)
    print("Attempt to create seq_3 with 480 tokens (30 blocks)")
    seq_3 = Sequence("seq_3")
    try:
        seq_3.append_tokens(manager, 480)
        print("SUCCESS - seq_3 allocated")
    except RuntimeError as e:
        print(f"Error - {e}")
        print("Expected: Only 20 free blocks, but seq_3 needs 30 blocks")
    print()
    
    # Step 4: Free seq_1 and try seq_3 again
    print("[STEP 4] seq_1 FINISHED - Freeing 40 blocks")
    seq_1.free_all_blocks(manager)
    print(f"[SIM] Free blocks: {manager.get_num_free_blocks()}/{manager.num_blocks}")
    print("[SIM] Note: These 60 free blocks are NON-CONTIGUOUS in physical memory")
    print("[SIM] In naive allocation, we still couldn't allocate seq_3!")
    print()
    
    # Step 5: Try to allocate seq_3 again (should succeed now)
    print("[STEP 5] Attempting to create seq_3 again (needs 30 blocks)")
    seq_3 = Sequence("seq_3")
    try:
        seq_3.append_tokens(manager, 480)
        print(f"[SIM] ✓ SUCCESS! seq_3 allocated {seq_3.get_num_blocks()} blocks")
        print(f"[SIM] Physical blocks: {sorted(seq_3.block_table.values())[:5]}...{sorted(seq_3.block_table.values())[-5:]}")
        print()
        print_status(manager, {"seq_2": seq_2, "seq_3": seq_3})
        print("[SIM] ✓ Paged memory solved the fragmentation problem!")
        print("[SIM] ✓ seq_3 uses non-contiguous physical blocks seamlessly")
    except RuntimeError as e:
        print(f"[SIM] ✗ FAILED - {e}")
    print()


def test_case_3_high_utilization():
    """
    Test Case 3: High Memory Utilization
    
    Demonstrates near-100% memory utilization with multiple sequences.
    """

    print("TEST CASE 3: High Memory Utilization\n")

    
    print("[INIT] Creating BlockManager with 50 blocks, block_size=16")
    manager = BlockManager(
        num_blocks=50,
        block_size=16,
        num_layers=4,
        num_heads=8,
        head_size=64,
        device="cpu"
    )
    print()
    
    # Create multiple sequences with varying sizes
    sequences = {}
    allocations = [
        ("req_1", 160),  # 10 blocks
        ("req_2", 80),   # 5 blocks
        ("req_3", 240),  # 15 blocks
        ("req_4", 128),  # 8 blocks
        ("req_5", 176),  # 11 blocks
    ]
    
    print("[SIM] Allocating multiple sequences:")
    for seq_id, num_tokens in allocations:
        seq = Sequence(seq_id)
        seq.append_tokens(manager, num_tokens)
        sequences[seq_id] = seq
        num_blocks = num_tokens // manager.block_size
        print(f"  - {seq_id}: {num_tokens} tokens ({num_blocks} blocks)")
    
    print()
    print_status(manager, sequences)
    
    total_blocks_used = sum(seq.get_num_blocks() for seq in sequences.values())
    utilization = (total_blocks_used / manager.num_blocks) * 100
    print(f"[SIM] Memory utilization: {utilization:.1f}% ({total_blocks_used}/{manager.num_blocks} blocks)")
    print()
    
    # Simulate some sequences finishing
    print("[SIM] req_1 and req_3 finish:")
    sequences["req_1"].free_all_blocks(manager)
    sequences["req_3"].free_all_blocks(manager)
    del sequences["req_1"]
    del sequences["req_3"]
    print_status(manager, sequences)
    
    # Allocate new sequences
    print("[SIM] New requests arrive:")
    new_allocations = [
        ("req_6", 192),  # 12 blocks
        ("req_7", 208),  # 13 blocks
    ]
    
    for seq_id, num_tokens in new_allocations:
        seq = Sequence(seq_id)
        try:
            seq.append_tokens(manager, num_tokens)
            sequences[seq_id] = seq
            num_blocks = num_tokens // manager.block_size
            print(f"  ✓ {seq_id}: {num_tokens} tokens ({num_blocks} blocks) - SUCCESS")
        except RuntimeError:
            print(f"  ✗ {seq_id}: {num_tokens} tokens - FAILED (OOM)")
    
    print()
    print_status(manager, sequences)
    
    total_blocks_used = sum(seq.get_num_blocks() for seq in sequences.values())
    utilization = (total_blocks_used / manager.num_blocks) * 100
    print(f"[SIM] Final memory utilization: {utilization:.1f}% ({total_blocks_used}/{manager.num_blocks} blocks)")
    print()


def main():
    print(" " * 25 + "Simulation Demo")

    # Run test cases
    test_case_1_simple_run()
    test_case_2_fragmentation_demo()
    test_case_3_high_utilization()
    


if __name__ == "__main__":
    main()
