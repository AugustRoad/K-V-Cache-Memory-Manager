"""
Sequence Module

This module implements the logical memory management for individual requests.
"""

from typing import Dict, Optional
from block_manager import BlockManager


class Sequence:
    """
    Represents a single inference request with its own logical memory view.
    """
    
    def __init__(self, request_id: str):
        """Create a new sequence for a request."""
        self.request_id = request_id
        self.token_count = 0
        
        self.block_table: Dict[int, int] = {}
    
    def append_token(self, manager: BlockManager) -> None:
        """
        Process a new token for this sequence.
        
        This is the core operation that implements dynamic memory allocation.
        
        Args:
            manager: The BlockManager to request physical blocks from
            
        Raises:
            RuntimeError: If the manager cannot allocate a block (OOM)
        """
        # Calculate which logical block this token belongs to
        logical_block_idx = self.token_count // manager.block_size
        
        # Check if we need to allocate a new physical block
        if logical_block_idx not in self.block_table:
            try:
                physical_block_idx = manager.allocate_block()
                self.block_table[logical_block_idx] = physical_block_idx
            except RuntimeError as e:
                raise RuntimeError(
                    f"Failed to allocate block for sequence {self.request_id} "
                    f"at token {self.token_count}: {e}"
                )
        
        self.token_count += 1
    
    def append_tokens(self, manager: BlockManager, num_tokens: int) -> None:
        """
        Convenience method to append multiple tokens at once.
        
        Args:
            manager: The BlockManager to request physical blocks from
            num_tokens: Number of tokens to append
            
        Raises:
            RuntimeError: If the manager runs out of blocks during allocation
        """
        for _ in range(num_tokens):
            self.append_token(manager)
    
    def get_physical_block(self, token_index: int, manager: BlockManager) -> tuple[int, int]:
        """
        Get the physical block and offset for a given token index.
        
        Args:
            token_index: The token's position in this sequence (0-indexed)
            manager: BlockManager (to get block_size)
            
        Returns:
            Tuple of (physical_block_index, offset_within_block)
            
        """
        if token_index < 0 or token_index >= self.token_count:
            raise ValueError(
                f"Invalid token index: {token_index}. "
                f"Must be between 0 and {self.token_count - 1}"
            )
        
        logical_block_idx = token_index // manager.block_size
        offset = token_index % manager.block_size
        
        if logical_block_idx not in self.block_table:
            raise ValueError(
                f"Logical block {logical_block_idx} not found in block table. "
                f"Token {token_index} has not been allocated yet."
            )
        
        physical_block_idx = self.block_table[logical_block_idx]
        return (physical_block_idx, offset)
    
    def get_num_blocks(self) -> int:
        """
        Get the number of physical blocks currently allocated to this sequence.
        
        Returns:
            Number of blocks in the block table
        """
        return len(self.block_table)
    
    def free_all_blocks(self, manager: BlockManager) -> None:
        """
        Release all physical blocks back to the manager.
        
        Args:
            manager: The BlockManager to return blocks to
        """
        # Return all physical blocks to the manager
        for physical_block_idx in self.block_table.values():
            manager.free_block(physical_block_idx)
        
        self.block_table.clear()
        self.token_count = 0
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"Sequence(id={self.request_id}, "
            f"tokens={self.token_count}, "
            f"blocks={len(self.block_table)}, "
            f"block_table={self.block_table})"
        )
