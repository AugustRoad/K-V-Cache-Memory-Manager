"""
Block manager - handles the actual physical memory pool on the GPU.
"""

from typing import Optional, List
import torch


class BlockManager:
    """
    Manages physical memory blocks on the GPU.
    """
    def __init__(self, num_blocks: int,block_size: int,num_layers: int,num_heads: int,head_size: int,device: str = "cuda"):
        """
        Create the block manager and allocate all memory upfront.
        
        Args:
            num_blocks: How many blocks to create
            block_size: Tokens per block
            num_layers: Transformer layers
            num_heads: Attention heads per layer
            head_size: Dimension of each head
            device: Where to allocate ("cuda" or "cpu")
        """

        self.num_blocks = num_blocks
        self.block_size = block_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_size = head_size
        self.device = device
        
        # Allocate all the memory upfront
        # shape: [num_blocks, num_layers, num_heads, block_size, head_size]
        self.key_cache = torch.zeros(
            (num_blocks, num_layers, num_heads, block_size, head_size),
            dtype=torch.float16,
            device=device
        )
        self.value_cache = torch.zeros(
            (num_blocks, num_layers, num_heads, block_size, head_size),
            dtype=torch.float16,
            device=device
        )
        
        # Set blocks start as free
        self.free_blocks: List[int] = list(range(num_blocks))
        
    def allocate_block(self) -> int:
        """
        allocate one free block.
        
        Returns the block's index, or raises an error if we're out of memory.
        """

        if not self.free_blocks:
            raise RuntimeError(
                f"Total blocks: {self.num_blocks}, All allocated."
            )
        
        return self.free_blocks.pop()
    
    def free_block(self, block_index: int) -> None:
        """Return a block to the free pool."""

        if block_index < 0 or block_index >= self.num_blocks:
            raise ValueError(
                f"Invalid block index: {block_index}. "
                f"Must be between 0 and {self.num_blocks - 1}"
            )
        
        if block_index in self.free_blocks:
            raise ValueError(
                f"Block {block_index} is already free. "
            )
        
        self.free_blocks.append(block_index)
    
    def get_num_free_blocks(self) -> int:
        """How many blocks are currently available?"""
        return len(self.free_blocks)
    
    def get_cache_block(self, block_index: int, cache_type: str = "key" ) -> torch.Tensor:
        """
        Get a reference to a specific block in the cache.
        
        Args:
            block_index: Physical block index to access
            cache_type: Either "key" or "value"
            
        Returns:
            Tensor view of shape [num_layers, num_heads, block_size, head_size]
            
        """
        if block_index < 0 or block_index >= self.num_blocks:
            raise ValueError(
                f"Invalid block index: {block_index}. "
                f"Must be between 0 and {self.num_blocks - 1}"
            )
        
        if cache_type == "key":
            return self.key_cache[block_index]
        elif cache_type == "value":
            return self.value_cache[block_index]
        else:
            raise ValueError(
                f"Invalid cache_type: {cache_type}. "
                "Must be 'key' or 'value'"
            )
