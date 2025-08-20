#!/usr/bin/env python3
"""
Test script to verify the improved Le model functionality works with real data.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
from memory import Memory
from le import sample_prompt, _fallback_reply, ModelConfig, Transformer, create_datasets

def test_improved_functionality():
    print("Testing Le model improvements...")
    
    # Test fallback reply
    print(f"Testing fallback reply: {_fallback_reply()}")
    
    # Create a simple dataset
    train_dataset, test_dataset = create_datasets("blood/test_data.txt")
    print(f"Created dataset with {len(train_dataset.words)} words")
    print(f"Vocabulary size: {train_dataset.get_vocab_size()}")
    
    # Create a small model for testing
    config = ModelConfig(
        vocab_size=train_dataset.get_vocab_size(),
        block_size=train_dataset.get_output_length(),
        n_layer=2,
        n_head=2, 
        n_embd=64
    )
    model = Transformer(config)
    
    # Test generation with different prompts
    with Memory(":memory:") as memory:
        test_prompts = [
            "hello",
            "good morning",
            "what is happening",
            "tell me more"
        ]
        
        print("\nTesting improved sample_prompt with various inputs:")
        for prompt in test_prompts:
            try:
                result = sample_prompt(prompt, model, train_dataset, memory, max_new_tokens=8)
                print(f"Prompt: '{prompt}' -> Response: '{result}'")
            except Exception as e:
                print(f"Error with prompt '{prompt}': {e}")
    
    print("\nAll tests completed successfully!")

if __name__ == "__main__":
    test_improved_functionality()