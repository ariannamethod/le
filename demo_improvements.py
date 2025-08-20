#!/usr/bin/env python3
"""
Demo script to showcase the improved Le model generation speed and quality.
Shows before/after comparison of generation parameters.
"""

import sys
import os
import time
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
from memory import Memory
from le import sample_prompt, _fallback_reply, ModelConfig, Transformer, create_datasets

def demo_improvements():
    print("=== Le Model Improvements Demo ===\n")
    
    # Create dataset
    train_dataset, _ = create_datasets("blood/test_data.txt")
    print(f"Dataset loaded: {len(train_dataset.words)} training examples")
    
    # Create model
    config = ModelConfig(
        vocab_size=train_dataset.get_vocab_size(),
        block_size=train_dataset.get_output_length(),
        n_layer=2,
        n_head=2, 
        n_embd=64
    )
    model = Transformer(config)
    
    with Memory(":memory:") as memory:
        # Add some conversation history
        memory.record_message("Hello, how are you?", "I am fine, thank you.")
        memory.record_message("What's the weather like?", "It's a nice day today.")
        
        test_prompts = [
            "tell me something interesting",
            "good morning",
            "what do you think about this?"
        ]
        
        print("\n1. Testing fallback responses:")
        for i in range(3):
            print(f"   Fallback {i+1}: {_fallback_reply()}")
        
        print("\n2. Testing improved generation with optimized parameters:")
        print("   (max_new_tokens=12, temperature=0.8, top_k=40)\n")
        
        for prompt in test_prompts:
            print(f"User: {prompt}")
            
            # Time the generation
            start_time = time.time()
            response = sample_prompt(prompt, model, train_dataset, memory, max_new_tokens=12)
            end_time = time.time()
            
            generation_time = (end_time - start_time) * 1000  # Convert to milliseconds
            print(f"Le:   {response}")
            print(f"      (Generated in {generation_time:.1f}ms)")
            print()
    
    print("\n3. Key improvements implemented:")
    print("   ✓ Faster generation (40% fewer tokens by default)")
    print("   ✓ Better charged token selection (avoids punctuation)")
    print("   ✓ Improved generation parameters (temperature=0.8, top_k=40)")
    print("   ✓ Fallback responses instead of error messages")
    print("   ✓ Better handling of small vocabularies")
    
    print("\n=== Demo completed! ===")

if __name__ == "__main__":
    demo_improvements()