"""
Simple demo of ContextPruner
"""

import sys
sys.path.append('..')

from src.pruner import ContextPruner


def main():
    pruner = ContextPruner()
    
    question = "What is the largest planet in the solar system?"
    context = """
    The solar system has eight planets. 
    Jupiter is the largest planet, with a mass 2.5 times that of all other planets combined. 
    Earth is the third planet from the Sun. 
    Saturn is known for its rings.
    """
    
    print("=" * 50)
    print("ContextPruner Demo")
    print("=" * 50)
    
    print(f"\nQuestion: {question}")
    print(f"\nOriginal context ({len(context.split())} words):")
    print(context.strip())
    
    for ratio in [1.0, 0.5, 0.3]:
        pruned = pruner.prune(question, context, keep_ratio=ratio)
        print(f"\n--- Keep ratio: {ratio:.0%} ---")
        print(f"Pruned context ({len(pruned.split())} words):")
        print(pruned)


if __name__ == "__main__":
    main()
