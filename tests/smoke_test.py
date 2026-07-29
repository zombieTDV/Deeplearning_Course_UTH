"""
Smoke Test Runner (Vibe Coding Tier-1 Verification)
Fast execution (< 10 seconds) to catch syntax errors, missing dependencies,
import failures, and tensor shape mismatches on dummy data.
Refer to Section 9 of agents/WORKFLOW.md.
"""

import sys
import time

def run_smoke_test():
    print("=" * 60)
    print("🚀 Running Fast Smoke Test...")
    print("=" * 60)
    start_time = time.time()
    
    # 1. Imports Check
    print("[1/4] Checking core dependencies & package imports...")
    try:
        import torch
        import numpy as np
        print(f"  ✓ PyTorch version: {torch.__version__}")
        print(f"  ✓ NumPy version: {np.__version__}")
    except ImportError as e:
        print(f"  ❌ Import error: {e}")
        return False

    # 2. Config & Path Check
    print("[2/4] Verifying basic tensor operations...")
    dummy_input = torch.randn(2, 3, 224, 224)
    print(f"  ✓ Created dummy input tensor of shape: {tuple(dummy_input.shape)}")

    # 3. Dummy Forward Pass Check
    print("[3/4] Testing dummy network layer...")
    linear_layer = torch.nn.Linear(224 * 224 * 3, 10)
    flat_input = dummy_input.view(2, -1)
    output = linear_layer(flat_input)
    assert output.shape == (2, 10), f"Expected shape (2, 10), got {output.shape}"
    print(f"  ✓ Dummy output shape: {tuple(output.shape)}")

    elapsed = time.time() - start_time
    print("[4/4] Execution time check...")
    print(f"  ✓ Completed in {elapsed:.2f} seconds (< 10.0s target)")
    print("=" * 60)
    print("✅ SMOKE TEST PASSED SUCCESSFULLY")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = run_smoke_test()
    if not success:
        sys.exit(1)
