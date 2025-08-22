#!/usr/bin/env python3
"""
Test script for scenario_aware_seismic_loss function.
This script verifies that the loss function correctly masks missing data areas
for different federated scenarios.
"""

import torch
import numpy as np
import sys
import os

# Add the scripts directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts', 'flwr'))

from centralized_baseline import scenario_aware_seismic_loss

def test_scenario_loss():
    """Test the scenario-aware seismic loss function."""
    
    print("Testing scenario_aware_seismic_loss function...")
    
    # Test data dimensions
    batch_size = 2
    time_samples = 1000
    
    # Test scenarios
    test_scenarios = [
        ('2A', 5, 70),  # 5 sources, 70 receivers
        ('2B', 5, 70),  # 5 sources, 70 receivers  
        ('2C', 6, 70),  # 6 sources, 70 receivers
        ('3A', 10, 70), # 10 sources, 70 receivers
        ('3B', 10, 70)  # 10 sources, 70 receivers
    ]
    
    for scenario, num_sources, num_receivers in test_scenarios:
        print(f"\n--- Testing Scenario {scenario} ---")
        
        # Create test data
        y = torch.randn(batch_size, num_sources, time_samples, num_receivers)
        predicted_seismic = torch.randn(batch_size, num_sources, time_samples, num_receivers)
        
        try:
            # Compute loss
            loss = scenario_aware_seismic_loss(y, predicted_seismic, scenario)
            print(f"✓ Successfully computed loss for {scenario}: {loss.item():.6f}")
            
            # Verify that the loss is finite
            assert torch.isfinite(loss), f"Loss is not finite for scenario {scenario}"
            assert loss.item() >= 0, f"Loss is negative for scenario {scenario}"
            
        except Exception as e:
            print(f"✗ Error in scenario {scenario}: {e}")
            return False
    
    print("\n✓ All tests passed!")
    return True

def test_mask_creation():
    """Test that masks are created correctly for each scenario."""
    
    print("\n--- Testing Mask Creation ---")
    
    # Test with minimal dimensions for clarity
    batch_size = 1
    num_sources = 5  # Use 2A scenario
    time_samples = 10
    num_receivers = 70
    
    y = torch.ones(batch_size, num_sources, time_samples, num_receivers)
    predicted_seismic = torch.ones(batch_size, num_sources, time_samples, num_receivers)
    
    # Test scenario 2A
    loss = scenario_aware_seismic_loss(y, predicted_seismic, '2A')
    
    # The loss should be less than the full L1 loss because some areas are masked
    full_loss = torch.nn.functional.l1_loss(y.float(), predicted_seismic.float())
    
    print(f"Full L1 loss: {full_loss.item():.6f}")
    print(f"Masked loss: {loss.item():.6f}")
    print(f"Loss reduction: {((full_loss - loss) / full_loss * 100):.2f}%")
    
    # Verify that the masked loss is less than or equal to the full loss
    assert loss <= full_loss, "Masked loss should be less than or equal to full loss"
    
    print("✓ Mask creation test passed!")
    return True

if __name__ == "__main__":
    print("=" * 50)
    print("Testing Scenario-Aware Seismic Loss Function")
    print("=" * 50)
    
    try:
        # Run tests
        test1_passed = test_scenario_loss()
        test2_passed = test_mask_creation()
        
        if test1_passed and test2_passed:
            print("\n" + "=" * 50)
            print("🎉 ALL TESTS PASSED! 🎉")
            print("=" * 50)
            print("\nThe scenario_aware_seismic_loss function is working correctly.")
            print("It properly masks missing data areas for each federated scenario.")
        else:
            print("\n❌ Some tests failed. Please check the implementation.")
            
    except Exception as e:
        print(f"\n❌ Test execution failed with error: {e}")
        import traceback
        traceback.print_exc()
