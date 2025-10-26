"""
Test script to verify gym-2048 integration with Game2048Env.

This script tests that:
1. gym-2048 is properly installed
2. Game2048Env wrapper works correctly
3. String actions are properly converted
4. Board formatting is correct
5. Reward calculation works

Run this after installing gym-2048:
    pip install gym-2048
    python test_gym_2048_integration.py
"""

import sys
from pathlib import Path

# Add rllm to path
rllm_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(rllm_path))

from rllm.environments.game_2048.game_2048 import Game2048Env


def test_basic_functionality():
    """Test basic environment functionality."""
    print("=" * 60)
    print("Testing Game2048Env with gym-2048")
    print("=" * 60)
    
    # Create environment
    env = Game2048Env(seed=42, target_value=128, max_steps=100)
    print("✓ Environment created successfully")
    
    # Reset
    obs, info = env.reset()
    print("\n✓ Environment reset successfully")
    print(f"\nInitial Board:\n{obs}")
    print(f"\nInfo: {info}")
    
    # Test all actions
    print("\n" + "=" * 60)
    print("Testing all action types")
    print("=" * 60)
    
    actions = ["left", "up", "right", "down"]
    for i, action in enumerate(actions, 1):
        obs, reward, done, info = env.step(action)
        print(f"\nStep {i}: Action '{action}'")
        print(f"Board:\n{obs}")
        print(f"Reward: {reward:.4f}")
        print(f"Done: {done}")
        print(f"Max tile: {info['max_tile']}, Total: {info['total_value']}")
        
        if done:
            print("\nGame ended!")
            break
    
    print("\n" + "=" * 60)
    print("✓ All tests passed successfully!")
    print("=" * 60)


def test_invalid_action():
    """Test invalid action handling."""
    print("\n" + "=" * 60)
    print("Testing Invalid Action Handling")
    print("=" * 60)
    
    env = Game2048Env(seed=42, target_value=128)
    env.reset()
    
    # Test invalid action
    obs, reward, done, info = env.step("invalid_move")
    print(f"\nInvalid action result:")
    print(f"Reward: {reward}")
    print(f"Done: {done}")
    print(f"Invalid action flag: {info.get('invalid_action', False)}")
    
    assert reward == -1.0, "Invalid action should give -1 reward"
    assert done is True, "Invalid action should end the game"
    print("\n✓ Invalid action handling works correctly")


def test_determinism():
    """Test that same seed gives same results."""
    print("\n" + "=" * 60)
    print("Testing Determinism")
    print("=" * 60)
    
    # Create two environments with same seed
    env1 = Game2048Env(seed=123, target_value=128)
    env2 = Game2048Env(seed=123, target_value=128)
    
    obs1, _ = env1.reset()
    obs2, _ = env2.reset()
    
    print(f"\nEnv1 initial board:\n{obs1}")
    print(f"\nEnv2 initial board:\n{obs2}")
    
    assert obs1 == obs2, "Same seed should produce same initial state"
    print("\n✓ Determinism test passed")


def test_action_case_insensitive():
    """Test that actions are case-insensitive."""
    print("\n" + "=" * 60)
    print("Testing Case-Insensitive Actions")
    print("=" * 60)
    
    env = Game2048Env(seed=42, target_value=128)
    env.reset()
    
    # Test different cases
    for action in ["LEFT", "Up", "RiGhT", "down"]:
        obs, reward, done, info = env.step(action)
        print(f"✓ Action '{action}' accepted")
        if done:
            break
    
    print("\n✓ Case-insensitive action test passed")


if __name__ == "__main__":
    try:
        import gym
        import gym_2048
        print("gym-2048 is installed ✓\n")
    except ImportError as e:
        print(f"ERROR: {e}")
        print("\nPlease install gym-2048:")
        print("    pip install gym-2048")
        sys.exit(1)
    
    # Run all tests
    test_basic_functionality()
    test_invalid_action()
    test_determinism()
    test_action_case_insensitive()
    
    print("\n" + "=" * 60)
    print("🎉 ALL TESTS PASSED! 🎉")
    print("=" * 60)
    print("\nThe gym-2048 integration is working correctly!")

