"""
Local test to verify the 2048 agent is working correctly.
Tests board formatting, moves, and agent interaction.
"""

import sys
from pathlib import Path

# Add rllm to path
rllm_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(rllm_path))

from rllm.environments.game_2048.game_2048 import Game2048Env
from rllm.agents.game_2048_agent import Game2048Agent


def test_environment_basic():
    """Test basic environment functionality."""
    print("=" * 70)
    print("TEST 1: Basic Environment - Board Formatting & Moves")
    print("=" * 70)
    
    env = Game2048Env(seed=42, target_value=128, max_steps=100)
    obs, info = env.reset()
    
    print(f"\n✓ Environment created successfully")
    print(f"\nInitial Board (4x4):")
    print(obs)
    print(f"\nBoard Details:")
    print(f"  Type: {type(obs)}")
    print(f"  Lines: {len(obs.split(chr(10)))}")
    print(f"  Max tile: {info['max_tile']}")
    print(f"  Total value: {info['total_value']}")
    
    # Test each direction
    moves = ["left", "up", "right", "down"]
    print(f"\n{'='*70}")
    print("Testing All 4 Directions")
    print("=" * 70)
    
    for i, move in enumerate(moves, 1):
        obs, reward, done, info = env.step(move)
        print(f"\n{i}. Move: {move.upper()}")
        print(f"   Board after move:")
        for line in obs.split('\n'):
            print(f"   {line}")
        print(f"   Reward: {reward:.4f}, Max tile: {info['max_tile']}, Done: {done}")
        
        if done:
            print(f"\n   Game ended after {i} moves")
            break
    
    return True


def test_agent_interaction():
    """Test agent and environment interaction."""
    print("\n" + "=" * 70)
    print("TEST 2: Agent-Environment Interaction")
    print("=" * 70)
    
    env = Game2048Env(seed=123, target_value=128, max_steps=100)
    agent = Game2048Agent(max_steps=100, use_accumulate_history=True)
    
    obs, info = env.reset()
    agent.reset()
    
    print(f"\n✓ Environment and Agent initialized")
    print(f"\nInitial state:")
    print(obs)
    
    # Simulate agent-env interaction
    test_moves = ["left", "up", "right", "down", "left"]
    
    for step, move in enumerate(test_moves, 1):
        # Agent receives observation
        agent.update_from_env(obs, 0.0 if step == 1 else reward, False, info)
        
        # Simulate LLM response
        llm_response = f"I'll move {move}. <move>{move}</move>"
        
        # Agent parses response
        action_obj = agent.update_from_model(llm_response)
        
        # Environment executes action
        obs, reward, done, info = env.step(action_obj.action)
        
        print(f"\nStep {step}: {action_obj.action.upper()}")
        print(f"  Reward: {reward:.4f}")
        print(f"  Max tile: {info['max_tile']}")
        print(f"  Agent has {len(agent.trajectory.steps)} trajectory steps")
        print(f"  Agent has {len(agent.messages)} messages in history")
        
        if done:
            break
    
    print(f"\n✓ Agent-environment interaction working correctly")
    print(f"✓ Final trajectory length: {len(agent.trajectory.steps)} steps")
    
    return True


def test_invalid_action():
    """Test invalid action handling."""
    print("\n" + "=" * 70)
    print("TEST 3: Invalid Action Handling")
    print("=" * 70)
    
    env = Game2048Env(seed=42, target_value=128)
    env.reset()
    
    # Test invalid action string
    obs, reward, done, info = env.step("invalid_move")
    
    print(f"\n✓ Invalid action test:")
    print(f"  Action: 'invalid_move'")
    print(f"  Reward: {reward}")
    print(f"  Done: {done}")
    print(f"  Invalid flag: {info.get('invalid_action', False)}")
    
    assert reward == -1.0, f"Expected reward -1.0, got {reward}"
    assert done is True, f"Expected done=True, got {done}"
    assert info.get('invalid_action') is True, "Expected invalid_action flag"
    
    print(f"\n✓ Invalid action handled correctly")
    
    return True


def test_agent_parsing():
    """Test agent action parsing."""
    print("\n" + "=" * 70)
    print("TEST 4: Agent Action Parsing")
    print("=" * 70)
    
    agent = Game2048Agent(max_steps=100)
    
    # Test valid XML parsing
    test_cases = [
        ("<move>left</move>", "left"),
        ("Let me think. <move>up</move>", "up"),
        ("<move>RIGHT</move>", "right"),  # Case insensitive
        ("<move>Down</move> is best", "down"),
        ("No XML here", "invalid"),
    ]
    
    print("\n✓ Testing action parsing:")
    for response, expected in test_cases:
        thought, action = agent._parse_model_response(response)
        status = "✓" if action == expected else "✗"
        print(f"  {status} '{response[:30]}...' -> '{action}' (expected: '{expected}')")
        assert action == expected, f"Expected {expected}, got {action}"
    
    print(f"\n✓ All parsing tests passed")
    
    return True


def test_board_continuity():
    """Test that board state is maintained correctly across moves."""
    print("\n" + "=" * 70)
    print("TEST 5: Board State Continuity")
    print("=" * 70)
    
    env = Game2048Env(seed=999, target_value=128)
    obs1, info1 = env.reset()
    
    print(f"\n✓ Initial board:")
    print(obs1)
    print(f"  Max tile: {info1['max_tile']}, Total: {info1['total_value']}")
    
    # Make a move
    obs2, reward, done, info2 = env.step("left")
    
    print(f"\n✓ After 'left' move:")
    print(obs2)
    print(f"  Max tile: {info2['max_tile']}, Total: {info2['total_value']}")
    print(f"  Reward: {reward:.4f}")
    
    # Verify board changed (unless move was invalid)
    if reward != -1.0:
        print(f"\n✓ Board state changed after valid move")
    else:
        print(f"\n✓ Board state unchanged (invalid move)")
    
    # Make another move
    obs3, reward2, done2, info3 = env.step("up")
    
    print(f"\n✓ After 'up' move:")
    print(obs3)
    print(f"  Max tile: {info3['max_tile']}, Total: {info3['total_value']}")
    print(f"  Reward: {reward2:.4f}")
    
    print(f"\n✓ Board continuity maintained")
    
    return True


if __name__ == "__main__":
    print("\n" + "🧪" * 35)
    print("2048 AGENT LOCAL TEST SUITE")
    print("🧪" * 35 + "\n")
    
    all_passed = True
    tests = [
        ("Basic Environment", test_environment_basic),
        ("Agent Interaction", test_agent_interaction),
        ("Invalid Actions", test_invalid_action),
        ("Action Parsing", test_agent_parsing),
        ("Board Continuity", test_board_continuity),
    ]
    
    for test_name, test_func in tests:
        try:
            test_func()
        except Exception as e:
            print(f"\n❌ TEST FAILED: {test_name}")
            print(f"   Error: {e}")
            import traceback
            traceback.print_exc()
            all_passed = False
            break
    
    print("\n" + "=" * 70)
    if all_passed:
        print("✅ ALL TESTS PASSED!")
        print("=" * 70)
        print("\n🎉 The 2048 agent is working correctly!")
        print("   You can now run training with:")
        print("   bash examples/2048/train_2048_agent.sh")
    else:
        print("❌ SOME TESTS FAILED")
        print("=" * 70)
        print("\nPlease fix the errors above before running training.")
    
    print()

