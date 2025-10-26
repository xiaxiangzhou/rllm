# 2048 Agent Examples

This directory contains examples for training and running 2048 game agents using the rLLM framework. The 2048 agent learns to play the classic 2048 puzzle game by combining tiles to reach higher values.

Our examples use the following:
* Qwen/Qwen2.5-3B-Instruct as the base model
* gym-2048 package for game logic with custom wrapper
* GRPO (Group Relative Policy Optimization) for training
* Reward structure matching the ART framework

## Game Overview

2048 is a puzzle game where:
- **Objective**: Combine tiles on a 4x4 grid to reach the target value (default: 128)
- **Moves**: Four possible actions - left, right, up, down
- **Mechanics**: 
  - When two tiles with the same number touch after a move, they merge into one
  - After each move, a new tile (2 or 4) appears in a random empty spot (only if the board changed)
  - The game ends when the target value is reached or no valid moves remain
- **Reward Structure** (matching ART):
  - Win reward: **2.0** (when reaching target value)
  - Progress reward: Combination of max tile and total board value using formula:
    - `max_value_reward = (log₂(max_tile) - 1) / (log₂(target) - 1)`
    - `board_value_reward = (log₂(total_value) - 1) / (log₂(target * 16) - 1)`
    - `final_reward = max_value_reward + (board_value_reward * 0.2)`
  - Invalid move penalty: **-1** (moves that don't change the board or invalid actions)

## Files

- `prepare_2048_data.py`: Prepares training and test datasets with random seeds
- `train_2048_agent.py`: Main training script using AgentTrainer
- `train_2048_agent.sh`: Bash script with training configuration (includes env_args)
- `run_2048_agent.py`: Inference script using AgentExecutionEngine (GPU-optimized)
- `run_2048_agent_cpu.py`: CPU-optimized inference script with smaller model
- `test_gym_2048_integration.py`: Test script to verify gym-2048 integration works correctly

## Installation

This example requires the gym-2048 package. Install it with:

```bash
pip install gym-2048
```

Or install rLLM with all dependencies:

```bash
pip install -e .
```

To verify the installation:

```bash
cd examples/2048
python test_gym_2048_integration.py
```

## Model Hosting

### Option 1: Using vLLM

Start a vLLM server with OpenAI-compatible API:

```bash
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-3B-Instruct \
    --host 0.0.0.0 \
    --port 30000 \
    --dtype bfloat16
```

### Option 2: Using SGLang

```bash
python -m sglang_router.launch_server \
    --model-path Qwen/Qwen2.5-3B-Instruct \
    --dp-size 1 \
    --dtype bfloat16
# increase dp_size to enable data-parallel processing on multiple GPUs
```

The server should be accessible at `http://localhost:30000/v1`

## Dataset Preparation

Prepare the 2048 datasets (randomly generated game configurations):

```bash
cd examples/2048
python prepare_2048_data.py
```

This will:
- Generate 720 random game configurations for training (matching ART: 40 steps × 18 games)
- Generate 200 random game configurations for testing
- Register both datasets with the rLLM DatasetRegistry
- Each game has a unique random seed for reproducibility
- Default target value: 128 (easier for training)

### Customizing Target Value

You can customize the difficulty by modifying the target value:

```python
from prepare_2048_data import prepare_2048_data

# Easy difficulty (default, matches ART)
train_ds, test_ds = prepare_2048_data(train_size=720, target_value=128)

# Medium difficulty
train_ds, test_ds = prepare_2048_data(train_size=720, target_value=512)

# Standard difficulty
train_ds, test_ds = prepare_2048_data(train_size=720, target_value=2048)

# More training data (if you want)
train_ds, test_ds = prepare_2048_data(train_size=5000, target_value=128)
```

## Running Inference

Once your model server is running and datasets are prepared, you can run inference:

```bash
cd examples/2048
python run_2048_agent.py
```

### Configuration Options

You can modify the inference script parameters:

- `n_parallel_agents`: Number of parallel agents (default: 128)
- `model_name`: Model to use (default: "Qwen/Qwen2.5-3B-Instruct")
- `base_url`: API server URL (default: "http://localhost:30000/v1")
- `max_response_length`: Maximum response length (default: 2048)
- `max_prompt_length`: Maximum prompt length (default: 4096)
- `temperature`: Sampling temperature (default: 0.7)
- `top_p`: Top-p sampling (default: 0.95)

The script will:
1. Load the 2048 test dataset (or generate if not exists)
2. Run parallel inference using the async agent execution engine
3. Evaluate results and compute success rates

## Training

### Basic Training

To train a 2048 agent:

```bash
bash examples/2048/train_2048_agent.sh
```

This uses the GRPO (Group Relative Policy Optimization) algorithm with the following default configuration:
- Model: Qwen/Qwen2.5-3B-Instruct
- Training epochs: 3
- Batch size: 2 per device
- Gradient accumulation: 8 steps
- Learning rate: 1e-5
- Mixed precision: bfloat16

### Custom Training

You can customize training by modifying the shell script or passing arguments:

```bash
python examples/2048/train_2048_agent.py \
    --config-name ppo_trainer \
    trainer=grpo \
    trainer.model_name_or_path=Qwen/Qwen2.5-3B-Instruct \
    trainer.output_dir=outputs/my_2048_agent \
    trainer.num_train_epochs=5 \
    trainer.learning_rate=2e-5
```

## Game Customization

### Dataset Parameters (Per-Game Configuration)

These vary for each game and are set during dataset preparation:

- **target_value**: The tile value needed to win
  - 128: Easier (default for training)
  - 256, 512, 1024: Medium difficulty
  - 2048: Standard game difficulty
  
  Set in `prepare_2048_data.py`:
  ```python
  prepare_2048_data(train_size=5000, test_size=200, target_value=128)
  ```

### Runtime Parameters (Global Configuration)

These apply to all games and are set via `env_args`:

- **max_steps**: Maximum number of moves per game (default: 100)

  Set in training script:
  ```bash
  +env.env_args.max_steps=100
  ```
  
  Or in inference script:
  ```python
  env_args = {"max_steps": 100}
  ```

**Note**: Board size is always 4x4 (standard 2048 game)

## Architecture

### File Structure

**Example Scripts** (`rllm/examples/2048/`):
- `prepare_2048_data.py` - Generate train/test datasets with random seeds
- `train_2048_agent.py` - Hydra-based training with AgentTrainer
- `train_2048_agent.sh` - Shell script with GRPO configuration
- `run_2048_agent.py` - Parallel inference with AgentExecutionEngine
- `test_gym_2048_integration.py` - Test script for gym-2048 integration

**Core Components** (`rllm/rllm/`):
- `agents/game_2048_agent.py` - Agent with chat interface and action parsing
- `environments/game_2048/game_2048.py` - Wrapper around gym-2048 with custom reward structure

### Agent Structure

The `Game2048Agent` (`rllm/rllm/agents/game_2048_agent.py`) implements:
- **System Prompt**: Concise instructions for playing 2048
  ```
  "You are an excellent 2048 player. Always choose the move most likely 
   to combine tiles and eventually reach the target value. Valid moves 
   are 'left', 'right', 'up', 'down'. Return your move as an XML object 
   with a single property 'move', like so: <move>left</move>"
  ```
- **Action Parsing**: Extracts moves from XML tags (`<move>left</move>`)
- **Trajectory Tracking**: Records all steps (observations, actions, rewards) for RL training
- **Conversation Management**: Maintains chat history with system prompt + alternating user/assistant messages
- **State Updates**:
  - `update_from_env()`: Receives observation from environment, adds to conversation
  - `update_from_model()`: Parses LLM response, extracts action, records in trajectory

### Environment Structure

The `Game2048Env` (`rllm/rllm/environments/game_2048/game_2048.py`) provides:
- **Game Logic**: Uses the gym-2048 package for game mechanics
- **Wrapper Features**:
  - Converts string actions ("left", "right", "up", "down") to gym-2048 integer actions (0, 1, 2, 3)
  - Formats numpy board arrays as human-readable strings for LLM agents
  - Implements custom reward calculation (overriding gym-2048's default)
- **Gym Interface**:
  - `reset()`: Creates new game with seeded RNG, returns initial board observation
  - `step(action)`: Applies move, adds random tile (if board changed), calculates reward
  - `from_dict(env_info)`: Creates environment from dataset entry + runtime config
- **Reward Calculation**: Matches ART formula (win: 2, invalid: -1, progress: logarithmic)
- **Termination Conditions**: 
  - Won (reached target value)
  - No valid moves remaining
  - Max steps reached

### How Dataset and Runtime Config Interact

The training system uses a two-level configuration:

**1. Dataset Entries** (per-game, varies):
```python
# From prepare_2048_data.py
{"seed": 42, "target_value": 128, "index": 0, "uid": "2048_game_42_0"}
```

**2. Runtime Config** (global, same for all games):
```python
# From train script or run script
env_args = {"max_steps": 100}
agent_args = {"max_steps": 100, "use_accumulate_history": True}
```

**3. Merging** (happens during training):
```python
# In trainer: agent_ppo_trainer.py:106
env_info = {**dataset_entry, **env_args}
# Result: {"seed": 42, "target_value": 128, "max_steps": 100, ...}

env = Game2048Env.from_dict(env_info)
```

**Why this design?**
- ✅ **Flexible**: Change `max_steps` without regenerating dataset
- ✅ **Efficient**: Dataset only stores what varies per game
- ✅ **Clear**: Separation between per-game (dataset) vs. global (runtime) config

### How Seeds Work

Seeds control all randomness in the game:

1. **Dataset Preparation**: Generates unique seeds for each game
   ```python
   random.seed(42)  # Reproducible seed generation
   train_seeds = [random.randint(0, 100000) for _ in range(720)]
   ```

2. **Environment Creation**: Each seed creates a deterministic game
   ```python
   env = Game2048Env(seed=42, target_value=128, max_steps=100)
   ```

3. **Game Execution**: Seed controls tile placement
   - Where initial tiles appear (via `random.Random(seed)`)
   - What value they have (2 or 4, with 90%/10% probability)
   - Where new tiles appear after each move

**Same seed = Same initial game state = Reproducible results**

This enables:
- 🔬 Reproducible experiments for scientific rigor
- 🐛 Debugging specific game scenarios
- ⚖️ Fair model comparisons on identical games
- 🎯 Guaranteed train/test split separation

### Reward Structure

The model learns through reinforcement learning with rewards (matching ART):

1. **Win**: `2.0` (when reaching target value)
2. **Progress** (when board changes):
   - Calculated using logarithmic scaling:
     ```python
     max_value_reward = (log₂(max_tile) - 1) / (log₂(target) - 1)
     board_value_reward = (log₂(total_value) - 1) / (log₂(target * 16) - 1)
     reward = max_value_reward + (board_value_reward * 0.2)
     ```
   - This rewards both increasing the max tile and building up board value
   - The formula ensures reward ranges from 0 (minimal progress) to 1 (near target)

3. **Invalid Move**: `-1` (move that doesn't change board or invalid action)

This reward structure is identical to the ART framework for fair comparison.

### Agent-Environment Interaction Loop

Here's how a single episode works:

**1. Initialization**:
```python
env = Game2048Env.from_dict({"seed": 42, "target_value": 128, "max_steps": 100})
agent = Game2048Agent(max_steps=100, use_accumulate_history=True)
```

**2. Episode Start**:
```python
# Environment creates game board with seed
observation, info = env.reset()
# observation = "Current board:\n0 0 0 0\n2 0 0 0\n0 0 2 0\n0 0 0 0\n..."

# Agent initializes with system prompt
agent.reset()
# agent.messages = [{"role": "system", "content": "You are an excellent 2048 player..."}]
```

**3. Interaction Loop** (repeat until done):
```python
# Agent receives observation
agent.update_from_env(observation, reward, done, info)
# Adds user message with board state to conversation

# LLM generates response
llm_response = call_llm(agent.chat_completions)
# llm_response = "I'll move left. <move>left</move>"

# Agent parses action
action = agent.update_from_model(llm_response)
# action = "left", adds assistant message to conversation

# Environment executes action
observation, reward, done, info = env.step(action)
# Applies move, adds random tile, calculates reward, checks termination
```

**4. Trajectory Collection**:
- Each step is recorded: (observation, action, reward, done)
- After episode ends, trajectory is used for PPO/GRPO training
- Model parameters updated to increase expected rewards

**Key Points**:
- Agent maintains conversation history (system → user → assistant → user → ...)
- Environment is stateless between episodes (reset creates fresh game)
- Same seed always produces same initial board, but different trajectories as agent learns
- Parallel execution: Multiple env/agent pairs run simultaneously for efficiency

