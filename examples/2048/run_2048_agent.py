import asyncio
import os

from transformers import AutoTokenizer

from rllm.agents.game_2048_agent import Game2048Agent
from rllm.data.dataset import DatasetRegistry
from rllm.engine.agent_execution_engine import AgentExecutionEngine
from rllm.environments.game_2048.game_2048 import Game2048Env
from rllm.utils import compute_pass_at_k


def load_2048_data():
    """Load 2048 test dataset or prepare it if not exists."""
    if DatasetRegistry.dataset_exists("2048", "test"):
        test_dataset = DatasetRegistry.load_dataset("2048", "test")
        return test_dataset.get_data()

    print("2048 datasets not found. Preparing datasets...")
    from prepare_2048_data import prepare_2048_data

    train_dataset, test_dataset = prepare_2048_data()

    return test_dataset.get_data()


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    # Configuration
    n_parallel_agents = 128
    model_name = "Qwen/Qwen2.5-3B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    sampling_params = {
        "temperature": 0.7,
        "top_p": 0.95,
        "model": model_name,
    }

    agent_args = {
        "max_steps": 100,
        "use_accumulate_history": True,
    }

    env_args = {
        "max_steps": 100,
    }

    engine = AgentExecutionEngine(
        agent_class=Game2048Agent,
        env_class=Game2048Env,
        agent_args=agent_args,
        env_args=env_args,
        engine_name="openai",
        tokenizer=tokenizer,
        sampling_params=sampling_params,
        rollout_engine_args={
            "base_url": "http://localhost:30000/v1",
            "api_key": "None",
        },
        max_response_length=2048,
        max_prompt_length=4096,
        n_parallel_agents=n_parallel_agents,
    )

    tasks = load_2048_data()

    results = asyncio.run(engine.execute_tasks(tasks))
    compute_pass_at_k(results)

