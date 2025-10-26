import hydra

from rllm.data import DatasetRegistry
from rllm.trainer.agent_trainer import AgentTrainer


@hydra.main(
    config_path="pkg://rllm.trainer.config", config_name="agent_ppo_trainer", version_base=None
)
def main(config):
    # Import here to avoid circular dependencies
    from rllm.agents.game_2048_agent import Game2048Agent
    from rllm.environments.game_2048.game_2048 import Game2048Env

    train_dataset = DatasetRegistry.load_dataset("2048", "train")
    val_dataset = DatasetRegistry.load_dataset("2048", "test")

    trainer = AgentTrainer(
        agent_class=Game2048Agent,
        env_class=Game2048Env,
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )
    trainer.train()


if __name__ == "__main__":
    main()

