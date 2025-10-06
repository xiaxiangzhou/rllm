import random

from rllm.data.dataset import DatasetRegistry


def prepare_2048_data(train_size=720, test_size=200, target_value=128):
    """
    Prepare and register 2048 game datasets for training and testing.

    Args:
        train_size (int): Number of training examples to generate (default: 720)
        test_size (int): Number of test examples to generate (default: 200)
        target_value (int): Target tile value to win (default: 128)
                           Common values: 128 (easy), 256, 512, 1024, 2048 (standard)

    Returns:
        tuple: (train_dataset, test_dataset)
    """
    # Set random seed for reproducibility
    random.seed(42)

    def game_2048_process_fn(seed, idx):
        """Process function to create 2048 game task instances."""
        return {
            "seed": seed,
            "index": idx,
            "uid": f"2048_game_{seed}_{idx}",
            "target_value": target_value,
        }

    # Generate random seeds for train and test sets
    train_seeds = [random.randint(0, 100000) for _ in range(train_size)]
    test_seeds = [random.randint(0, 100000) for _ in range(test_size)]

    # Create train and test data
    train_data = [
        game_2048_process_fn(seed, idx) for idx, seed in enumerate(train_seeds)
    ]
    test_data = [game_2048_process_fn(seed, idx) for idx, seed in enumerate(test_seeds)]

    # Register the datasets with the DatasetRegistry
    train_dataset = DatasetRegistry.register_dataset("2048", train_data, "train")
    test_dataset = DatasetRegistry.register_dataset("2048", test_data, "test")

    return train_dataset, test_dataset


if __name__ == "__main__":
    # You can customize the target value here
    # target_value: 128 (easy), 256, 512, 1024, or 2048 (standard)
    train_dataset, test_dataset = prepare_2048_data(target_value=128)
    print(f"Train dataset: {len(train_dataset.get_data())} examples")
    print(f"Test dataset: {len(test_dataset.get_data())} examples")
    print("Sample train example:", train_dataset.get_data()[0])
    print("Sample test example:", test_dataset.get_data()[0])

