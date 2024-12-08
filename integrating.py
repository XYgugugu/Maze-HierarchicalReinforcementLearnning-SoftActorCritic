# Update main.py with enhanced logging and debugging for testing
# main_testing_code = """
import numpy as np
import torch
from maze import Maze
from player import Player
from hierarchical_trainer import HierarchicalTrainer


maze = Maze('./data/mazes.txt')
player = Player(maze.start_pos)

# Dimensions
state_dim = player.state_dim  # Dimensionality of the state space
action_dim = player.action_dim  # Dimensionality of the action space
subgoal_dim = maze.subgoal_dim  # Dimensionality of subgoals

# Hyperparameters
buffer_capacity = 10000
high_lr = 0.001
low_lr = 0.001
batch_size = 64
episodes = 10  # Run for 10 episodes to test

# Initialize trainer
trainer = HierarchicalTrainer(state_dim, action_dim, subgoal_dim, buffer_capacity, high_lr, low_lr)

# Game loop for testing
for episode in range(episodes):
    state_local = player.reset()  # Reset the player for a new episode
    state_global = maze.get_global_state()  # Retrieve the global state from the maze
    done = False
    episode_reward = 0

    print(f"Starting Episode {episode + 1}/{episodes}")

    while not done:
        # High-level policy selects a subgoal
        subgoal = trainer.high_policy(torch.tensor(state_global, dtype=torch.float32)).detach().numpy()
        print(f"Selected Subgoal: {subgoal}")

        # Low-level policy executes actions to achieve the subgoal
        for step in range(maze.max_steps_per_subgoal):
            action = trainer.low_policy(torch.tensor(state_local, dtype=torch.float32)).detach().numpy()
            next_state_local, reward, done, info = player.step(action)
            episode_reward += reward

            # Log details of each step
            print(f"  Step {step + 1}: Action: {action}, Reward: {reward}, Done: {done}")

            # Add experience to low-level replay buffer
            trainer.low_level_step(state_local, action, reward, next_state_local, done)
            state_local = next_state_local

            if done:
                break

        # Add experience to high-level replay buffer
        next_state_global = maze.get_global_state()
        trainer.high_level_step(state_global, subgoal, reward, next_state_global, done)
        state_global = next_state_global

    # Train both policies after each episode
    trainer.train(1, batch_size)

    print(f"Episode {episode + 1}/{episodes} completed with total reward: {episode_reward}")

# Save the trained models
torch.save(trainer.high_policy.state_dict(), "high_policy_test.pth")
torch.save(trainer.low_policy.state_dict(), "low_policy_test.pth")
print("Models saved successfully.")
# """

# # Save the testing-enhanced main script
# main_testing_file_path = os.path.join(main_folder_path, "main_test.py")

# with open(main_testing_file_path, "w") as f:
#     f.write(main_testing_code)

# # Confirm successful creation of the test script
# os.listdir(main_folder_path)
