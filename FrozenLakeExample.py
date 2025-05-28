import gymnasium as gym
import argparse

class main():
    def __init__(self):
        # create environment
        self.env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True, render_mode="human")
        episode = 0
        for i in range(1000):
            # reset environment
            state, info = self.env.reset()
            done = False
            step = 0
            while not done:
                # random generate action
                action = self.env.action_space.sample()
                
                # interact with environment
                next_state , reward , terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                print("------")
                print("Episode : ", episode)
                print("Step :",step)
                print("State :",state)
                print("Next state :",next_state)
                print("Action :",action)
                print("Reward :",reward)
                print("Done :",done)

                state = next_state
                step += 1
            episode += 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameters Setting for Q-learning")
    args = parser.parse_args()
    main()
