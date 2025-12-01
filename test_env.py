from dm_control import suite
import numpy as np

def main():
    # Load the DeepMind Control Suite Cartpole Balance environment
    env = suite.load(domain_name="cartpole", task_name="balance")

    # Reset environment
    time_step = env.reset()

    print("\nEnvironment loaded successfully!")
    print("Observation keys:", time_step.observation.keys())

    action_spec = env.action_spec()
    print("Action spec:", action_spec)

    # Take 10 random steps
    for i in range(10):
        # Sample random action from allowed range
        action = np.random.uniform(action_spec.minimum,
                                   action_spec.maximum,
                                   size=action_spec.shape)

        time_step = env.step(action)

        print(f"Step {i+1}: reward = {time_step.reward}, done = {time_step.last()}")

if __name__ == "__main__":
    main()
