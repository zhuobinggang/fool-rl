import gym


env = gym.make("LunarLander-v2")
env.action_space.seed(42)

observation, info = env.reset(seed=42, return_info=True)

done = False

while not done:
    observation, reward, done, info = env.step(env.action_space.sample())
    env.render()
    if done:
        observation, info = env.reset(return_info=True)

env.close()
