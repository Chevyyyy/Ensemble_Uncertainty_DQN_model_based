import gym
import d4rl 

# Create the environment
env = gym.make('maze2d-open-v0')
while True:
    done=False
    _=env.reset()
    while not done:
        o,r,done,_=env.step(env.action_space.sample())
        env.render()
        print(o,r,done)