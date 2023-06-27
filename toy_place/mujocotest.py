import gym



env=gym.make("Reacher-v4",render_mode="human")
env.reset()



while True:
    _,_,_,_,_=env.step(env.action_space.sample())
    print(_)

    
