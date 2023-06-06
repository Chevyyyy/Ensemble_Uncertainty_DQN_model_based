#!/usr/bin/python3
import gym
import matplotlib.pyplot as plt
from itertools import count
import torch
from networks.deep_endemble_NN_model import GaussianMixtureMLP
from buffer import ReplayMemory
from utilis import * 
from logger import set_log_file
import logging
import argparse
from torch.distributions import Categorical
from agent.DQN_Agent import DQN 
from agent.DQN_ensemble_Agent import DQN_ensemble 
from agent.model_1_AI import model_1_AI 
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

parser = argparse.ArgumentParser(description='|model|env|')
parser.add_argument("--model",default="model_1_AI",help="DQN|PPO|ensemble_DQN|model_1_AI|model_1_AI_actor")
parser.add_argument("--env",default="CartPole-v1",help="CartPole-v1|MountainCar-v0|LunarLander-v2|Acrobot-v1|Pendulum-v1")
parser.add_argument("--BATCH_SIZE",type=int,default=300)
parser.add_argument("--NUM_episodes",type=int,default=3000000)
parser.add_argument("--GAMMA",default=0.99)
parser.add_argument("--TAU",default=0.005)
parser.add_argument("--PRINT",default=False)
parser.add_argument("--render_mode",default="rgb_array")
parser.add_argument("--device",default="cpu")
parser.add_argument("--NUM_ensemble",default=5)
parser.add_argument("--ID",default="")
parser.add_argument("--foot_record",default=False)
parser.add_argument("--max_steps",type=int,default=10e5)
args = parser.parse_args()

###############################################################################################
# config the args
# set the log file
set_log_file(f"log/{args.model}_{args.env}_{args.ID}.txt")
# set env
env = gym.make(args.env,render_mode=args.render_mode)
# Get number of actions from gym action space
n_actions = env.action_space.n
# Get the number of state observations
state, info = env.reset()
n_observations = len(state)
# set device
device = torch.device(args.device)

# set the model
if args.model=="DQN":
    agent = DQN(n_observations, n_actions,env)
elif args.model=="ensemble_DQN":
    agent = DQN_ensemble(args.NUM_ensemble,n_observations,n_actions)
elif args.model=="model_1_AI":
    agent = model_1_AI(args.NUM_ensemble,n_observations,n_actions)
elif args.model=="model_1_AI_actor":
    policy_net = GaussianMixtureMLP(args.NUM_ensemble,n_observations, n_actions)
    target_net = GaussianMixtureMLP(args.NUM_ensemble,n_observations, n_actions)
    policy_net_T = GaussianMixtureMLP(args.NUM_ensemble,n_observations+1, n_observations)
    target_net_T = GaussianMixtureMLP(args.NUM_ensemble,n_observations+1, n_observations)
    target_net_T.load_state_dict(policy_net_T.state_dict())
    actor_net = DQN(n_observations, n_actions)
    optimizer = torch.optim.AdamW(actor_net.parameters(), lr=1e-4, amsgrad=True)
##########################################################################################################

steps_done=0
if __name__=="__main__":
    cum_R=[]
    steps_episode=[]
    for i_episode in range(args.NUM_episodes):
        # Initialize the environment and get it's state
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        E_count=0
        if steps_done>args.max_steps:
            break
        for t in count():
            steps_done+=1
            if args.foot_record: 
                if steps_done<20000:
                    logging.info(f"foot: {state[0,0].item()} {state[0,1].item()}")
            # select action accroding to Free energy
            if args.model=="DQN":
                action,E=agent.select_action(state)
            elif args.model=="ensemble_DQN":
                action,E = agent.select_action(state)
            elif args.model=="model_1_AI":
                action,E = agent.select_action(state)
            elif args.model=="model_1_AI_actor":
                out=actor_net(state).squeeze()
                action=Categorical(torch.softmax(out,0)).sample().reshape(1,1)
                if action.item()==torch.argmax(out):
                    E=0
                else:
                    E=1
            # count the explore step number
            E_count+=E
            # step forward
            observation, reward, terminated, truncated, _ = env.step(action.item())
            
            reward = torch.tensor([reward], device=device)
            if args.env=="MountainCar-v0":
                done = terminated
            else:
                done = terminated or truncated


            if terminated:
                next_state=None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

            # Store the transition in memory
            agent.buffer.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network)
            # select action accroding to Free energy
            if args.model=="DQN":
                agent.update() 
            elif args.model=="ensemble_DQN":
                agent.update() 
            elif args.model=="model_1_AI":
                agent.update() 
            elif args.model=="model_1_AI_actor":
                optimize_model_ensemble(buffer,policy_net,target_net,GAMMA=args.GAMMA,BATCH_SIZE=args.BATCH_SIZE,device=device)
                optimize_model_ensemble(buffer,policy_net_T,target_net_T,GAMMA=args.GAMMA,BATCH_SIZE=args.BATCH_SIZE,device=device,T=True)
                optimize_actor(buffer,policy_net,policy_net_T,actor_net,optimizer,GAMMA=args.GAMMA,BATCH_SIZE=args.BATCH_SIZE,device=device)

            # soft update th target network
            # soft_update_model_weights(policy_net,target_net,args.TAU)

            if done:
                msg=f" {i_episode}, step: {t+1}, E: {E_count/(t+1)}, Total_steps: {steps_done}"
                print(msg)
                logging.info(msg)
                cum_R.append(t+1)
                steps_episode.append(steps_done)
                writer.add_scalar("cum R of episode",t+1,i_episode)
                writer.add_scalar("E rate",E_count/(t+1),i_episode)
                break

    print('Complete')
    plt.plot(cum_R)
    plt.savefig(f"imgs/cumR_episode_{args.model}_{i_episode}_{args.ID}.png")
    plt.show()
    plt.close()

    plt.plot(steps_episode,cum_R)
    plt.savefig(f"imgs/cumR_steps_{args.model}_{i_episode}_{args.ID}.png")
    plt.show()
    plt.close()

    torch.save(policy_net.state_dict(),f"models_saved/{args.env}_{args.model}_{i_episode}_{args.ID}.pt")    



    