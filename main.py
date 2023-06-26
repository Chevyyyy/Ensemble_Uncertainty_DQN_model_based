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
from agent.PPO_agent import PPO 
from agent.SAC_Agent import SAC
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser(description='|model|env|')
parser.add_argument("--model",default="ensemble_DQN",help="SAC|DQN|PPO|ensemble_DQN|bootstrap_DQN|model_1_AI|model_1_AI_actor")
parser.add_argument("--env",default="CartPole-v1",help="CartPole-v1|MountainCar-v0|LunarLander-v2|Acrobot-v1|Pendulum-v1|ALE/Breakout-v5|ALE/MontezumaRevenge-v5|MinAtar/Breakout-v1")
parser.add_argument("--BATCH_SIZE",type=int,default=32)
parser.add_argument("--NUM_episodes",type=int,default=20000)
parser.add_argument("--GAMMA",default=0.99)
parser.add_argument("--TAU",default=0.005)
parser.add_argument("--PRINT",default=False)
parser.add_argument("--render_mode",default="rgb_array")
parser.add_argument("--device",default="cpu")
parser.add_argument("--NUM_ensemble",default=5)
parser.add_argument("--ID",default="DEBUG")
parser.add_argument("--foot_record",default=False)
parser.add_argument("--max_steps",type=int,default=1e5)
parser.add_argument("--repeat_average",type=int,default=3)
parser.add_argument("--eval_intervel",type=int,default=10)
parser.add_argument("--eval",type=int,default=0)
parser.add_argument("--update_intervel",type=int,default=1)
args = parser.parse_args()
if args.env.find("/")>-1:
    args.CNN=True
else:
    args.CNN=False
envstr=args.env.split("/")[-1]


###############################################################################################
# config the args
# set the log file
set_log_file(f"log/{args.model}_{envstr}_{args.ID}.txt")
writer = SummaryWriter(f"runs/{envstr}/{args.model}_{envstr}_{args.ID}")
# set env
env = gym.make(args.env,render_mode=args.render_mode)
# Get number of actions from gym action space
n_actions = env.action_space.n
# Get the number of state observations
state, info = env.reset()
state_shape = state.shape
# set device
device = torch.device(args.device)

# set the model
if args.model=="DQN":
    agent = DQN(state_shape, n_actions,env,args.CNN,GAMMA=args.GAMMA,BATCH_SIZE=args.BATCH_SIZE)
elif args.model=="PPO":
    agent = PPO(state_shape, n_actions,env,writer)
elif args.model=="ensemble_DQN":
    agent = DQN_ensemble(args.NUM_ensemble,state_shape,n_actions,writer,args.CNN,GAMMA=args.GAMMA,BATCH_SIZE=args.BATCH_SIZE)
elif args.model=="bootstrap_DQN":
    agent = DQN_ensemble(args.NUM_ensemble,state_shape,n_actions,writer,args.CNN,GAMMA=args.GAMMA,BATCH_SIZE=args.BATCH_SIZE,bootstrap=True)
elif args.model=="model_1_AI":
    agent = model_1_AI(args.NUM_ensemble,state_shape,n_actions)
elif args.model=="SAC":
    agent = SAC(state_shape,n_actions)
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
        cum_R_float=0
        if steps_done>args.max_steps:
            break
        for t in count():
            steps_done+=1
            if args.foot_record: 
                if steps_done<20000:
                    logging.info(f"foot: {state[0,0].item()} {state[0,1].item()}")
            # select action accroding to Free energy
            action,E,action_prob = agent.select_action(state)
            # count the explore step number
            E_count+=E
            # step forward
            observation, reward, terminated, truncated, _ = env.step(action.item())
            cum_R_float+=reward
            
            reward = torch.tensor([reward])
            terminated = torch.tensor([terminated],dtype=torch.float32)
            action_prob = torch.tensor([action_prob],dtype=torch.float32)

            done = terminated or truncated

            next_state = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)

            # Store the transition in memory
            agent.buffer.push(state, action,action_prob, next_state, reward,terminated)

            # Move to the next state
            state = next_state

            # update the network 
            if steps_done%args.update_intervel==0:
                agent.update() 

            if done:
                if i_episode%args.eval_intervel==0 and args.eval==True:
                    eva_cum_R=evaluate(env,agent,repeat_average=args.repeat_average)
                    print(f"{i_episode}, evaluate cum R: {eva_cum_R}, Total_steps: {steps_done}")
                    writer.add_scalar("eva cum R of steps",eva_cum_R,steps_done)
                    writer.add_scalar("eva cum R of episode",eva_cum_R,i_episode)
                msg=f" {i_episode}  R: {cum_R_float} step: {t+1}  E: {E_count/(t+1)}  Total_steps: {steps_done}"
                print(msg)
                logging.info(msg)
                cum_R.append(t+1)
                steps_episode.append(steps_done)
                writer.add_scalar("cum R of episode",cum_R_float,i_episode)
                writer.add_scalar("cum R of steps",cum_R_float,steps_done)
                writer.add_scalar("E rate",E_count/(t+1),i_episode)
                break

    print('Complete')
    plt.plot(cum_R)
    plt.savefig(f"imgs/cumR_episode_{args.model}_{i_episode}_{args.ID}.png")
    # plt.show()
    plt.close()

    plt.plot(steps_episode,cum_R)
    plt.savefig(f"imgs/cumR_steps_{args.model}_{i_episode}_{args.ID}.png")
    # plt.show()
    plt.close()

    # torch.save(policy_net.state_dict(),f"models_saved/{args.env}_{args.model}_{i_episode}_{args.ID}.pt")    



    