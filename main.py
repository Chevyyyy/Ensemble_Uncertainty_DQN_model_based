#!/usr/bin/python3
import gym
import matplotlib.pyplot as plt
from itertools import count
import torch
from networks.deep_ensemble_NN_model import GaussianMixtureMLP
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
from agent.R_uncertainty_Agent import R_uncertainty
from agent.SAC_Agent import SAC
from env.deepsea import DeepSea 
try:
    import gym_maze
except:
    pass
from torch.utils.tensorboard import SummaryWriter


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device: ", device)
    
    
parser = argparse.ArgumentParser(description="|model|env|")
parser.add_argument(
    "--model",
    default="bootstrap_DQN",
    help="SAC|DQN|PPO|ensemble_DQN|bootstrap_DQN|model_1_AI|model_1_AI_actor|R_uncertainty",
)
parser.add_argument(
    "--env",
    default="MinAtar/Freeway-v1",
    help="maze-sample-5x5|maze-random-10x10-v0|MountainCarContinuous-v0|CartPole-v1|MountainCar-v0|LunarLander-v2|Acrobot-v1|Pendulum-v1|ALE/Breakout-v5|ALE/MontezumaRevenge-v5|MinAtar/Breakout-v0|MinAtar/Freeway-v1|maze2d-open-v0|DeepSea",
)
parser.add_argument("--BATCH_SIZE", type=int, default=64)
parser.add_argument("--NUM_episodes", type=int, default=600)
parser.add_argument("--LR", type=float, default=1e-4)
parser.add_argument("--GAMMA", default=0.99)
parser.add_argument("--TAU", default=0.005)
parser.add_argument("--buffer", default=30000, type=int)
parser.add_argument("--render", default=0, type=int)
parser.add_argument("--device", default="cpu")
parser.add_argument("--A_Change", type=int, default=0)
parser.add_argument("--prior", default=0, type=float)
parser.add_argument("--prior_noise", default=0, type=float)
parser.add_argument("--NUM_ensemble", default=5)
parser.add_argument("--ID", default="DEBUG")
parser.add_argument("--log_folder", default=None)
parser.add_argument("--foot_record", default=False)
parser.add_argument("--max_steps", type=int, default=2e8)
parser.add_argument("--seed", default=0, type=int)
parser.add_argument("--repeat_average", type=int, default=3)
parser.add_argument("--eval_intervel", type=int, default=10)
parser.add_argument("--eval", type=int, default=0)
parser.add_argument("--OLD_GYM", type=int, default=0)
parser.add_argument("--no_truncated", type=int, default=0)
parser.add_argument("--update_intervel", type=int, default=1)
parser.add_argument("--update_epoch", type=int, default=1)

# the property of the Algorithm
parser.add_argument("--real_bootstrap", type=int, default=1)
parser.add_argument("--p_net", type=int, default=0)
parser.add_argument("--DP_init", type=int, default=0)
parser.add_argument("--var_net", type=int, default=0)
parser.add_argument("--T_net", type=int, default=0)


args = parser.parse_args()
def main(args):
    if args.OLD_GYM:
        import d4rl
    if args.env.find("/") > -1:
        args.CNN = True
    else:
        args.CNN = False
    envstr = args.env.split("/")[-1]


    ###############################################################################################
    # config the args
    # set the log file
    config = f"{args.model}_{envstr}_{args.ID}"
    print("###########################")
    print("###########################")
    print("trainning start: ", config)
    print("###########################")
    print("###########################")
    if args.log_folder is not None:
        set_log_file(f"log/{args.log_folder}/{config}.txt")
    else:
        set_log_file(f"log/{config}.txt")   
    if args.log_folder is not None:
        writer = SummaryWriter(f"runs/{args.log_folder}/{envstr}/{config}")
    else:
        writer = SummaryWriter(f"runs/{envstr}/{config}")
    # set env
    if args.env.split("_")[0] == "DeepSea":
        size=int(args.env.split("_")[1])
        env = DeepSea(size,args.seed)
        print(env._action_mapping)
    else:
        env = gym.make(args.env)
        if (not args.OLD_GYM) and args.render:
            env = gym.make(args.env, render_mode="human")

    # Get number of actions from gym action space
    n_actions = env.action_space.n
    # Get the number of state observations
    if args.OLD_GYM:
        state = env.reset()
    else:
        state, info = env.reset()

    state_shape = state.shape
    # set device
    device = torch.device(args.device)

    # set the model
    if args.model == "DQN":
        agent = DQN(
            state_shape,
            n_actions,
            env,
            args.CNN,
            GAMMA=args.GAMMA,
            BATCH_SIZE=args.BATCH_SIZE,
            prior=args.prior,
            buffer_size=args.buffer,
        )
    elif args.model == "PPO":
        agent = PPO(state_shape, n_actions, env, writer, prior=args.prior)
    elif args.model == "ensemble_DQN":
        agent = DQN_ensemble(
            env,
            args.NUM_ensemble,
            state_shape,
            n_actions,
            writer,
            args.CNN,
            GAMMA=args.GAMMA,
            BATCH_SIZE=args.BATCH_SIZE,
            prior=args.prior,
            prior_noise=args.prior_noise,
            p_net=args.p_net,
            buffer_size=args.buffer,
        )
    elif args.model == "bootstrap_DQN":
        agent = DQN_ensemble(
            env,
            args.NUM_ensemble,
            state_shape,
            n_actions,
            writer,
            args.CNN,
            GAMMA=args.GAMMA,
            BATCH_SIZE=args.BATCH_SIZE,
            bootstrap=True,
            prior=args.prior,
            prior_noise=args.prior_noise,
            DP_init=args.DP_init,
            real_bootstrap=args.real_bootstrap,
            A_change=args.A_Change,
            var_net_flag=args.var_net,
            p_net=args.p_net,
            T_net=args.T_net,
            buffer_size=args.buffer,
            lr=args.LR,
        )
    elif args.model == "model_1_AI":
        agent = model_1_AI(args.NUM_ensemble, state_shape, n_actions)
    elif args.model == "SAC":
        agent = SAC(state_shape, n_actions)
    elif args.model == "R_uncertainty":
        agent = R_uncertainty(
            state_shape,
            n_actions,
            writer,
            env,
            args.CNN,
            GAMMA=args.GAMMA,
            BATCH_SIZE=args.BATCH_SIZE,
            TAU=args.TAU,
            prior=args.prior,
        )
    ##########################################################################################################


    steps_done = 0
    cum_R = []
    steps_episode = []
    for i_episode in range(args.NUM_episodes):

        state_list_deepsea=[]
        # Initialize the environment and get it's state
        if args.OLD_GYM:
            state = env.reset()
            if args.render:
                env.render()
        else:
            state, info = env.reset()
            state_list_deepsea.append(state.squeeze().tolist())
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0).to(device=device)
        E_count = 0
        cum_R_float = 0.0

        if steps_done > args.max_steps:
            break


        for t in count():
            # print(f"steps_done: {steps_done}")
            steps_done += 1
            if args.foot_record:
                if steps_done < 20000:
                    logging.info(f"foot: {state[0,0].item()} {state[0,1].item()}")
            # select action accroding to Free energy

            action, E, action_prob = agent.select_action(state)
            # count the explore step number
            E_count += E
            # step forward
            truncated = False
            if args.OLD_GYM:
                observation, reward, terminated, _ = env.step(action.item())
                if args.render:
                    env.render()
            else:
                observation, reward, terminated, truncated, _ = env.step(action.item())
            cum_R_float += reward
            state_list_deepsea.append(observation.squeeze().tolist())

            reward = torch.tensor([reward],dtype=torch.float32).to(device=device)
            terminated = torch.tensor([terminated], dtype=torch.float32).to(device=device)
            action_prob = torch.tensor([action_prob], dtype=torch.float32).to(device=device)

            if args.env == "MountainCar-v0":
                if t == 999:
                    truncated = True
                else:
                    truncated = False
            if args.no_truncated:
                truncated = False
            done = terminated or truncated

            next_state = torch.tensor(observation, dtype=torch.float32).unsqueeze(0).to(device=device)

            # Store the transition in memory
            agent.buffer.push(
                state, action, action_prob, next_state, reward, terminated,1
            )

            # Move to the next state
            state = next_state

            # update the network
            if steps_done % args.update_intervel == 0:
                for i in range(args.update_epoch):
                    agent.update()

            # if steps_done % 100 == 0:
            #     getRM(model=agent.Ensemble_Q_net,plot=False)
            #     getRM_mean(model=agent.Ensemble_Q_net,plot=False)

            if done:
                if i_episode % args.eval_intervel == 0 and args.eval == True:
                    eva_cum_R = evaluate(env, agent, repeat_average=args.repeat_average)
                    print(
                        f"{i_episode}, evaluate cum R: {eva_cum_R}, Total_steps: {steps_done}"
                    )
                    writer.add_scalar("eva cum R of steps", eva_cum_R, steps_done)
                    writer.add_scalar("eva cum R of episode", eva_cum_R, i_episode)
                msg = f" {i_episode}  R: {cum_R_float} step: {t+1}  E: {E_count/(t+1)}  Total_steps: {steps_done}"
                print(msg)  
                if args.env.split("_")[0] == "DeepSea":
                    print(f"ending state: {state_list_deepsea}")
                logging.info(msg)
                cum_R.append(t + 1)
                steps_episode.append(steps_done)
                writer.add_scalar("cum R of episode", cum_R_float, i_episode)
                writer.add_scalar("cum R of steps", cum_R_float, steps_done)
                writer.add_scalar("E rate", E_count / (t + 1), i_episode)
                agent.buffer.save_cum_R(float(cum_R_float))
                break

        if args.env.split("_")[0] == "DeepSea" and cum_R_float > 0.9: 
            print(f"solve {args.env} in {i_episode} episode")
            return True,i_episode
    print("Complete: ", config)
    return False,i_episode
    
 
def run_deep_sea():
    args.NUM_episodes=5000
    args.NUM_ensemble=10
    args.BATCH_SIZE=64
    args.real_bootstrap=0
    seed_array=np.random.randint(100,size=5)
    seed_array=[58 ,99 ,43 ,37 ,43]


    f = open(f"DeepSea_{args.ID}.txt", "a",buffering=1)
    f.write(seed_array.__str__()+"\n")

    # args.model="DQN"
    # for i in range(1,15): 
    #     success=0
    #     args.env="DeepSea_"+str(i)
    #     args.update_intervel=i
    #     for j in range(5):
    #         args.seed=seed_array[j]
    #         args.ID="804_"+str(j+1)
    #         suc,i_epsiode=main(args)
    #         success+=suc
    #         msg=f"DeepSea_{i} random: solve in {i_epsiode} episode\n"
    #         f.write(msg)
    #     if success<5:
    #         break


            
    
    # args.model="bootstrap_DQN"
    # args.p_net=False
    # for i in range(1,15): 
    #     success=0
    #     args.env="DeepSea_"+str(i)
    #     args.update_intervel=i
    #     for j in range(5):
    #         args.seed=seed_array[j]
    #         args.ID="bs804_"+str(j+1)
    #         suc,i_epsiode=main(args)
    #         success+=suc
    #         msg=f"DeepSea_{i} BS: solve in {i_epsiode} episode\n"
    #         f.write(msg)
    #     if success<5:
    #         break

    # args.model="bootstrap_DQN"
    # args.p_net=True
    # for i in range(1,15): 
    #     success=0
    #     args.env="DeepSea_"+str(i)
    #     args.update_intervel=i
    #     for j in range(5):
    #         args.seed=seed_array[j]
    #         args.ID="bsp804_"+str(j+1)
    #         suc,i_epsiode=main(args)
    #         success+=suc
    #         msg=f"DeepSea_{i} BSP: solve in {i_epsiode} episode\n"
    #         f.write(msg)
    #     if success<5:
    #         break


    # args.model="bootstrap_DQN"
    # args.p_net=True
    # args.DP_init=True
    # for i in range(15,20): 
    #     success=0
    #     args.env="DeepSea_"+str(i)
    #     args.update_intervel=i
    #     for j in range(5):
    #         args.seed=seed_array[j]
    #         args.ID="bsdp804_"+str(j+1)
    #         suc,i_epsiode=main(args)
    #         success+=suc
    #         msg=f"DeepSea_{i} BSDP: solve in {i_epsiode} episode\n"
    #         f.write(msg)
    #     if success<5:
    #         break

    args.model="bootstrap_DQN"
    args.p_net=1
    args.DP_init=0
    args.T_net=1 
    for i in range(1,20): 
        success=0
        args.env="DeepSea_"+str(i)
        args.update_intervel=i
        for j in range(5):
            args.seed=seed_array[j]
            args.ID="mbbsdp804_"+str(j+1)
            suc,i_epsiode=main(args)
            success+=suc
            msg=f"DeepSea_{i} DU: solve in {i_epsiode} episode\n"
            f.write(msg)
        if success<5:
            break

    f.close()

def run():
    main(args)
if __name__ == "__main__":
    if args.env.split("_")[0] == "DeepSea":
        run_deep_sea() 
    else:
        run()