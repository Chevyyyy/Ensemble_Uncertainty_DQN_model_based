import matplotlib.pyplot as plt
import numpy as np
from visualize_trainning_log import *


def main(env_name="Acrobot-v1",alpha=10,std=False):

    E,R1,S=get_E_R_S(f"log/bootstrap_DQN_{env_name}_prior0.0_std0.0_726_1_bs.txt")
    E,R2,S=get_E_R_S(f"log/bootstrap_DQN_{env_name}_prior0.0_std0.0_726_2_bs.txt")
    E,R3,S=get_E_R_S(f"log/bootstrap_DQN_{env_name}_prior0.0_std0.0_726_3_bs.txt")
    E,R4,S=get_E_R_S(f"log/bootstrap_DQN_{env_name}_prior0.0_std0.0_726_4_bs.txt")
    E,R5,S=get_E_R_S(f"log/bootstrap_DQN_{env_name}_prior0.0_std0.0_726_5_bs.txt")

    means=np.concatenate([R1[:,:],R2[:,:],R3[:,:],R4[:,:],R5[:,:]],0)
    mean_std_steps(means,alpha,"BS",std=std)

    
    E,R1,S=get_E_R_S(f"log/bootstrap_DQN_{env_name}_prior0.0_std0.0_726_1_bsp.txt")
    E,R2,S=get_E_R_S(f"log/bootstrap_DQN_{env_name}_prior0.0_std0.0_726_2_bsp.txt")
    E,R3,S=get_E_R_S(f"log/bootstrap_DQN_{env_name}_prior0.0_std0.0_726_3_bsp.txt")
    E,R4,S=get_E_R_S(f"log/bootstrap_DQN_{env_name}_prior0.0_std0.0_726_4_bsp.txt")
    E,R5,S=get_E_R_S(f"log/bootstrap_DQN_{env_name}_prior0.0_std0.0_726_5_bsp.txt")

    means=np.concatenate([R1[:,:],R2[:,:],R3[:,:],R4[:,:],R5[:,:]],0)
    mean_std_steps(means,alpha,"BSP",std=std)

    E,R1,S=get_E_R_S(f"log/bootstrap_DQN_{env_name}_prior0.0_std0.0_726_1_bsdp.txt")
    E,R2,S=get_E_R_S(f"log/bootstrap_DQN_{env_name}_prior0.0_std0.0_726_2_bsdp.txt")
    E,R3,S=get_E_R_S(f"log/bootstrap_DQN_{env_name}_prior0.0_std0.0_726_3_bsdp.txt")
    E,R4,S=get_E_R_S(f"log/bootstrap_DQN_{env_name}_prior0.0_std0.0_726_4_bsdp.txt")
    E,R5,S=get_E_R_S(f"log/bootstrap_DQN_{env_name}_prior0.0_std0.0_726_5_bsdp.txt")
    means=np.concatenate([R1[:,:],R2[:,:],R3[:,:],R4[:,:],R5[:,:]],0)
    mean_std_steps(means,alpha,"BSDP",std=std)

    # E,R1,S=get_E_R_S(f"log/DQN_{env_name}_prior0.0_std0.0_726_1.txt")
    # E,R2,S=get_E_R_S(f"log/DQN_{env_name}_prior0.0_std0.0_726_2.txt")
    # E,R3,S=get_E_R_S(f"log/DQN_{env_name}_prior0.0_std0.0_726_3.txt")
    # E,R4,S=get_E_R_S(f"log/DQN_{env_name}_prior0.0_std0.0_726_4.txt")
    # E,R5,S=get_E_R_S(f"log/DQN_{env_name}_prior0.0_std0.0_726_5.txt")

    # means=np.concatenate([R1[:,:],R2[:,:],R3[:,:],R4[:,:],R5[:,:]],0)
    # mean_std_steps(means,alpha,"DQN")

    E,R1,S=get_E_R_S(f"log/bootstrap_DQN_{env_name}_prior0.0_std0.0_726_1_mbbsdp.txt")
    E,R2,S=get_E_R_S(f"log/bootstrap_DQN_{env_name}_prior0.0_std0.0_726_2_mbbsdp.txt")
    E,R3,S=get_E_R_S(f"log/bootstrap_DQN_{env_name}_prior0.0_std0.0_726_3_mbbsdp.txt")
    E,R4,S=get_E_R_S(f"log/bootstrap_DQN_{env_name}_prior0.0_std0.0_726_4_mbbsdp.txt")
    E,R5,S=get_E_R_S(f"log/bootstrap_DQN_{env_name}_prior0.0_std0.0_726_5_mbbsdp.txt")

    means=np.concatenate([R1[:,:],R2[:,:],R3[:,:],R4[:,:],R5[:,:]],0)
    mean_std_steps(means,alpha,"MBBSDP",std=std)


    plt.title(f"{env_name}")
    plt.legend(loc=2)
    
 
main("Acrobot-v1",50,1)
plt.savefig("imgs/Acrobot_BSP_BS_BSDP_MBBSDP.png")
plt.close()

main("MountainCar-v0",50,1)
plt.savefig("imgs/MountainCar_BSP_BS_BSDP_MBBSDP.png")
plt.close()

main("CartPole-v1",50,1)
plt.savefig("imgs/CartPole_BSP_BS_BSDP_MBBSDP.png")
plt.close()