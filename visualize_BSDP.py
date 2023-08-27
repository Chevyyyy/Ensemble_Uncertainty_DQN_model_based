import matplotlib.pyplot as plt
import numpy as np
from visualize_trainning_log import *


def main(env_name="Acrobot-v1",alpha=10,std=False,R=True):
    epsiode_num=500
    # E,R5,S=get_E_R_S(f"log/classic_control/DQN_{env_name}_0.txt")
    # E,R1,S=get_E_R_S(f"log/classic_control/DQN_{env_name}_1.txt")
    # E,R2,S=get_E_R_S(f"log/classic_control/DQN_{env_name}_2.txt")
    # E,R3,S=get_E_R_S(f"log/classic_control/DQN_{env_name}_3.txt")
    # E,R4,S=get_E_R_S(f"log/classic_control/DQN_{env_name}_4.txt")

    # means=np.concatenate([R1[:,:],R2[:,:],R3[:,:],R4[:,:],R5[:,:]],0)
    # mean_std_steps(means,alpha,"DQN",std=std)

    E1,R5,S=get_E_R_S(f"log/classic_control/bootstrap_DQN_{env_name}_BS_0.txt")
    E2,R1,S=get_E_R_S(f"log/classic_control/bootstrap_DQN_{env_name}_BS_1.txt")
    E3,R2,S=get_E_R_S(f"log/classic_control/bootstrap_DQN_{env_name}_BS_2.txt")
    E4,R3,S=get_E_R_S(f"log/classic_control/bootstrap_DQN_{env_name}_BS_3.txt")
    E5,R4,S=get_E_R_S(f"log/classic_control/bootstrap_DQN_{env_name}_BS_4.txt")

    if R:
        means=np.concatenate([R1[:,:epsiode_num],R2[:,:epsiode_num],R3[:,:epsiode_num],R4[:,:epsiode_num],R5[:,:epsiode_num]],0)
        mean_std_steps(means,alpha,"BS",std=std)
    else:
        means=np.concatenate([E1[:,:epsiode_num],E2[:,:epsiode_num],E3[:,:epsiode_num],E4[:,:epsiode_num],E5[:,:epsiode_num]],0)
        mean_std_steps(means,alpha,"BS",std=std)


    E1,R5,S=get_E_R_S(f"log/classic_control/bootstrap_DQN_{env_name}_BSP_0.txt")
    E2,R1,S=get_E_R_S(f"log/classic_control/bootstrap_DQN_{env_name}_BSP_1.txt")
    E3,R2,S=get_E_R_S(f"log/classic_control/bootstrap_DQN_{env_name}_BSP_2.txt")
    E4,R3,S=get_E_R_S(f"log/classic_control/bootstrap_DQN_{env_name}_BSP_3.txt")
    E5,R4,S=get_E_R_S(f"log/classic_control/bootstrap_DQN_{env_name}_BSP_4.txt")

    if R:
        means=np.concatenate([R1[:,:epsiode_num],R2[:,:epsiode_num],R3[:,:epsiode_num],R4[:,:epsiode_num],R5[:,:epsiode_num]],0)
        mean_std_steps(means,alpha,"BSP",std=std)
    else:
        means=np.concatenate([E1[:,:epsiode_num],E2[:,:epsiode_num],E3[:,:epsiode_num],E4[:,:epsiode_num],E5[:,:epsiode_num]],0)
        mean_std_steps(means,alpha,"BSP",std=std)
        

    E1,R5,S=get_E_R_S(f"log/classic_control/bootstrap_DQN_{env_name}_BSDP_0.txt")
    E2,R1,S=get_E_R_S(f"log/classic_control/bootstrap_DQN_{env_name}_BSDP_1.txt")
    E3,R2,S=get_E_R_S(f"log/classic_control/bootstrap_DQN_{env_name}_BSDP_2.txt")
    E4,R3,S=get_E_R_S(f"log/classic_control/bootstrap_DQN_{env_name}_BSDP_3.txt")
    E5,R4,S=get_E_R_S(f"log/classic_control/bootstrap_DQN_{env_name}_BSDP_4.txt")

    if R:
        means=np.concatenate([R1[:,:epsiode_num],R2[:,:epsiode_num],R3[:,:epsiode_num],R4[:,:epsiode_num],R5[:,:epsiode_num]],0)
        mean_std_steps(means,alpha,"BSDP",std=std)
    else:
        means=np.concatenate([E1[:,:epsiode_num],E2[:,:epsiode_num],E3[:,:epsiode_num],E4[:,:epsiode_num],E5[:,:epsiode_num]],0)
        mean_std_steps(means,alpha,"BSDP",std=std)
        
        
        
    plt.title(f"{env_name}")
    plt.legend(loc=4)
    
main("Acrobot-v1",50,1,0)
plt.savefig("imgs/BSDP/Acrobot_BSP_BS_BSDP_E.png",dpi=500)
plt.close()

main("MountainCar-v0",50,1,0)
plt.savefig("imgs/BSDP/MountainCar_BSP_BS_BSDP_E.png",dpi=500)
plt.close()

main("CartPole-v1",50,1,0)
plt.savefig("imgs/BSDP/CartPole_BSP_BS_BSDP_E.png",dpi=500)
plt.close()