import matplotlib.pyplot as plt
import numpy as np
from visualize.visualize_trainning_log import *


def main(env_name="Acrobot-v1",alpha=10,std=False,y_lable=None):
    epsiode_num=500


    E,R5,S=get_E_R_S(f"log/classic_control/bootstrap_DQN_{env_name}_MBSP_0.txt")
    E,R1,S=get_E_R_S(f"log/classic_control/bootstrap_DQN_{env_name}_MBSP_1.txt")
    E,R2,S=get_E_R_S(f"log/classic_control/bootstrap_DQN_{env_name}_MBSP_2.txt")
    E,R3,S=get_E_R_S(f"log/classic_control/bootstrap_DQN_{env_name}_MBSP_3.txt")
    E,R4,S=get_E_R_S(f"log/classic_control/bootstrap_DQN_{env_name}_MBSP_4.txt")

    means=np.concatenate([R1[:,:epsiode_num],R2[:,:epsiode_num],R3[:,:epsiode_num],R4[:,:epsiode_num],R5[:,:epsiode_num]],0)
    var_buffer=np.zeros((5,500))
    scale_IM=np.zeros((5,500))
    for i in range(500):
        var_buffer[:,i]=means[:,:i+1].var(1)
    scale_IM=1/np.clip(1*var_buffer,0.1,10)


    
    mean_std_steps(var_buffer,alpha,"MBBSDP",std=std,y_lable=y_lable)
    # mean_std_steps(scale_IM,alpha,"Intrinsic Reward Scale Factor",std=std,y_lable=y_lable)

    plt.title(f"{env_name}")
    # plt.legend(loc=1)
    
# main("MountainCar-v0",50,1,y_lable="Intrinsic Reward Scale Factor")
# plt.savefig("imgs/DU/MountainCar_scale.png",dpi=500)
# plt.close()

main("MountainCar-v0",50,1,y_lable="Varience of Episodic Rewards")
plt.savefig("imgs/DU/MountainCar_buffer_var.png",dpi=500)
plt.close()