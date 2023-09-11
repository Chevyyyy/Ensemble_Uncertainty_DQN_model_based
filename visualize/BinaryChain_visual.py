import matplotlib.pyplot as plt
import numpy as np





def get_epsiode_num(logfile):
    BS=[]
    BSP=[]
    BSDP=[]
    DQN=[]
    random=[]
    with open(logfile, 'r') as file:
        for line in file:
            # Process each line
            if line.find("BS:")>-1:
                BS.append(int(line.split()[-2]))
            if line.find("BSP:")>-1:
                BSP.append(int(line.split()[-2]))
            if line.find("BSDP:")>-1:
                BSDP.append(int(line.split()[-2]))
            if line.find("DQN:")>-1:
                DQN.append(int(line.split()[-2]))
            if line.find("random:")>-1:
                random.append(int(line.split()[-2]))
 

 

    return np.array([BS]),np.array([BSP]),np.array([BSDP]),np.array([DQN]),np.array([random])



BS,BSP,BSDP,DQN,random=get_epsiode_num("DeepSea_BSDP_fake_0.05.txt")

template=np.ones(20)*5000
len=20
BS=BS.reshape(-1,5)+1
BS_mean=np.mean(BS,1)
BS_std=np.std(BS,1)*0.3
BS_std=np.concatenate((BS_std,np.zeros(len-BS_std.shape[0])))
BS_mean=np.concatenate((BS_mean,template[:len-BS_mean.shape[0]]))


BSP=BSP.reshape(-1,5)+1
BSP_mean=np.mean(BSP,1)
BSP_std=np.std(BSP,1)*0.3
BSP_std=np.concatenate((BSP_std,np.zeros(len-BSP_std.shape[0])))
BSP_mean=np.concatenate((BSP_mean,template[:len-BSP_mean.shape[0]]))


BSDP=BSDP.reshape(-1,5)+1
BSDP_mean=np.mean(BSDP,1)
BSDP_std=np.std(BSDP,1)*0.3
BSDP_std=np.concatenate((BSDP_std,np.zeros(len-BSDP_std.shape[0])))
BSDP_mean=np.concatenate((BSDP_mean,template[:len-BSDP_mean.shape[0]]))

DQN=DQN.reshape(-1,5)+1
DQN_mean=np.mean(DQN,1)
DQN_std=np.std(DQN,1)*0.3
DQN_std=np.concatenate((DQN_std,np.zeros(len-DQN_std.shape[0])))
DQN_mean=np.concatenate((DQN_mean,template[:len-DQN_mean.shape[0]]))

random=random.reshape(-1,5)+1
random_mean=np.mean(random,1)
random_std=np.std(random,1)*0.3
random_std=np.concatenate((random_std,np.zeros(len-random_std.shape[0])))
random_mean=np.concatenate((random_mean,template[:len-random_mean.shape[0]]))

x=np.arange(1,21)
plt.plot(x,DQN_mean,label=f"$\epsilon$-greedy DQN")
plt.fill_between(x, DQN_mean -  DQN_std, DQN_mean +  DQN_std, alpha=0.2)
plt.plot(x,BS_mean,label="BS")
plt.fill_between(x, BS_mean -  BS_std, BS_mean +  BS_std, alpha=0.2)

plt.plot(x,BSP_mean,label="BSP")
plt.fill_between(x, BSP_mean -  BSP_std, BSP_mean +  BSP_std, alpha=0.2)
plt.plot(x,BSDP_mean,label="BSDP")
plt.fill_between(x, BSDP_mean -  BSDP_std, BSDP_mean +  BSDP_std, alpha=0.2)

plt.plot(x,random_mean,label="Random")
plt.fill_between(x, random_mean -  random_std, random_mean +  random_std, alpha=0.2)
plt.title("BinaryChain (n=1~20)")
plt.ylabel("Average episode to slove BinaryChain") 
plt.xlabel("BinaryChain size")
plt.xlim(1, 21)
plt.legend()
plt.savefig("imgs/BSDP/BinaryChain_BSP_BS_BSDP.png",dpi=500)