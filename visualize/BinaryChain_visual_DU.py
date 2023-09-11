import matplotlib.pyplot as plt
import numpy as np





def get_epsiode_num(logfile):
    BSP=[]
    random=[]
    DU=[]
    with open(logfile, 'r') as file:
        for line in file:
            # Process each line
            if line.find("BSP:")>-1:
                BSP.append(int(line.split()[-2]))
            if line.find("random:")>-1:
                random.append(int(line.split()[-2]))
            if line.find("DU:")>-1:
                DU.append(int(line.split()[-2]))
 

 

    return np.array([BSP]),np.array([random]),np.array([DU])


BSP,random,DU=get_epsiode_num("DeepSea_BSDP_fake_0.05.txt")

template=np.ones(20)*5000
len=20


BSP=BSP.reshape(-1,5)+1
BSP_mean=np.mean(BSP,1)
BSP_std=np.std(BSP,1)*0.3
BSP_std=np.concatenate((BSP_std,np.zeros(len-BSP_std.shape[0])))
BSP_mean=np.concatenate((BSP_mean,template[:len-BSP_mean.shape[0]]))


random=random.reshape(-1,5)+1
random_mean=np.mean(random,1)
random_std=np.std(random,1)*0.3
random_std=np.concatenate((random_std,np.zeros(len-random_std.shape[0])))
random_mean=np.concatenate((random_mean,template[:len-random_mean.shape[0]]))

DU=DU.reshape(-1,5)+1
DU_mean=np.mean(DU,1)
DU_std=np.std(DU,1)*0.3
DU_std=np.concatenate((DU_std,np.zeros(len-DU_std.shape[0])))
DU_mean=np.concatenate((DU_mean,template[:len-DU_mean.shape[0]]))


x=np.arange(1,21)
plt.plot(x,BSP_mean,label="BSP")
plt.fill_between(x, BSP_mean -  BSP_std, BSP_mean +  BSP_std, alpha=0.2)

plt.plot(x,DU_mean,label="DU")
plt.fill_between(x, DU_mean -  DU_std, DU_mean +  DU_std, alpha=0.2)

plt.plot(x,random_mean,label="Random")
plt.fill_between(x, random_mean -  random_std, random_mean +  random_std, alpha=0.2)
plt.title("BinaryChain (n=1~20)")
plt.ylabel("Average episode to slove BinaryChain") 
plt.xlabel("BinaryChain size")
plt.xlim(1, 21)
plt.legend()
plt.savefig("imgs/DU/BinaryChain_BSP_random_DU.png",dpi=500)