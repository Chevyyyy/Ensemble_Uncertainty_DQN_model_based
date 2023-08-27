import matplotlib.pyplot as plt
import numpy as np




def get_epsiode_num(logfile):
    BS=[]
    BSP=[]
    BSDP=[]
    with open(logfile, 'r') as file:
        for line in file:
            # Process each line
            if line.find("BS:")>-1:
                BS.append(int(line.split()[-2]))
            if line.find("BSP:")>-1:
                BSP.append(int(line.split()[-2]))
            if line.find("BSDP:")>-1:
                BSDP.append(int(line.split()[-2]))
 
    return np.array([BS]),np.array([BSP]),np.array([BSDP])



BS,BSP,BSDP=get_epsiode_num("DeepSea_BSDP_fake_0.05.txt")
pass

BS=BS.reshape(-1,5)
BSP=BSP.reshape(-1,5)
BSDP=BSDP.reshape(-1,5)


plt.plot(np.mean(BS,1),label="BS")
plt.plot(np.mean(BSP,1),label="BSP")
plt.plot(np.mean(BSDP,1),label="BSDP")
plt.legend()
plt.show()