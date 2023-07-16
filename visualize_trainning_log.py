import matplotlib.pyplot as plt
import numpy as np


def get_E_R(logfile):
    R=[]
    E=[]
    with open(logfile, 'r') as file:
        for line in file:
            # Process each line
            R.append(float(line.split()[3]))
            E.append(float(line.split()[7]))
    return E,R
def get_state(logfile):
    s1=[]
    s2=[]
    with open(logfile, 'r') as file:
        for line in file:
            if line.find("foot")>-1:
                # Process each line
                s1.append(float(line.split()[-2]))
                s2.append(float(line.split()[-1]))
    return s1,s2   
def plot_three_median_fill_area(E_ensemble,filename=None,reward=True):
    median_losses=E_ensemble[:,1]
    lower_bounds = E_ensemble[:,0] 
    upper_bounds = E_ensemble[:,-1]

    # Create the x-axis values
    x = np.arange(lower_bounds.shape[0])

    # Plot the median losses
    if reward:
        plt.plot(x, median_losses, color='blue', label='Median reward')
    else:
        plt.plot(x, median_losses, color='blue', label='Median E-rate')
        

    # Fill the area between the upper and lower bounds
    plt.fill_between(x, lower_bounds, upper_bounds, color='lightblue', alpha=0.5)

    # Set plot labels and title
    if reward:
        plt.xlabel('episode number')
        plt.ylabel('reward')
        plt.title('R v.s. episode')
    else:
        plt.xlabel('episode number')
        plt.ylabel('E-rate')
        plt.title('E-rate v.s. episode')
 

    # Add legend
    plt.legend()

    if filename is not None:
        plt.savefig(f"imgs/results/{filename}.png")

    plt.show()
def plot_three_median_fill_area_compare(E_ensemble,E_DQN,filename=None,reward=True,w=50):
    median_losses=moving_average(E_ensemble[:,1],w)
    lower_bounds =moving_average (E_ensemble[:,0],w) 
    upper_bounds =moving_average (E_ensemble[:,-1],w)
    median_losses_DQN=moving_average(E_DQN[:,1],w)
    lower_bounds_DQN =moving_average( E_DQN[:,0],w) 
    upper_bounds_DQN =moving_average( E_DQN[:,-1],w)


    # Create the x-axis values
    x = np.arange(lower_bounds.shape[0])

    # Plot the median losses
    plt.plot(x, median_losses, color='blue', label='Ensemble')
    plt.plot(x, median_losses_DQN, color='red', label='DQN')
        

    # Fill the area between the upper and lower bounds
    plt.fill_between(x, lower_bounds, upper_bounds, color='lightblue', alpha=0.5)
    plt.fill_between(x, lower_bounds_DQN, upper_bounds_DQN, color='red', alpha=0.5)

    # Set plot labels and title
    if reward:
        plt.xlabel('episode number')
        plt.ylabel('reward')
        plt.title('R v.s. episode')
    else:
        plt.xlabel('episode number')
        plt.ylabel('E-rate')
        plt.title('E-rate v.s. episode')
 

    # Add legend
    plt.legend()

    if filename is not None:
        plt.savefig(f"imgs/results/{filename}.png")

    plt.show()
    

def timestep_Reward(logfilename,save=""):

    E,R=get_E_R(logfilename)
    R=np.array(R)
    x=R.copy()
    for i in range(len(R)):
        x[i]=R[:i].sum()
    
    plt.plot(x,R)
    plt.xlabel("timesteps")
    plt.ylabel("reward")
    if save != "":
        
        plt.savefig(f"imgs/results/{save}.png")

    plt.show()
def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w   
# timestep_Reward("log/bootstrap_DQN_Acrobot-v1_prior0.0_0.txt","cartpole_timestep_R_double uncertainty")
    
    
    
    
    
    
# s1,s2=get_state("log\DQN_foot.txt")
# plt.scatter(s1,s2,alpha=0.2,linewidths=0.1)
# plt.xlim([-1.2,0.6])
# plt.ylim([-0.07,0.07])
# plt.show()
    
# s1,s2=get_state("log\ensemble_DQN_foot.txt")
# plt.scatter(s1,s2,alpha=0.2,linewidths=0.1)
# plt.xlim([-1.2,0.6])
# plt.ylim([-0.07,0.07])
# plt.show()
    
    
    
# exit()
    
    
    
    
    
    
  
E1,R1=get_E_R("log/bootstrap_DQN_Acrobot-v1_prior20.0_std2.0_711_1.txt")
E2,R2=get_E_R("log/bootstrap_DQN_Acrobot-v1_prior20.0_std2.0_711_2.txt")
E3,R3=get_E_R("log/bootstrap_DQN_Acrobot-v1_prior20.0_std2.0_711_3.txt")

clip=2000
E_ensemble=np.array([E1[:clip],E2[:clip],E3[:clip]]).T
R_ensemble=np.array([R1[:clip],R2[:clip],R3[:clip]]).T
E_sort_en=np.sort(E_ensemble,1)
R_sort_en=np.sort(R_ensemble,1)

# plot_three_median_fill_area(E_sort_en,"E_rate_ensemble",reward=False)
# plot_three_median_fill_area(R_sort_en,"R_ensemble",reward=True)

E1,R1=get_E_R("log/bootstrap_DQN_Acrobot-v1_prior0.0_73.txt")
E2,R2=get_E_R("log/bootstrap_DQN_Acrobot-v1_prior0.0_76.txt")
E3,R3=get_E_R("log/bootstrap_DQN_Acrobot-v1_prior0.0_79.txt")
# E1,R1=get_E_R("log/DQN_Acrobot-v1_prior0.0_79.txt")
# E2,R2=get_E_R("log/DQN_Acrobot-v1_prior0.0_76.txt")
# E3,R3=get_E_R("log/DQN_Acrobot-v1_prior0.0_73.txt")

E_ensemble=np.array([E1[:clip],E2[:clip],E3[:clip]]).T
R_ensemble=np.array([R1[:clip],R2[:clip],R3[:clip]]).T
E_sort=np.sort(E_ensemble,1)
R_sort=np.sort(R_ensemble,1)

# plot_three_median_fill_area(E_sort,"E_rate_DQN",reward=False)
# plot_three_median_fill_area(R_sort,"R_DQN",reward=True)

plot_three_median_fill_area_compare(R_sort_en,R_sort,"compare_b202_b0_R",True)
plot_three_median_fill_area_compare(E_sort_en,E_sort,"compare_b202_b0_E",False)
