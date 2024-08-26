import os
import matplotlib.pyplot as plt
import pandas as pd
from utils import *
# Specify the folder path
folder_path = "exp/coco_conceptual_"

exp_list = ["cos_annealing_low", "cos_annealing_mid", "cos_annealing_overfit", "wo_annealing_low", "wo_annealing_mid", "wo_annealing_overfit"]

judge_list = ["Reject", "Accept"]

# Plot the data
plt.figure(figsize=(10, 5))

sign_index = 0



for exp_name in exp_list:
    coincident=0

    i = 0
    acceeptance_rate = []
    file_name_A = f"agent_A_{i}.csv"
    file_path_A = os.path.join(folder_path+exp_name, file_name_A)
    df_A = pd.read_csv(file_path_A)
    sign_A = [[judge_list[int(df_A["accept"][sign_index])], df_A["after_w"][sign_index]]]
    sign_A_set = [[i, df_A["after_w"][sign_index]]]

    file_name_B = f"agent_B_{i}.csv"
    file_path_B = os.path.join(folder_path+exp_name, file_name_B)
    df_B = pd.read_csv(file_path_B)
    sign_B = [[judge_list[int(df_B["accept"][sign_index])], df_B["after_w"][sign_index]]]
    sign_B_set = [[i, df_B["after_w"][sign_index]]]

    while True:        
        # Generate the file name
        file_name_A = f"agent_A_{i}.csv"
        file_path_A = os.path.join(folder_path+exp_name, file_name_A)
        file_name_B = f"agent_B_{i}.csv"
        file_path_B = os.path.join(folder_path+exp_name, file_name_B)
        # Check if the file exists
        if not os.path.isfile(file_path_A):
            break

        # Read the CSV file into a pandas DataFrame
        df_A = pd.read_csv(file_path_A)
        sign_A.append([judge_list[int(df_A["accept"][sign_index])], df_A["after_w"][sign_index]])
        # acceeptance_rate.append(df_A['accept'].mean())
        if df_A["after_w"][sign_index] != sign_A_set[-1][1]:
            sign_A_set.append([i, df_A["after_w"][sign_index]])
        
        df_B = pd.read_csv(file_path_B)
        sign_B.append([judge_list[int(df_B["accept"][sign_index])], df_B["after_w"][sign_index]])
        acceeptance_rate.append(df_B['accept'].mean())
        if df_B["after_w"][sign_index] != sign_B_set[-1][1]:
            sign_B_set.append([i, df_B["after_w"][sign_index]])

        if df_A["after_w"][sign_index] == df_B["after_w"][sign_index]:
            coincident += 1

        i += 1
    
    print(f"{exp_name}: coincident rate: {coincident / i}")

    # Plot the data
    plt.plot(acceeptance_rate, label=exp_name)
    df = pd.DataFrame(sign_A, columns = ['Acceptance', 'sign'])
    df.to_csv(f"exp/coco_conceptual_{exp_name}_{sign_index}_A.csv")

    df_set = pd.DataFrame(sign_A_set, columns = ['MH_iter', 'sign'])
    df_set.to_csv(f"exp/coco_conceptual_{exp_name}_{sign_index}_set_A.csv")

    df = pd.DataFrame(sign_B, columns = ['Acceptance', 'sign'])
    df.to_csv(f"exp/coco_conceptual_{exp_name}_{sign_index}_B.csv")

    df_set = pd.DataFrame(sign_B_set, columns = ['MH_iter', 'sign'])
    df_set.to_csv(f"exp/coco_conceptual_{exp_name}_{sign_index}_set_B.csv")


# Add labels
plt.xlabel("MH iteration")
plt.ylabel("Acceptance rate")

# Add legend outside of the plot
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# Adjust layout to make room for the legend
plt.tight_layout()

# Save the plot with the legend outside
plt.savefig("exp/coco_conceptual.png")

