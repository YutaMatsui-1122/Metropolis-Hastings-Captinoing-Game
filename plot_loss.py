import numpy as np
import matplotlib.pyplot as plt

exp_path = "exp/gemcg_unified_dataset_coco5000_cc3m5000_agentA05_agentBbeta005_1"


update_epoch = 10

for agent_name in ["A", "B"]:
    loss_list = []

    for i in range(10):
        file_path = f"{exp_path}/{agent_name}/clipcap_{agent_name}_{i}_loss.npy"

        loss = np.load(file_path)

        # update_epochの数にlossを分割し，それぞれの平均を取る
        loss = np.mean(np.split(loss, update_epoch), axis=1)
        loss_list.extend(loss)

    print(loss_list)
    print(len(loss_list))



    plt.plot(loss_list, label="loss")
    plt.xlabel("iteration", fontsize=14)
    plt.ylabel("loss", fontsize=14)
    # update_epochごとに薄い色で縦線を引く
    for i in range(1, len(loss_list)//update_epoch):
        if i ==1:
            plt.axvline(x=i*update_epoch, color="gray", linestyle="--", label="sign_update_epoch")
        else:
            plt.axvline(x=i*update_epoch, color="gray", linestyle="--")
    plt.legend(fontsize=14)
    plt.savefig(f"{exp_path}/{agent_name}/clipcap_{agent_name}_loss.png")

    plt.clf()