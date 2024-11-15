import os
import matplotlib.pyplot as plt

def load_acceptance_rates(directory):
    # acceptance_rate_0.txt から acceptance_rate_29.txt まで読み込む
    all_rates = []
    for i in range(30):  # 0 から 29 まで
        file_name = f"acceptance_rate_{i}.txt"
        file_path = os.path.join(directory, file_name)
        
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                # ファイル内の値を1行ずつ読み込んでリストに追加
                rates = [float(line.strip()) for line in f.readlines()]
                all_rates.extend(rates)
        else:
            print(f"Warning: {file_name} does not exist in the specified directory.")
    
    return all_rates

def plot_acceptance_rates(rates, directory):
    # リストをプロット
    plt.figure(figsize=(10, 6))
    plt.plot(rates, marker='o', linestyle='-', color='b')
    plt.title('Acceptance Rates', fontsize=18)
    plt.xlabel('Index', fontsize=14)
    plt.ylabel('Acceptance Rate', fontsize=14)
    plt.grid(True)

    # 保存するファイル名
    save_path = os.path.join(directory, 'acceptance_rates_plot.png')
    plt.savefig(save_path)
    plt.show()
    print(f"Plot saved as {save_path}")

# 使用例
directory = input("Enter the directory path containing the acceptance rate files: ")

# acceptance_rate のデータを読み込み
rates = load_acceptance_rates(directory)

# プロットして保存
plot_acceptance_rates(rates, directory)
# 