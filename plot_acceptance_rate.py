import os
import numpy as np
import matplotlib.pyplot as plt

# ファイルのディレクトリを指定して、それぞれをつなげてプロットする関数
def plot_two_directories(directory_A, directory_B):
    combined_rates_A = []
    combined_rates_B = []

    # acceptance_rate_0.txt から acceptance_rate_24.txt までのファイルを順に読み込んで結合 (ディレクトリ A)
    for i in range(25):
        filename = f"acceptance_rate_{i}.txt"
        filepath_A = os.path.join(directory_A, filename)

        # ファイルを読み込んで値を取得 (ディレクトリ A)
        if os.path.exists(filepath_A):
            with open(filepath_A, 'r') as file:
                rates_A = [float(line.strip()) for line in file.readlines()]
                combined_rates_A.extend(rates_A)
        else:
            print(f"File not found: {filepath_A}")

    # acceptance_rate_0.txt から acceptance_rate_24.txt までのファイルを順に読み込んで結合 (ディレクトリ B)
    for i in range(25):
        filename = f"acceptance_rate_{i}.txt"
        filepath_B = os.path.join(directory_B, filename)

        # ファイルを読み込んで値を取得 (ディレクトリ B)
        if os.path.exists(filepath_B):
            with open(filepath_B, 'r') as file:
                rates_B = [float(line.strip()) for line in file.readlines()]
                combined_rates_B.extend(rates_B)
        else:
            print(f"File not found: {filepath_B}")

    # プロット
    plt.figure(figsize=(10, 6))
    plt.plot(combined_rates_A, label='Acceptance Rates A', color='blue')
    plt.plot(combined_rates_B, label='Acceptance Rates B', color='orange')

    # 文字サイズを大きく設定
    plt.title("Combined Acceptance Rates from Directories A and B", fontsize=20)
    plt.xlabel("Iteration", fontsize=20)
    plt.ylabel("Acceptance Rate", fontsize=20)
    plt.legend(loc="best", fontsize=18)
    plt.grid(True)

    # 結果をファイルに保存
    plt.savefig(f"{directory_A}/acceptance_rateAB.png", dpi=300)

# ここにディレクトリパスを指定して使用する
example_directory_A = "exp/mhcg_derpp_0.05_1/A"
example_directory_B = "exp/mhcg_derpp_0.05_1/B"
plot_two_directories(example_directory_A, example_directory_B)
