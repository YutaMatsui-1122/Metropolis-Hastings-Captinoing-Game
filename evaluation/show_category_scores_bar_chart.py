import json
import numpy as np
import matplotlib.pyplot as plt

# JSONファイルの設定
json_files = {
    "COCO_All Pretrain": "../exp_eval/pretrain/synonym_analysis/coco_2017_common_person_only_coco_all_candidate_all_epoch_temperature_1.0_eval.json",
    "COCO_A Pretrain": "../exp_eval/pretrain/synonym_analysis/coco_2017_common_person_only_coco_all_candidate_a_epoch_temperature_1.0_eval.json",
    "COCO_A Fine-tune": "../exp_eval/all_acceptance_person_only_0/synonym_analysis/coco_all_candidate_A_epoch_29_temperature_1.0_eval.json",
    "COCO_A KD": "../exp_eval/distillation_debug_0/synonym_analysis/coco_all_candidate_A_epoch__temperature_1.0_eval.json",
    "COCO_A MHCG": "../exp_eval/mhcg_person_only_0/synonym_analysis/coco_all_candidate_A_epoch_29_temperature_1.0_eval.json",
    "COCO_B Pretrain": "../exp_eval/pretrain/synonym_analysis/coco_2017_common_person_only_coco_all_candidate_b_epoch_temperature_1.0_eval.json",
    "COCO_B Fine-tune": "../exp_eval/all_acceptance_person_only_0/synonym_analysis/coco_all_candidate_B_epoch_29_temperature_1.0_eval.json",
    "COCO_B KD": "../exp_eval/distillation_debug_0/synonym_analysis/coco_all_candidate_B_epoch__temperature_1.0_eval.json",
    "COCO_B MHCG": "../exp_eval/mhcg_person_only_0/synonym_analysis/coco_all_candidate_B_epoch_29_temperature_1.0_eval.json",
    "Weight Averaging": "../exp_eval/linear_merged/synonym_analysis/coco_2017_common_person_only_coco_all_candidate_linear_merged_epoch_temperature_1.0_eval.json",
    "Ensemble": "../exp_eval/ensemble_sampling/synonym_analysis/coco_2017_common_person_only_coco_all_candidate_ensemble_sampling_epoch_temperature_1.0_eval.json",
    "PackLLM": "../exp_eval/packllm_sim/synonym_analysis/coco_2017_common_person_only_coco_all_candidate_packllm_sim_temperature_1.0_eval.json",
}

# Load COCO category names
def load_coco_categories(path):
    with open(path, "r") as f:
        annotations = json.load(f)
    return {str(cat["id"]): cat["name"] for cat in annotations["categories"]}

# Load data from JSON files
def load_data(json_file):
    with open(json_file, "r") as f:
        return json.load(f)

# Extract and sort data by category ID
def extract_sorted_data(dataset):
    return {k: v for k, v in sorted(dataset.items(), key=lambda item: int(item[0]))}

# COCO category processing
coco_categories = load_coco_categories("../dataset/coco/annotations/instances_val2017.json")
all_methods = []
all_data = []

for method_name, file_path in json_files.items():
    data = load_data(file_path)
    dataset_a = extract_sorted_data(data["dataset_a"]["CF1_per_category"])
    dataset_b = extract_sorted_data(data["dataset_b"]["CF1_per_category"])

    combined_data = list(dataset_a.values()) + list(dataset_b.values())
    combined_category_names = [coco_categories.get(cat_id, f"Unknown({cat_id})") for cat_id in dataset_a.keys()] + \
                              [coco_categories.get(cat_id, f"Unknown({cat_id})") for cat_id in dataset_b.keys()]

    all_methods.append(method_name)
    all_data.append(combined_data)

# Convert data to a NumPy array for bar chart
bar_data = np.array(all_data)
num_dataset_a = len(dataset_a)

# 色設定
method_colors = {
    "COCO_A Pretrain": "#7FB8D4",
    "COCO_A Fine-tune": "#1F78A6",
    "COCO_A KD": "#1F78A6",
    "COCO_A MHCG": "#195675",
    "COCO_B Pretrain": "#E8B874",
    "COCO_B Fine-tune": "#B45C0B",
    "COCO_B KD": "#B45C0B",
    "COCO_B MHCG": "#934709",
    "COCO_All Pretrain": "#606060",
    "Weight Averaging": "#A090B0",
    "Ensemble": "#A090B0",
    "PackLLM": "#A090B0",
}

# **修正: height_ratios のサイズを自動調整**
num_methods = len(all_methods)
height_ratios = [1.1] * num_methods
height_ratios.insert(1, 0.15)  # COCO_AとCOCO_Bの間
height_ratios.insert(6, 0.15)  # COCO_BとCOCO_Allの間
height_ratios.insert(11, 0.15)  # COCO_BとCOCO_Allの間

fig, axes = plt.subplots(num_methods + 3, 1, figsize=(24, 16), sharex=True,
                         gridspec_kw={"height_ratios": height_ratios})
plt.subplots_adjust(right=0.98, left=0.13, top=0.95, bottom=0.16)

# 空白サブプロットを非表示
axes[1].set_visible(False)
axes[6].set_visible(False)
axes[11].set_visible(False)

# Plot the bar charts
for i, (method, ax) in enumerate(zip(all_methods[:1] + [None] + all_methods[1:5] + [None] + all_methods[5:9] + [None] + all_methods[9:], axes)):
    if method is None:
        continue
    # 各セクション名を追加
    if i == 4:
        ax.text(-0.105, 0.5, "           " + r"$\mathbf{COCO_A}$", fontsize=26, ha="right", va="center", transform=ax.transAxes, rotation=90, color="#195675")
    if i == 9:
        ax.text(-0.105, 0.5, "           " + r"$\mathbf{COCO_B}$", fontsize=26, ha="center", va="center", transform=ax.transAxes, rotation=90, color="#934709")
    if i == 0:  
        ax.text(-0.105, 0.5, r"$\mathbf{COCO_{All}}$", fontsize=26, ha="right", va="center", transform=ax.transAxes, rotation=90)

    method_label = method.replace("COCO_A ", "").replace("COCO_B ", "").replace("COCO_All", "").replace("Ensemble", "Ensemble")

    args = {"rotation": 0, "ha": "right", "fontsize": 20, "labelpad": 10}
    if method_label in ["MHCG"]:
        args["fontweight"] = "bold"  

    ax.set_ylabel(method_label, **args)

    # **修正: インデックス調整**
    # adjusted_index = i if i < 3 else i - 2  # `None` の分だけ調整
    if i < 1:
        adjusted_index = i
    elif i < 7:
        adjusted_index = i - 1
    elif i < 11:
        adjusted_index = i - 2
    else:
        adjusted_index = i - 3

    ax.bar(combined_category_names, bar_data[adjusted_index], width=0.7, color=method_colors[method], alpha=0.9)

    ax.set_ylim(0, 1)  
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.axvline(x=num_dataset_a - 0.5, color="black", linestyle="--", linewidth=2)

    # 5つごとに縦線を追加（バーとバーの間に配置）
    for x in range(1, len(combined_category_names), 5):
        ax.axvline(x - 1.5, linestyle="--", color="gray", alpha=0.5)

# X-axis labels
axes[-1].set_xticks(range(len(combined_category_names)))  
axes[-1].set_xticklabels(combined_category_names, rotation=90, ha="center", fontsize=18)
axes[-1].set_xlabel("Category Name", fontsize=26)

# "Categories in COCO_A" と "Categories in COCO_B" を上に追加
plt.text(0.35, 0.96, r'$\mathbf{Categories~in~ COCO_A}$', ha='center', va='bottom', fontsize=26, fontweight='bold', transform=fig.transFigure, color="#195675")
plt.text(0.76, 0.96, r'$\mathbf{Categories~ in~ COCO_B}$', ha='center', va='bottom', fontsize=26, fontweight='bold', transform=fig.transFigure, color="#934709")

# Save the figure
plt.savefig("../results_for_paper/bar_chart_cf1_scores_separated.pdf")

print("Saved bar chart to bar_chart_cf1_scores_separated.pdf")
