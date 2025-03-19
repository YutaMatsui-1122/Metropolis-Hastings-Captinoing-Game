import json
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable

methods = {
    r"$\mathrm{COCO_{All}}$ Pretrain": {
        "json_file": "../exp_eval/pretrain/synonym_analysis/coco_2017_common_person_only_coco_all_candidate_all_epoch_temperature_1.0_eval.json",
        "color": "#404040"
    },
    r"$\mathrm{COCO_A}$ Pretrain": {
        "json_file": "../exp_eval/pretrain/synonym_analysis/coco_2017_common_person_only_coco_all_candidate_a_epoch_temperature_1.0_eval.json",
        "color": "#4D6E80"
    },
    r"$\mathrm{COCO_A}$ Fine-tune": {
        "json_file": "../exp_eval/all_acceptance_person_only_0/synonym_analysis/coco_all_candidate_A_epoch_29_temperature_1.0_eval.json",
        "color": "#134864"
    },
    r"$\mathrm{COCO_A}$ KD": {
        "json_file": "../exp_eval/distillation_16_32_0/synonym_analysis/coco_all_candidate_A_epoch__temperature_1.0_eval.json",
        "color": "#134864"
    },
    r"$\mathrm{COCO_A}$ MHCG": {
        "json_file": "../exp_eval/mhcg_person_only_0/synonym_analysis/coco_all_candidate_A_epoch_29_temperature_1.0_eval.json",
        "color": "#134158"
    },
    r"$\mathrm{COCO_B}$ Pretrain": {
        "json_file": "../exp_eval/pretrain/synonym_analysis/coco_2017_common_person_only_coco_all_candidate_b_epoch_temperature_1.0_eval.json",
        "color": "#8C6E46"
    },
    r"$\mathrm{COCO_B}$ Fine-tune": {
        "json_file": "../exp_eval/all_acceptance_person_only_0/synonym_analysis/coco_all_candidate_B_epoch_29_temperature_1.0_eval.json",
        "color": "#6C3807"
    },
    r"$\mathrm{COCO_B}$ KD": {
        "json_file": "../exp_eval/distillation_16_32_0/synonym_analysis/coco_all_candidate_B_epoch__temperature_1.0_eval.json",
        "color": "#6C3807"
    },
    r"$\mathrm{COCO_B}$ MHCG": {
        "json_file": "../exp_eval/mhcg_person_only_0/synonym_analysis/coco_all_candidate_B_epoch_29_temperature_1.0_eval.json",
        "color": "#6E3507"
    },
    r"$\mathrm{Weight}$ Averaging": {
        "json_file": "../exp_eval/linear_merged/synonym_analysis/coco_2017_common_person_only_coco_all_candidate_linear_merged_epoch_temperature_1.0_eval.json",
        "color": "#404040"
    },
    r"$\mathrm{Ensemble}$": {
        "json_file": "../exp_eval/ensemble_sampling/synonym_analysis/coco_2017_common_person_only_coco_all_candidate_ensemble_sampling_epoch_temperature_1.0_eval.json",
        "color": "#404040"
    },
    r"$\mathrm{PackLLM}$": {
        "json_file": "../exp_eval/packllm_sim/synonym_analysis/coco_2017_common_person_only_coco_all_candidate_packllm_sim_temperature_1.0_eval.json",
        "color": "#404040"
    },
}


def load_coco_categories(path):
    with open(path, "r") as f:
        annotations = json.load(f)
    return {str(cat["id"]): cat["name"] for cat in annotations["categories"]}

def load_data(json_file):
    with open(json_file, "r") as f:
        return json.load(f)

def extract_sorted_data(dataset):
    return {k: v for k, v in sorted(dataset.items(), key=lambda item: int(item[0]))}

coco_categories = load_coco_categories("../dataset/coco/annotations/instances_val2017.json")
all_methods = []
all_data = []
combined_category_names = None
num_dataset_a = None

for method, props in methods.items():
    data = load_data(props["json_file"])
    dataset_a = extract_sorted_data(data["dataset_a"]["CF1_per_category"])
    dataset_b = extract_sorted_data(data["dataset_b"]["CF1_per_category"])
    combined_data = list(dataset_a.values()) + list(dataset_b.values())
    if combined_category_names is None:
        combined_category_names = ([coco_categories.get(cat_id, f"Unknown({cat_id})") for cat_id in dataset_a.keys()] +
                                   [coco_categories.get(cat_id, f"Unknown({cat_id})") for cat_id in dataset_b.keys()])
        num_dataset_a = len(dataset_a)
    all_methods.append(method)
    all_data.append(combined_data)

heatmap_data = np.array(all_data)
num_methods, num_categories = heatmap_data.shape

left_data = ma.masked_array(heatmap_data, mask=False)
left_data[:, num_dataset_a:] = ma.masked
right_data = ma.masked_array(heatmap_data, mask=False)
right_data[:, :num_dataset_a] = ma.masked

left_cmap = LinearSegmentedColormap.from_list("left_cmap", [(0, "white"), (1, "#195675")])
right_cmap = LinearSegmentedColormap.from_list("right_cmap", [(0, "white"), (1, "#934709")])

fig, ax = plt.subplots(figsize=(24, 12))
extent = [-0.5, num_categories - 0.5, num_methods - 0.5, -0.5]
im_left = ax.imshow(left_data, aspect='auto', cmap=left_cmap, interpolation='none', extent=extent)
im_right = ax.imshow(right_data, aspect='auto', cmap=right_cmap, interpolation='none', extent=extent)
ax.axvline(x=num_dataset_a - 0.5, color="black", linestyle="--", linewidth=2)
ax.set_xticks(np.arange(num_categories))
ax.set_xticklabels(combined_category_names, rotation=90, fontsize=19)
ax.set_xlabel("Category Name", fontsize=26)
ax.set_yticks(np.arange(num_methods))
ax.set_yticklabels(all_methods, fontsize=20)

for tick_label in ax.get_yticklabels():
    method = tick_label.get_text()
    if method in methods:
        tick_label.set_color(methods[method]["color"])
        if "MHCG" in method:
            tick_label.set_fontweight("bold")

divider = make_axes_locatable(ax)
cax_left = divider.append_axes("right", size="2%", pad=0.2)
cax_right = divider.append_axes("right", size="2%", pad=0.05)
cbar_left = fig.colorbar(im_left, cax=cax_left, orientation="vertical")
cbar_right = fig.colorbar(im_right, cax=cax_right, orientation="vertical")
cbar_right.ax.tick_params(labelsize=16)
cbar_right.set_label(r"$CF1^S~~$Score", fontsize=24)
plt.text(0.32, 0.92, r'$\mathbf{Categories~in~ COCO_A}$', ha='center', va='bottom', fontsize=26, fontweight='bold', transform=fig.transFigure, color="#195675")
plt.text(0.73, 0.92, r'$\mathbf{Categories~ in~ COCO_B}$', ha='center', va='bottom', fontsize=26, fontweight='bold', transform=fig.transFigure, color="#934709")
plt.tight_layout()
plt.savefig("../results_for_paper/heatmap_cf1_scores_modified.pdf")
print("Saved heatmap to heatmap_cf1_scores_modified.pdf")
