import pandas as pd

control_path = "results/1_paper/main/full_t5-base_llama3_t5_base_p3_Y0709-202534.json"
control_df = pd.read_json(control_path, lines=True)

experiment_path = "results/llama3_ft/full_t5-base_llama3_t5_base_p3_Y0715-214651.json"
experiment_df = pd.read_json(experiment_path, lines=True)

# control_mean = control_df["m1_masked_a2_f1"].mean()
# experiment_mean = experiment_df["m1_masked_a2_f1"].mean()

# control_stderr = control_df["m1_masked_a2_f1"].sem()
# experiment_stderr = experiment_df["m1_masked_a2_f1"].sem()

# calculate 1 tailed t test
from scipy.stats import ttest_ind
t_stat, p_value = ttest_ind(experiment_df["m1_masked_a2_f1"], control_df["m1_masked_a2_f1"], alternative="greater")
print(f"t_stat: {t_stat}, p_value: {p_value}")
print