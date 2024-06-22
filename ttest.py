import pandas as pd

control_path = "results/alexpaca/checkpointed/full_t5-base_alexpaca_t5_base_None_Y0622-043011.json"
control_df = pd.read_json(control_path, lines=True)

experiment_path = "results/alexpaca/checkpointed/full_t5-base_alexpaca_t5_base_None_Y0621-203331.json"
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