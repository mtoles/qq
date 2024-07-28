import pandas as pd

# lower
control_path = "results/1_paper/main/full_t5-base_gpt-4_t5_base_p2_Y0724-000418.json"
control_df = pd.read_json(control_path, lines=True)

# higher
experiment_path = "results/llama3_ft/checkpointed/E6/full_t5-base_llama3_t5_base_p3_Y0722-045952.json"
experiment_df = pd.read_json(experiment_path, lines=True)

# calculate 1 tailed t test
from scipy.stats import ttest_ind
t_stat, p_value = ttest_ind(experiment_df["m1_masked_a2_f1"], control_df["m1_masked_a2_f1"], alternative="greater")
print(f"t_stat: {t_stat}, p_value: {p_value}")
print