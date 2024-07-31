import pandas as pd

# lower
control_path = "results/1_paper/main/full_t5-base_llama3_t5_base_p3_Y0728-013556.json"
control_df = pd.read_json(control_path, lines=True)

# higher
experiment_paths = [
    "results/1_paper/E9/s0/full_t5-base_llama3_t5_base_p3_Y0731-033529.json",
    "results/1_paper/E9/s100/full_t5-base_llama3_t5_base_p3_Y0731-033530.json",
    "results/1_paper/E9/s200/full_t5-base_llama3_t5_base_p3_Y0731-033530.json",
    "results/1_paper/E9/s300/full_t5-base_llama3_t5_base_p3_Y0731-033530.json",
    "results/1_paper/E9/s400/full_t5-base_llama3_t5_base_p3_Y0731-033529.json",
]
# experiment_df = pd.read_json(experiment_path, lines=True)
experiment_df = pd.concat([pd.read_json(path, lines=True) for path in experiment_paths])

# calculate 1 tailed t test
from scipy.stats import ttest_ind
t_stat, p_value = ttest_ind(experiment_df["m1_masked_a2_f1"], control_df["m1_masked_a2_f1"], alternative="greater")
print(f"t_stat: {t_stat}, p_value: {p_value}")
print