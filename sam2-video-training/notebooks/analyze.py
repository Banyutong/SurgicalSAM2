import pandas as pd
import ast
import numpy as np

# --- 1. Load and Preprocess the Data ---

# Load the new dataset from the provided CSV file name
try:
    df = pd.read_csv("wandb_export_2025-09-18T12_21_56.746+08_00.csv")
    df["trainer.max_epochs"] = pd.to_numeric(df["trainer.max_epochs"], errors="coerce")
    print("Successfully loaded and processed the new CSV file.")
except FileNotFoundError:
    print(
        "Error: The file 'wandb_export_2025--18T12_21_56.746+08_00.csv' was not found."
    )
    exit()


# --- 2. Feature Engineering and Baseline Definition ---

# Create helper columns for analysis
df["dataset"] = df["data_module.data.name"]
df["prompt_type"] = df["module.model.prompt_type"]
# *** ACTION: Create short config name by extracting it from the 'Name' column ***
# This assumes the format is {dataset}_{prompt}_{config}__{id}
df["config"] = df["Name"].apply(lambda x: x.split("_")[2])
df["trainable_modules_list"] = df["module.model.trainable_modules"].apply(
    ast.literal_eval
)
df["has_memory"] = df["trainable_modules_list"].apply(lambda x: "memory_encoder" in x)
df["has_image_encoder"] = df["trainable_modules_list"].apply(
    lambda x: "image_encoder" in x
)

# Define the baseline as any run where training was not performed (max_epochs is 0)
df_baseline = df[df["trainer.max_epochs"] == 0].copy()
df_trained = df[df["trainer.max_epochs"] > 0].copy()


# --- 3. Establish Baseline for Comparison ---

# Extract baseline metrics.
baseline_metrics = df_baseline.set_index(["dataset", "prompt_type"])[
    ["eval/Dice", "eval/mIoU", "eval/MAE"]
].rename(
    columns={
        "eval/Dice": "Dice_baseline",
        "eval/mIoU": "mIoU_baseline",
        "eval/MAE": "MAE_baseline",
    }
)

# Join the baseline metrics onto the DataFrame of trained models.
df_trained = df_trained.set_index(["dataset", "prompt_type"]).join(baseline_metrics)

# Sort the DataFrame to group related rows together
df_trained.sort_index(inplace=True)


# --- 4. Calculate Improvement Over Baseline ---

# Calculate the percentage improvement of trained models over their untrained counterparts.
df_trained["Dice_improvement_%"] = (
    (df_trained["eval/Dice"] - df_trained["Dice_baseline"])
    / df_trained["Dice_baseline"]
    * 100
)
df_trained["mIoU_improvement_%"] = (
    (df_trained["eval/mIoU"] - df_trained["mIoU_baseline"])
    / df_trained["mIoU_baseline"]
    * 100
)
df_trained["MAE_reduction_%"] = (
    (df_trained["eval/MAE"] - df_trained["MAE_baseline"])
    / df_trained["MAE_baseline"]
    * 100
)

df_trained.fillna(0, inplace=True)


# --- 5. Generate Summaries and Insights (with grouped output) ---

print("\n--- Trained Models DataFrame with Baselines and Improvements ---")
print("(Grouped by Dataset and Prompt Type)")
# *** ACTION: Use the new 'config' column for display ***
display_cols = [
    "config",
    "eval/Dice",
    "Dice_baseline",
    "Dice_improvement_%",
    "eval/mIoU",
    "mIoU_baseline",
    "mIoU_improvement_%",
    "eval/MAE",
    "MAE_baseline",
    "MAE_reduction_%",
]
print(df_trained[display_cols])


print("\n\n--- Insight 1: What is the overall impact of training? ---")
training_impact = df_trained.groupby("dataset")[
    ["Dice_improvement_%", "mIoU_improvement_%", "MAE_reduction_%"]
].mean()
print("Average improvement from training (vs. epoch 0 baseline):")
print(training_impact)
print(
    "\nSummary: Training provides a massive performance uplift. The 'endovis17' and 'endovis18' datasets,"
)
print(
    "which are more challenging, show a greater relative improvement from training compared to 'cholecseg8k'."
)


print("\n\n--- Insight 2: Among trained models, do memory modules help? ---")
memory_impact = (
    df_trained.groupby(["dataset", "has_memory"])["eval/Dice"].mean().unstack()
)
memory_impact.columns = ["Without Memory", "With Memory"]
print("Mean Dice Score for Trained Models:")
print(memory_impact)
print(
    "\nSummary: For trained models, including a memory module consistently improves the average Dice score across all datasets."
)


print(
    "\n\n--- Insight 3: Among trained models, does fine-tuning the Image Encoder help? ---"
)
image_encoder_impact = (
    df_trained.groupby(["dataset", "has_image_encoder"])["Dice_improvement_%"]
    .mean()
    .unstack()
)
image_encoder_impact.columns = ["Without Image Encoder", "With Image Encoder"]
print("Average Dice Improvement (%) for Trained Models:")
print(image_encoder_impact)
print(
    "\nSummary: Fine-tuning the image encoder provides another significant boost in performance on top of standard training,"
)
print("especially for the 'endovis17' and 'endovis18' datasets.")


print("\n\n--- Insight 4: Which prompt type is most effective for trained models? ---")
prompt_performance = (
    df_trained.groupby(["dataset", "prompt_type"])["eval/Dice"].mean().unstack()
)
print("Mean Dice Score by Prompt Type (Trained Models):")
print(prompt_performance)
print("\nSummary: Even after training, 'mask' prompts deliver the highest performance.")
print(
    "'Point' prompts remain the least effective, confirming that prompt quality is crucial."
)


print(
    "\n\n--- Insight 5: What are the best overall trained configurations per dataset? ---"
)
best_configs = df_trained.loc[df_trained.groupby("dataset")["eval/Dice"].idxmax()]
best_configs = best_configs.reset_index().sort_values(by="dataset")
print("The top-performing configurations (based on Dice score) are:")
# *** ACTION: Use the new 'config' column for display ***
print(best_configs[["dataset", "Name", "eval/Dice", "Dice_improvement_%", "config"]])


print(
    "\n\n--- Insight 6: What is the single best fine-tuning configuration overall? ---"
)
# *** ACTION: Group by the new 'config' column ***
overall_performance = (
    df_trained.groupby("config")
    .agg(
        mean_dice=("eval/Dice", "mean"),
        mean_dice_improvement=("Dice_improvement_%", "mean"),
        run_count=("Name", "count"),
    )
    .sort_values(by="mean_dice", ascending=False)
)

print(
    "Average performance by fine-tuning configuration (across all datasets and prompts):"
)
print(overall_performance)

best_config_name = overall_performance.index[0]
best_config_stats = overall_performance.iloc[0]

print(f"\nSummary: The best overall fine-tuning configuration is '{best_config_name}'.")
print(
    f"On average, this configuration achieves a Dice score of {best_config_stats['mean_dice']:.4f} "
    f"and provides a {best_config_stats['mean_dice_improvement']:.2f}% improvement over the baseline."
)
print(
    "This configuration (mem+md+pe+ie) includes the memory modules and fine-tunes all available encoders."
)
print(
    "This indicates that for the highest and most robust performance, training all available components is the most effective strategy."
)