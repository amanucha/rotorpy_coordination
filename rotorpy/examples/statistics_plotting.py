import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.ticker as ticker # Required import

# --- Configuration ---
FILE_NAME = 'experiment_results.csv'
COLUMNS = ['communication_disturbance_interval', 'no_communication_percentage', 'consensus_time']
sns.set_style("whitegrid")
plt.rcParams.update({'font.size': 12})


def perform_interaction_analysis(file_path, cols):
    """
    Performs comprehensive EDA including univariate, bivariate (discrete), 
    and interaction analysis (heatmap).
    """
    try:
        # 1. Load Data
        df = pd.read_csv(file_path)
        df_eda = df[cols].copy()
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found. Please ensure it is in the correct directory.")
        return
    except KeyError as e:
        print(f"Error: One or more required columns are missing in the CSV: {e}")
        return
    
    TARGET_COL = 'consensus_time'
    INPUT_COL_1 = 'communication_disturbance_interval'
    INPUT_COL_2 = 'no_communication_percentage'
    
    # 2. Check for missing values (Essential step, included for completeness)
    if df_eda.isnull().sum().any():
         print("--- ⚠️ Warning: Missing Values Found. Imputation/Removal is recommended. ---")
    
    # --- Visualization Code Starts Here ---
    
    # --- Step 1: Univariate Analysis (for completeness, although not the focus now) ---
    print("--- 📊 Generating Univariate Visualizations (Skipping to Interaction Plot for Focus) ---")
    # ... (Code for Histograms/Boxplots would go here) ...
    
    # --- Step 2: Bivariate Discrete Analysis (Previous Request: Boxplots) ---
    print("--- 📦 Generating Discrete Parameter Boxplots (Skipping to Interaction Plot for Focus) ---")
    # ... (Code for Boxplots would go here) ...
    
    # --- Step 3: Interaction Analysis (NEW: Heatmap) ---
    print("\n--- 🔥 Generating Interaction Heatmap Plot ---")

    # Calculate the average consensus_time for every unique combination of the two parameters
    pivot_table = df_eda.pivot_table(
        values=TARGET_COL, 
        index=INPUT_COL_1, 
        columns=INPUT_COL_2, 
        aggfunc='mean'
    )

    plt.figure(figsize=(12, 8))
    sns.heatmap(
        pivot_table,
        annot=True, 
        fmt=".2f", # Format to 2 decimal places
        cmap='viridis', # Good for continuous data
        linewidths=.5, 
        linecolor='black',
        cbar_kws={'label': f'Mean {TARGET_COL.replace("_", " ").title()} (Time Units)'}
    )
    
    # Set titles and labels
    plt.title('Interaction Effect on Consensus Time (Mean)', fontsize=16)
    plt.xlabel(INPUT_COL_2.replace("_", " ").title())
    plt.ylabel(INPUT_COL_1.replace("_", " ").title())
    
    # Ensure all ticks are integers and visible
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig('05_Interaction_Heatmap.png')
    plt.close()
    print("Saved: 05_Interaction_Heatmap.png (Interaction Heatmap)")

    print("\n--- ✅ Interaction Analysis Script Execution Complete ---")

# --- Execute the Function ---
perform_interaction_analysis(FILE_NAME, COLUMNS)

# --- Interpretation Guide for the Heatmap ---
print("\n\n--- 💡 Focused Interpretation Guide (Interaction Heatmap) ---")

print("This plot is designed to capture the **interaction effect**, where the impact of one variable (e.g., communication interval) depends on the level of the other variable (e.g., communication percentage).")

print("### Key Insights to Look For:")
print("1. **Synergistic Failure (The 'Hot Spot'):** Look for the area on the map (usually a corner) where the color is the **hottest** (highest consensus time). This combination of specific disturbance interval and communication loss percentage represents the system's worst-case performance.")
print("2. **Independent vs. Dependent Effects:**")
print("   * **Independent:** If the color gradually changes across rows (fixed communication interval, changing percentage) and columns (fixed percentage, changing interval), the effects are mostly independent and additive.")
print("   * **Interaction:** If the color is generally light, but suddenly spikes (a small, dark cluster) only when *both* parameters are at specific adverse levels, you have a strong **interaction effect**. For instance, a long interval might be fine *unless* the percentage loss is also at 50% or higher.")
print("3. **Optimal Operating Zone:** Look for the **coldest** area (lowest consensus time). This combination represents the most stable and fastest scenario, which is usually when communication loss is low and the disturbance interval is long.")



def plot_index_vs_target(file_path, target_col):
    """
    Plots the target column against the DataFrame index (run order).
    """
    try:
        # 1. Load Data
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found. Please ensure it is in the current directory.")
        return
    except KeyError:
        print(f"Error: The column '{target_col}' was not found in the CSV.")
        print(f"Available columns: {df.columns.tolist()}")
        return

    print("--- 📉 Generating Index vs. Consensus Time Plot ---")

    plt.figure(figsize=(14, 6))

    # Plot the consensus_time against the DataFrame index (0, 1, 2, ...)
    plt.plot(
        df.index, 
        df[target_col].values, # Using .values for robustness
        marker='o', 
        linestyle='-', 
        linewidth=1, 
        markersize=4, 
        color='darkblue',
        alpha=0.6
    )

    # Add a horizontal line for the mean consensus time to provide context
    mean_time = df[target_col].mean()
    plt.axhline(mean_time, color='red', linestyle='--', linewidth=2, label=f'Mean Time ({mean_time:.2f})')
    ax = plt.gca() 
    ax.xaxis.set_major_locator(ticker.MultipleLocator(5)) 
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    # Set titles and labels
    plt.title(f'{target_col.replace("_", " ").title()} vs. Experiment Run Order (Index)', fontsize=16)
    plt.xlabel('Experiment Run Order (DataFrame Index)')
    plt.ylabel(target_col.replace("_", " ").title())
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('06_Index_vs_ConsensusTime.png')
    plt.close()
    print("Saved: 06_Index_vs_ConsensusTime.png")

    print("\n--- ✅ Plot Generation Script Complete ---")

# --- Execute the Function ---
plot_index_vs_target(FILE_NAME, "consensus_time")