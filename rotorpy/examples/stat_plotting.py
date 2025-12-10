import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# --- Configuration ---
FILE_NAME = 'experiment_results.csv'
COLUMNS = ['communication_disturbance_interval', 'no_communication_percentage', 'consensus_time']
sns.set_style("whitegrid")
# Set font size for better readability
plt.rcParams.update({'font.size': 12})


def perform_eda(file_path, cols):
    """
    Performs comprehensive Exploratory Data Analysis (EDA) including descriptive
    statistics and multiple visualizations.
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
        print(f"Available columns: {df.columns.tolist()}")
        return
    
    print("--- 📋 Data Overview and Descriptive Statistics ---")
    print("\nFirst 5 Rows:")
    print(df_eda.head().to_markdown(index=False))
    
    print("\nData Types and Missing Values Check:")
    print(df_eda.info())
    
    # 2. Descriptive Statistics
    desc_stats = df_eda.describe().T
    print("\nDetailed Descriptive Statistics:")
    print(desc_stats.to_markdown(floatfmt=".2f"))

    # Check for missing values
    if df_eda.isnull().sum().any():
        print("\n--- ⚠️ Warning: Missing Values Found ---")
        print(df_eda.isnull().sum())
        # Discuss imputation/removal strategy here if applicable
    else:
        print("\n--- ✅ No Missing Values Found ---")


    # 3. Univariate Analysis (Histograms and Boxplots)
    print("\n--- 📊 Generating Univariate Visualizations (Distribution & Outliers) ---")
    
    fig, axes = plt.subplots(len(cols), 2, figsize=(15, 4 * len(cols)))
    fig.suptitle('Distribution and Outlier Analysis of Key Variables', fontsize=18, y=1.01)

    for i, col in enumerate(cols):
        title = col.replace("_", " ").title()
        
        # Histogram
        sns.histplot(df_eda[col].values, kde=True, ax=axes[i, 0], bins=20, color='skyblue')
        axes[i, 0].set_title(f'Distribution of {title}')
        axes[i, 0].set_xlabel(title)
        
        # Boxplot
        sns.boxplot(y=df_eda[col].values, ax=axes[i, 1], color='lightcoral')
        axes[i, 1].set_title(f'Outlier Check for {title}')
        axes[i, 1].set_ylabel(title)
        axes[i, 1].set_xticks([]) # Remove x-axis ticks
        
    plt.tight_layout()
    plt.savefig('01_univariate_analysis.png')
    plt.close(fig)
    print("Saved: 01_univariate_analysis.png (Histograms and Boxplots)")


    # 4. Bivariate Analysis (Scatterplots)
    print("\n--- 📈 Generating Bivariate Visualizations (Relationships) ---")

    # Scatterplot Matrix for pairwise relationships
    g = sns.pairplot(df_eda, kind='reg', 
                     plot_kws={'line_kws':{'color':'red', 'alpha':0.7}, 'scatter_kws': {'alpha': 0.6}})
    g.fig.suptitle('Pairwise Relationships (Scatterplots) with Regression Line', y=1.02, fontsize=18)
    # Customize labels
    for ax in g.axes.flat:
        if ax.get_xlabel():
            ax.set_xlabel(ax.get_xlabel().replace("_", " ").title())
        if ax.get_ylabel():
            ax.set_ylabel(ax.get_ylabel().replace("_", " ").title())
    plt.savefig('02_pairwise_scatterplots.png')
    plt.close()
    print("Saved: 02_pairwise_scatterplots.png (Pairplot)")


    # 5. Correlation Analysis (Heatmap)
    print("\n--- 🔥 Generating Correlation Heatmap ---")

    correlation_matrix = df_eda.corr()

    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', linewidths=.5, linecolor='black',
                cbar_kws={'label': 'Pearson Correlation Coefficient'})
    plt.title('Correlation Matrix of Experiment Variables', fontsize=16)
    # Adjust for seaborn/matplotlib display bug
    plt.ylim(len(correlation_matrix), 0) 
    plt.tight_layout()
    plt.savefig('03_correlation_heatmap.png')
    plt.close()
    print("Saved: 03_correlation_heatmap.png (Correlation Heatmap)")
    
    print("\n--- ✅ EDA Script Execution Complete ---")


# --- Execute the Function ---
perform_eda(FILE_NAME, COLUMNS)

# --- Summary and Interpretation Guide ---
print("\n\n--- 💡 EDA Interpretation Guide ---")

print("\n## 1. Data Patterns and Variability (From Histograms/Boxplots):")
print("* **`communication_disturbance_interval`**: Look at the histogram shape. Is it uniform, normal, or skewed? High variability (wide boxplot) suggests a wide range of tested disturbance settings.")
print("* **`no_communication_percentage`**: This often follows a bounded distribution (0-100%). Look for clustering near the boundaries. The boxplot will highlight any specific high or low values.")
print("* **`consensus_time`**: This is the key outcome. A highly skewed (right-skewed) distribution might indicate that most runs are fast, but a few problematic runs take significantly longer. This skewness might require a log or square root **data transformation** before modeling.")

print("\n## 2. Potential Relationships (From Scatterplots/Heatmap):")
print("* **`consensus_time` vs. `no_communication_percentage`**: We would expect a **positive correlation**. As the percentage of lost communication increases, the time to reach consensus should increase. The correlation heatmap coefficient (e.g., $r = 0.85$) will quantify this strength.")
print("* **`consensus_time` vs. `communication_disturbance_interval`**: The relationship might be more complex. Does a *shorter* interval (more frequent disturbance) lead to a *higher* consensus time? The sign of the correlation coefficient will confirm this.")

print("\n## 3. Outliers and Data Transformation:")
print("* Check the **Boxplots** for points outside the whiskers (outliers). Outliers in `consensus_time` (e.g., extremely long times) are critical. They could be errors, or they could represent rare but important failure scenarios.")
print("* If **`consensus_time`** is heavily skewed, a **log transformation** (e.g., $Y' = \log(Y)$) should be considered to normalize the distribution, which often improves the performance of linear regression and other models.")

print("\n## 4. Initial Modeling Insights:")
print(f"* If a strong, linear correlation exists between the inputs (e.g., `no_communication_percentage`) and the target (`consensus_time`), a **Linear Regression** model might be a strong baseline.")
print("* If the relationships are non-linear or feature interactions are suspected, a **Tree-based model** (like Random Forest or XGBoost) would be more appropriate, as they don't rely on linear assumptions.")