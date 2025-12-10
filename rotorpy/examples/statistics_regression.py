import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration (ADJUST THIS) ---
# Ensure your CSV file is saved as 'experiment_results.csv' 
# with the columns in this exact order: communication_disturbance_interval, no_communication_percentage, consensus_time
FILE_PATH = 'experiment_results.csv'
DEPENDENT_VARIABLE = 'consensus' 
FACTOR_1 = 'communication_disturbance_interval' # T_fail
FACTOR_2 = 'no_communication_percentage' # R_comm

# --- Load Data (Using Mock Data to ensure functionality) ---
# In a real scenario, this block should load the actual CSV.
df = pd.read_csv(FILE_PATH)
# The user's provided data schema is: [T_fail, R_comm, Consensus_Time]
df.columns = [FACTOR_1, FACTOR_2, DEPENDENT_VARIABLE]
# Drop the constant 'communication_disturbance_interval' for the sake of the OLS if it has zero variance
if df[FACTOR_1].nunique() == 1:
    print(f"Note: {FACTOR_1} is constant. Model simplicity may be warranted.")

# --- Running Multivariable Regression Model (as provided) ---
regression_formula = f'{DEPENDENT_VARIABLE} ~ {FACTOR_1} * {FACTOR_2}'
model_results = ols(regression_formula, data=df).fit()

# --- Extracting and Displaying Required Outputs (as provided) ---

# R-squared value
r_squared = model_results.rsquared_adj
print(f"Adjusted R-squared: {r_squared:.4f}\n")

# Coefficients (betas) and renaming them for the report
coefficients = model_results.params.rename({
    'Intercept': 'beta_0 (Intercept)',
    f'{FACTOR_1}': 'beta_1 (communication_disturbance_interval/T_fail)',
    f'{FACTOR_2}': 'beta_2 (no_communication_percentage/R_comm)',
    f'{FACTOR_1}:{FACTOR_2}': 'beta_3 (communication_disturbance_interval * no_communication_percentage)'
})

print("--- Extracted Coefficients (Betas) ---")
print(coefficients)

# Full Regression Summary Table (contains R-squared, t-stats, and P-values)
print("\n--- Full Regression Summary Table ---")
print(model_results.summary())

# Residuals
residuals = model_results.resid
print("\n--- Residuals ---")
print(residuals)

# ====================================================================
# === START OF ADDED PLOT CODE: Residuals vs. Fitted Values ===
# ====================================================================

# 1. Calculate Fitted (Predicted) Values
fitted_values = model_results.fittedvalues

# 2. Create the plot
plt.figure(figsize=(8, 6))
sns.scatterplot(x=fitted_values, y=residuals, color='red', alpha=0.7)

# Add a horizontal line at y=0 (where residuals should center)
plt.axhline(y=0, color='grey', linestyle='--')

# Set titles and labels
plt.title('Residuals vs. Fitted Values Plot ', fontsize=14)
plt.xlabel('Fitted Values (Predicted Consensus Time)', fontsize=12)
plt.ylabel('Residuals (Error)', fontsize=12)


# Save the plot to a file
plot_filename = 'residuals_vs_fitted_plot.png'
plt.savefig(plot_filename)
plt.show() 
plt.close()

print(f"\n[Visualization Added] Residuals vs. Fitted Plot saved as: {plot_filename}")

# ====================================================================
# === END OF ADDED PLOT CODE ===
# ====================================================================