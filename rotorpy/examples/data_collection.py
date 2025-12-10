import subprocess
import csv
import os
import re # We use regex to safely find and replace variable lines

# --- Configuration for the Experiment Script ---
# The path to your main executable Python script.
EXECUTABLE_SCRIPT = "coordination_with_mpc_mixed.py"

# The path to your configuration file (e.g., config.py).
CONFIG_FILE = "../rotorpy/config.py"

# The name of the CSV file to save results.
RESULTS_CSV_FILE = "experiment_results.csv"

# The parameters you want to change (Variable_Name, Value_List)
PARAMETERS_TO_CHANGE = [
    ("communication_disturbance_interval", [5, 10, 20, 35, 50, 65, 70, 80, 100, 200]),
    ("no_communication_percentage", [0.1,0.2,0.3,0.4,0.5,0.6,0.7, 0.8, 0.9]),
]
def update_py_config(var_name, new_value):
    """
    Reads the config.py file, finds the line defining var_name,
    and replaces its value with new_value.
    """
    with open(CONFIG_FILE, 'r') as f:
        lines = f.readlines()

    updated_lines = []
    found = False
    
    # We create the new assignment line. Floats/Ints are fine as-is.
    # If the variable were a string, you would need to add quotes:
    # new_line = f'{var_name} = "{new_value}"\n'
    new_line = f'{var_name} = {new_value}\n'
    
    # Regex to find lines like 'VAR_NAME = 123'
    # It accounts for optional whitespace and ensures the whole word is matched.
    pattern = re.compile(rf'^\s*{var_name}\s*=\s*.*$', re.IGNORECASE)

    for line in lines:
        if pattern.match(line):
            updated_lines.append(new_line)
            found = True
            print(f"  > Updated {var_name} to {new_value}")
        else:
            updated_lines.append(line)
            
    if not found:
        raise ValueError(f"Variable '{var_name}' not found in {CONFIG_FILE}")

    with open(CONFIG_FILE, 'w') as f:
        f.writelines(updated_lines)

def run_experiment(config_updates):
    print(f"Applying new config:")
    for var_name, new_value in config_updates:
        update_py_config(var_name, new_value)

    try:
        result = subprocess.run(
            ["python", EXECUTABLE_SCRIPT],
            capture_output=True,
            text=True,
            check=True  # Raise an exception for non-zero return codes
        )
        
        captured_variable = result.stdout.strip().split('\n')[-1]
        print(f"Captured variable: {captured_variable}")
        return captured_variable

    except subprocess.CalledProcessError as e:
        print(f"Error running script: {EXECUTABLE_SCRIPT}")
        print(f"Stdout: {e.stdout}")
        print(f"Stderr: {e.stderr}")
        return "RUN_FAILED"
    except FileNotFoundError:
        print(f"Error: Could not find executable script {EXECUTABLE_SCRIPT}")
        return "SCRIPT_NOT_FOUND"


def main():
    
    param_names = [p[0] for p in PARAMETERS_TO_CHANGE]
    csv_header = param_names + ["collected_variable"]
    results_to_write = []
    
    param1_name, param1_values = PARAMETERS_TO_CHANGE[0]
    param2_name, param2_values = PARAMETERS_TO_CHANGE[1]
    
    run_count = 0
    
    for val1 in param1_values:
        for val2 in param2_values:
            run_count += 1
            print(f"\n--- Running Experiment {run_count} ---")
            
            current_config_updates = [
                (param1_name, val1),
                (param2_name, val2),
            ]
            
            collected_var = run_experiment(current_config_updates)
            
            results_row = [val1, val2, collected_var]
            results_to_write.append(results_row)

    print(f"\n--- Writing Results to {RESULTS_CSV_FILE} ---")
    with open(RESULTS_CSV_FILE, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(csv_header) # Write header
        csv_writer.writerows(results_to_write) # Write data rows

    print(f"All {run_count} experiments complete. Data saved to {RESULTS_CSV_FILE}.")

if __name__ == "__main__":
    main()