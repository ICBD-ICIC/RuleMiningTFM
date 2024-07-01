import pandas as pd
import time

# Start time to measure the duration of the script
start_time = time.time()

"""
Itemization Process Script.

This script loads data from a CSV file, preprocesses it by converting specific columns to boolean categories based on their mean values, and saves the resulting dataframe to a new CSV file.

Parameters to be adjusted:
    - `input_file_with_profile`: Path to the input data file with profile information.
    - `input_file_without_profile`: Path to the input data file without profile information.
    - `output_file_with_profile`: Path to the output CSV file with profile data.
    - `output_file_without_profile`: Path to the output CSV file without profile data.
    - `profile_information` to include or exclude profile data processing.
"""

# Parameters to be adjusted
input_file_with_profile = "path/to/All_Data_with_profile.csv"
input_file_without_profile = "path/to/All_Data.csv"
output_file_with_profile = "path/to/Boolean_Data_with_profile.csv"
output_file_without_profile = "path/to/Boolean_Data.csv"
profile_information = False

### 1. CONVERT TO BOOLEAN CATEGORIES ###

def convert_to_categories(df, column_name, labels, num_categories):
    """
    Convert a column to categorical based on mean value.
    """
    categories = None
    
    mean_value = df[column_name].mean()
    categories = df[column_name].apply(lambda x: labels[0][1] if x > mean_value else labels[0][0])

    df[f"{column_name}_category"] = categories

    for label in labels[num_categories - 2]:
        df[f"{column_name}_{label}"] = (categories == label).astype(int)

    df.drop(columns=[column_name], inplace=True)
    df.drop(columns=[f"{column_name}_category"], inplace=True)

    return df

# Read the DataFrame
input_file = input_file_with_profile if profile_information else input_file_without_profile
all_data = pd.read_csv(input_file)

# Remove rows with any NaN values
all_data = all_data.dropna()

# Define labels for each column
labels = [["low", "high"]]

# Apply the conversion for each column that is not already boolean
num_categories = 2

# List of columns to exclude
excluded_columns = ['inf_fairness_vice', 'inf_authority_vice', 'inf_sanctity_vice']

# Iterate over all columns and apply the condition
for column in all_data.columns:
    if column not in excluded_columns and all_data[column].dropna().isin([0, 1]).all():
        all_data[column] = all_data[column].astype(int)
        continue
    
    # Calculate and display the mean and variance of each column
    mean_value = all_data[column].mean()
    variance_value = all_data[column].var()
    # print(f"Column: {column} | Mean: {mean_value:.2f} | Variance: {variance_value:.2f}")
    
    all_data = convert_to_categories(all_data, column, labels, num_categories)
    
# Save the final DataFrame to a new CSV file
output_file = output_file_with_profile if profile_information else output_file_without_profile
all_data.to_csv(output_file, index=False)

end_time = time.time()
print("Execution time: {:.2f} seconds".format(end_time - start_time))
