import pandas as pd
import sys
import os
import time
from sklearn.model_selection import train_test_split
from mlxtend.frequent_patterns import apriori, association_rules

# Start time to measure the duration of the script
start_time = time.time()

"""
Apriori Implementation Script.

This script executes the Apriori algorithm on a training set to find frequent itemsets and then generates association rules. The results are recorded in an Excel file.

Usage:
    python training_Apriori.py <num_samples> <num_features> <minimum_support> <minimum_confidence>

Arguments:
    - num_samples: int - Number of samples to be used from the dataset.
    - num_features: int - Number of features to be used from the dataset.
    - minimum_support: float - Minimum support value for the Apriori algorithm.
    - minimum_confidence: float - Minimum confidence value for the association rules.

Parameters to be adjusted:
    - `profile_information`: Set to True if profile information is included, otherwise set to False.
    - `excel_file`: Path to the Excel file where results will be recorded.
    - `data_path_with_profile`: Path to the Boolean data CSV file with profile information.
    - `data_path_without_profile`: Path to the Boolean data CSV file without profile information.
    - `output_file_with_profile`: Path to the output CSV file for association rules with profile data.
    - `output_file_without_profile`: Path to the output CSV file for association rules without profile data.
    - `train_data_path_with_profile`: Path to save the training data CSV file with profile information.
    - `train_data_path_without_profile`: Path to save the training data CSV file without profile information.
    - `validation_data_path_with_profile`: Path to save the validation data CSV file with profile information.
    - `validation_data_path_without_profile`: Path to save the validation data CSV file without profile information.

Outputs:
    - An Excel file recording the initial parameters, Apriori execution time, number of frequent itemsets, association rules execution time, and total execution time. The Excel file will be saved at the specified path.
    - A CSV file containing the generated association rules. The file name is constructed based on the input parameters and indicates whether profile information is included. The CSV file includes the following columns:
        - antecedents: The antecedent itemsets of the rule.
        - consequents: The consequent itemsets of the rule.
        - support: The support value of the rule.
        - confidence: The confidence value of the rule.
"""

# Parameters to be adjusted
data_path_with_profile = "path/to/Boolean_Data_with_profile.csv"
data_path_without_profile = "path/to/Boolean_Data.csv"
excel_file = "path/to/Apriori_results.xlsx"
output_file_with_profile = "path/to/rules_apriori_{num_samples}x{num_features}_with_profile.csv"
output_file_without_profile = "path/to/rules_apriori_{num_samples}x{num_features}.csv"
train_data_path_with_profile = "path/to/Train_Data_with_profile.csv"
train_data_path_without_profile = "path/to/Train_Data.csv"
validation_data_path_with_profile = "path/to/Validation_Data_with_profile.csv"
validation_data_path_without_profile = "path/to/Validation_Data.csv"
profile_information = False

# Verify parameters
if len(sys.argv) != 5:
    print("Error. Enter arguments correctly")
    sys.exit()

num_samples = int(sys.argv[1])
num_features = int(sys.argv[2])
minimum_support = float(sys.argv[3])
minimum_confidence = float(sys.argv[4])

print(f"{num_samples}, {num_features}, {minimum_support}, {minimum_confidence}")

# Start time to measure the duration of the script
start_time = time.time()

# Initial results data for Excel, assuming execution has not yet completed
initial_results = {
    "Num Samples": [num_samples],
    "Num Features": [num_features],
    "Minimum Support": [minimum_support],
    "Minimum Confidence": [minimum_confidence],
    "Execution Completed": ["No"],
}

initial_df = pd.DataFrame(initial_results)

# Write initial data to Excel
if os.path.exists(excel_file):
    old_df = pd.read_excel(excel_file)
    updated_df = pd.concat([old_df, initial_df], ignore_index=True)
else:
    updated_df = initial_df
updated_df.to_excel(excel_file, index=False)

print("Initial parameters recorded in Excel.")

# Continue with the rest of the script
data_path = data_path_with_profile if profile_information else data_path_without_profile
boolean_data = pd.read_csv(data_path)
boolean_data = boolean_data.sample(n=num_samples, random_state=41).sample(n=num_features, axis=1, random_state=41)
train_data, validation_data = train_test_split(boolean_data, test_size=0.3, random_state=42)

# Save the training and validation sets
if profile_information:
    train_data.to_csv(train_data_path_with_profile, index=False)
    validation_data.to_csv(validation_data_path_with_profile, index=False)
else:
    train_data.to_csv(train_data_path_without_profile, index=False)
    validation_data.to_csv(validation_data_path_without_profile, index=False)

train_data = train_data.astype(bool)

# Apriori Algorithm
apriori_start = time.time()
frequent_itemsets = apriori(train_data, min_support=minimum_support, use_colnames=True)
apriori_end = time.time()
apriori_time = apriori_end - apriori_start
num_frequent_itemsets = len(frequent_itemsets)

print("Apriori completed.")

# Update the Excel file with Apriori results before starting association rules
if os.path.exists(excel_file):
    existing_df = pd.read_excel(excel_file)
    last_row_index = existing_df.index[-1]
    existing_df.loc[last_row_index, "Apriori Execution Time (s)"] = apriori_time
    existing_df.loc[last_row_index, "Number of Frequent Itemsets"] = num_frequent_itemsets
    existing_df.loc[last_row_index, "Execution Completed"] = "No"
    existing_df.to_excel(excel_file, index=False)

# Association rules
association_start = time.time()
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=minimum_confidence)

# Filter rules to include only those with antecedents starting with 'inf_' and consequents starting with 'usr_'
rules['antecedents_inf'] = rules['antecedents'].apply(lambda x: all(str(item).startswith('inf_') for item in x))
rules['consequents_usr'] = rules['consequents'].apply(lambda x: all(str(item).startswith('usr_') for item in x))

# Filter rules to exclude those with any 'usr_' in antecedents or any 'inf_' in consequents
rules = rules[rules['antecedents_inf']]
rules = rules[rules['consequents_usr']]

# Drop unnecessary columns
rules.drop(columns=['antecedents_inf', 'consequents_usr'], inplace=True)

# Drop unnecessary metrics
rules.drop(columns=['lift', 'leverage', 'conviction', 'zhangs_metric'], inplace=True)

# Calculate the support of each rule
rules['support'] = rules['support'] * len(train_data)

# Sort the rules by confidence
rules = rules.sort_values(by=['confidence'], ascending=False)

# Construct file name based on parameters
file_name = output_file_with_profile if profile_information else output_file_without_profile
file_name = file_name.format(num_samples=num_samples, num_features=num_features)

# Save the DataFrame to a CSV file
rules.to_csv(file_name, index=False)

association_end = time.time()
association_time = association_end - association_start

print("Association rules generated.")

# Total execution time
end_time = time.time()
total_time = end_time - start_time

# Final results for Excel, updating the initial data
if os.path.exists(excel_file):
    existing_df = pd.read_excel(excel_file)
    existing_df.loc[last_row_index, "Association Rules Time (s)"] = association_time
    existing_df.loc[last_row_index, "Total Execution Time (s)"] = total_time
    existing_df.loc[last_row_index, "Execution Completed"] = "Yes"
    existing_df.to_excel(excel_file, index=False)

print("All results recorded in Excel.\n")
