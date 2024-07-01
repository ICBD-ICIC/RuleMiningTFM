import pandas as pd
import time

# Start time to measure the duration of the script
start_time = time.time()

"""
Validation Script for Association Rules.

This script evaluates the performance of association rules on a validation dataset. It calculates the accuracy of each rule as well as the average accuracy.

Parameters to be adjusted:
    - `validation_data_path`: Path to the validation data CSV file.
    - `rules_file_path`: Path to the rules CSV file.
    - `evaluation_results_path`: Path to the CSV file where evaluation results will be saved.

Note:
    This script does not use the `profile_information` parameter. The user decides which rules file to validate.
    
Outputs:
    - A CSV file containing the evaluation results, including accuracy for each rule and average accuracy.
"""

# Parameters to be adjusted
validation_data_path = "path/to/Validation_Data_with_profile.csv"
rules_file_path = "path/to/rules_apriori_1000x30_with_profile.csv"
evaluation_results_path = "path/to/Evaluation_Results.csv"

# Load the validation set
validation_data = pd.read_csv(validation_data_path)

# Define a function to evaluate rules
def evaluate_rules(rules, data):
    rule_accuracies = []

    for _, rule in rules.iterrows():
        antecedents = frozenset(rule['antecedents'].strip('frozenset({})').replace('\'', '').split(', '))
        consequents = frozenset(rule['consequents'].strip('frozenset({})').replace('\'', '').split(', '))

        # Check if antecedents are present in the data
        antecedents_mask = data.apply(lambda row: antecedents.issubset(row.index[row == 1]), axis=1)
        antecedents_data = data[antecedents_mask]

        # Calculate accuracy for the rule
        if len(antecedents_data) > 0:
            # Check if consequents are present in the filtered data
            consequents_mask = antecedents_data.apply(lambda row: consequents.issubset(row.index[row == 1]), axis=1)
            accuracy = consequents_mask.mean()
        else:
            accuracy = 0

        rule_accuracies.append({
            'Rule': f"{rule['antecedents']} -> {rule['consequents']}",
            'Accuracy': accuracy
        })

    return rule_accuracies

# Load the rules
rules = pd.read_csv(rules_file_path)

# Evaluate the rules
rule_accuracies = evaluate_rules(rules, validation_data)

# Calculate average accuracy
average_accuracy = pd.DataFrame(rule_accuracies)['Accuracy'].mean()

# Store evaluation results
results = [{
    'Average Accuracy': average_accuracy
}]
results_df = pd.DataFrame(results)
rule_accuracies_df = pd.DataFrame(rule_accuracies)
combined_results_df = pd.concat([results_df, rule_accuracies_df], axis=1)
combined_results_df.to_csv(evaluation_results_path, index=False)

end_time = time.time()
print("Execution time: {:.2f} seconds".format(end_time - start_time))
