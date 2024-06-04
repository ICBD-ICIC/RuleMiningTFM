import pandas as pd
from sklearn.model_selection import train_test_split
from mlxtend.frequent_patterns import fpmax, association_rules
import time
import sys

start_time = time.time()
profile_information = False

# Security check, execute only if 2 real arguments are received
if len(sys.argv) == 3:
    num_samples = int(sys.argv[1])
    num_features = int(sys.argv[2])
else:
    print("“Error. Enter arguments correctly”")
    sys.exit()
    
### 1. READ AND SPLIT THE DATA ###

# Load the cleaned DataFrame
if profile_information:
    boolean_data = pd.read_csv("Boolean_Data_with_profile.csv")
else:
    boolean_data = pd.read_csv("Boolean_Data.csv")

# Sample the data to reduce size
boolean_data = boolean_data.sample(n=num_samples, random_state=41)
boolean_data = boolean_data.sample(n=num_features, axis=1, random_state=41)

# Split the data into training and validation sets
train_data, validation_data = train_test_split(boolean_data, test_size=0.3, random_state=42)

### 2. SAVE THE TRAINING AND VALIDATION SETS ###

# Save the training set
if profile_information:
    train_data.to_csv("Train_Data_with_profile.csv", index=False)
else:
    train_data.to_csv("Train_Data.csv", index=False)

# Save the validation set
if profile_information:
    validation_data.to_csv("Validation_Data_with_profile.csv", index=False)
else:
    validation_data.to_csv("Validation_Data.csv", index=False)

### 3. RUN FPMAX ALGORITHM ON TRAINING SET ###

# Convert the training data to boolean
train_data = train_data.astype(bool)

# Define the FPMax algorithm parameters
supports = [0.1]
confidences = [0.3]

for minimum_support in supports:
    for minimum_confidence in confidences: 
        
        # Apply the FPMax algorithm to find maximal frequent itemsets in the training DataFrame
        maximal_itemsets = fpmax(train_data, min_support=minimum_support, use_colnames=True)

        # Generate association rules from the maximal itemsets
        rules = association_rules(maximal_itemsets, metric="confidence", min_threshold=minimum_confidence, support_only=True)

        # Filter rules to include only those with antecedents starting with 'inf_' and consequents starting with 'usr_'
        rules['antecedents_inf'] = rules['antecedents'].apply(lambda x: all(str(item).startswith('inf_') for item in x))
        rules['consequents_usr'] = rules['consequents'].apply(lambda x: all(str(item).startswith('usr_') for item in x))

        # Filter rules to exclude those with any 'usr_' in antecedents or any 'inf_' in consequents
        rules = rules[rules['antecedents_inf']]
        rules = rules[rules['consequents_usr']]

        # Drop unnecessary columns if they exist
        for col in ['antecedents_inf', 'consequents_usr']:
            if col in rules.columns:
                rules.drop(columns=[col], inplace=True)        

        # Drop unnecessary metrics
        # rules.drop(columns=['lift', 'leverage', 'conviction', 'zhangs_metric'], inplace=True)

        # Calculate the support of each rule
        rules['support'] = rules['support'] * len(train_data)

        # Sort the rules by confidence
        rules = rules.sort_values(by=['confidence'], ascending=False)

        # Construct file name based on parameters
        if profile_information:
            file_name = f"rules_support_{minimum_support}_confidence_{minimum_confidence}_with_profile.csv"
        else:
            file_name = f"rules_support_{minimum_support}_confidence_{minimum_confidence}.csv"

        # Save the DataFrame to a CSV file
        rules.to_csv(file_name, index=False)

end_time = time.time()
print("Execution time: {:.2f} seconds".format(end_time - start_time))
