import pandas as pd
from sklearn.model_selection import train_test_split
from mlxtend.frequent_patterns import apriori, association_rules
import time

start_time = time.time()
profile_information = False


### 1. READ AND SPLIT THE DATA ###

# Load the cleaned DataFrame
if profile_information:
    boolean_data = pd.read_excel("Boolean_Data_with_profile.xlsx")
else:
    boolean_data = pd.read_excel("Boolean_Data.xlsx")

# Get the size of the original DataFrame
original_size = boolean_data.shape[0]
print("The size of the original dataset is:", original_size)

# Sample the data to reduce size
sample_size = 1000
boolean_data = boolean_data.sample(n=sample_size, random_state=42)

# Split the data into training and validation sets
train_data, validation_data = train_test_split(boolean_data, test_size=0.3, random_state=42)


### 2. SAVE THE TRAINING AND VALIDATION SETS ###

# Save the training set
if profile_information:
    train_data.to_excel("Train_Data_with_profile.xlsx", index=False)
else:
    train_data.to_excel("Train_Data.xlsx", index=False)

# Save the validation set
if profile_information:
    validation_data.to_excel("Validation_Data_with_profile.xlsx", index=False)
else:
    validation_data.to_excel("Validation_Data.xlsx", index=False)


### 3. RUN APRIORI ALGORITHM ON TRAINING SET ###

# Convert the training data to boolean
train_data = train_data.astype(bool)

# Define the Apriori algorithm parameters
supports = [0.1]
confidences = [0.3]

for minimum_support in supports:
    for minimum_confidence in confidences: 
        
        # Apply the Apriori algorithm to find frequent itemsets in the training DataFrame
        frequent_itemsets = apriori(train_data, min_support=minimum_support, use_colnames=True)

        # Generate association rules from the frequent itemsets
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
        if profile_information:
            file_name = f"rules_support_{minimum_support}_confidence_{minimum_confidence}_with_profile.csv"
        else:
            file_name = f"rules_support_{minimum_support}_confidence_{minimum_confidence}.csv"

        # Save the DataFrame to a CSV file
        rules.to_csv(file_name, index=False)

end_time = time.time()
print("Execution time: {:.2f} seconds".format(end_time - start_time))
