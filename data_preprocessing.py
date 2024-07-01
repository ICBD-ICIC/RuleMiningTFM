import pandas as pd
import time

# Start time to measure the duration of the script
start_time = time.time()

"""
Data Preprocessing Script.

This script loads and preprocesses data from CSV files for influencers and users. It performs the following steps:
1. Loads influencer data and renames columns with an 'inf_' prefix.
2. Loads user data and renames columns with a 'usr_' prefix.
3. Optionally loads and aggregates profile data if `profile_information` is set to True.
4. Converts the 'emotions2' column in the user data to binary features.
5. Merges the dataframes based on common columns.
6. Removes columns with unique values, renames certain columns, and creates new features.
7. Saves the processed dataframe to a CSV file.

Parameters to be adjusted:
    - `influencers_file`: Path to the influencers CSV file.
    - `users_file`: Path to the users CSV file.
    - `profile_file`: Path to the profile CSV file (optional).
    - `output_file_with_profile`: Path to the output CSV file with profile data.
    - `output_file_without_profile`: Path to the output CSV file without profile data.
    - `profile_information` to include or exclude profile data processing.
    - `relevant_columns_influencers`: List of columns to load from the influencers CSV file.
    - `relevant_columns_users`: List of columns to load from the users CSV file.
    - `relevant_columns_user_profile`: List of columns to load from the user profile CSV file.
"""

# Parameters to be adjusted
influencers_file = "path/to/Influencers.csv"
users_file = "path/to/Users.csv"
profile_file = "path/to/Profile.csv"
output_file_with_profile = "path/to/All_Data_with_profile.csv"
output_file_without_profile = "path/to/All_Data.csv"
profile_information = False

relevant_columns_influencers = [
    'id', 'valence_score', 'num_moral_words',
    'num_polar_words', 'num_mfd_care_virtue',
    'num_mfd_care_vice', 'num_mfd_fairness_virtue',
    'num_mfd_fairness_vice', 'num_mfd_loyalty_virtue',
    'num_mfd_loyalty_vice', 'num_mfd_authority_virtue',
    'num_mfd_authority_vice', 'num_mfd_sanctity_virtue',
    'num_mfd_sanctity_vice'
]

relevant_columns_users = [
    'conversation_id', 'username',
    'valence_score', 'ethos',
    'abusive_words_ratio', 'num_moral_words',
    'num_polar_words', 'num_mfd_care_virtue',
    'num_mfd_care_vice', 'num_mfd_fairness_virtue',
    'num_mfd_fairness_vice', 'num_mfd_loyalty_virtue',
    'num_mfd_loyalty_vice', 'num_mfd_authority_virtue',
    'num_mfd_authority_vice', 'num_mfd_sanctity_virtue',
    'num_mfd_sanctity_vice', 'emotions2'
]

relevant_columns_user_profile = [
    'negative_words_ratio', 'positive_words_ratio',
    'moral_words_ratio', 'polar_words_ratio',
    'username'
]

### 1. LOAD ALL THE DATA ###

influencers_df = pd.read_csv(influencers_file, usecols=relevant_columns_influencers)

# Rename the columns by adding the 'inf_' prefix except for 'id'
influencers_df.columns = ['inf_' + col if col != 'id' else col for col in influencers_df.columns]

users_df = pd.read_csv(users_file, usecols=relevant_columns_users)

# Rename the columns by adding the 'usr_' prefix except for 'conversation_id' and 'username'
users_df.columns = ['usr_' + col if col not in ['conversation_id', 'username'] else col for col in users_df.columns]

# Load the DataFrame from "Profile.csv" if profile information is required
if profile_information:
    profile_df = pd.read_csv(profile_file, usecols=relevant_columns_user_profile)

### 2. EXTRACT AND CONVERT EMOTIONS2 COLUMN TO BINARY FEATURES ###

# Drop the row with index 125782
users_df = users_df.drop(index=125782)

# Extract all unique emotions from the 'emotions2' column
emotions = set()
users_df['usr_emotions2'].str.split().apply(emotions.update)

# Create a binary column for each unique emotion
for emotion in emotions:
    users_df['usr_emotion_' + emotion] = users_df['usr_emotions2'].str.contains(emotion).astype(int)

### 3. AGGREGATE PROFILE DATA ###

if profile_information:
    # Aggregate the 'Profile' DataFrame by 'username'
    profile_agg_df = profile_df.groupby('username').mean().reset_index()

### 4. MERGE DATAFRAMES ###

if profile_information:
    # Merge 'influencers_df' and 'users_df' on 'id' and 'conversation_id'
    merged_df = pd.merge(influencers_df, users_df, how='left', left_on='id', right_on='conversation_id')

    # Merge the result with 'profile_agg_df' using 'username' as the merging column, fill NaN values with 0
    all_data = pd.merge(merged_df, profile_agg_df, how='left', on='username').fillna(0)
else:
    all_data = pd.merge(influencers_df, users_df, how='left', left_on='id', right_on='conversation_id')

### 5. RENAME AND DROP COLUMNS ###

# Remove columns with unique values
unique_value_columns = all_data.columns[all_data.nunique() == 1]

if len(unique_value_columns) > 0:
    print(f"Columns removed due to having a unique value: {', '.join(unique_value_columns)}")
    all_data.drop(columns=unique_value_columns, inplace=True)

# Create new features based on 'usr_ethos'
all_data['usr_ethos_attack'] = (all_data['usr_ethos'] == 'attack').astype(int)
all_data['usr_ethos_neutral'] = (all_data['usr_ethos'] == 'neutral').astype(int)
all_data['usr_ethos_support'] = (all_data['usr_ethos'] == 'support').astype(int)

# Rename columns to remove 'num_' prefix
all_data.rename(columns=lambda x: x.replace("num_mfd_", ""), inplace=True)

# Rename columns to remove '_density' suffix
all_data.columns = [col.replace('_density', '') for col in all_data.columns]

if profile_information:
    # Rename the selected columns in profile_agg_df with 'profile_' prefix
    all_data.rename(columns={
        "negative_words_ratio": "inf_profile_negative_words_ratio",
        "positive_words_ratio": "inf_profile_positive_words_ratio",
        "moral_words_ratio": "inf_profile_moral_words_ratio",
        "polar_words_ratio": "inf_profile_polar_words_ratio"
    }, inplace=True)

    # Select only the rows where not all columns starting with 'profile' are zero
    all_data = all_data.loc[~(all_data.filter(regex='^inf_profile').eq(0).all(axis=1))]

# Drop columns
all_data.drop(columns=['id', 'username', 'conversation_id', 'usr_ethos', 'usr_emotions2'], inplace=True)

### 6. SAVE THE FINAL DATA ###

# Save the final DataFrame to a new CSV file
output_file = output_file_with_profile if profile_information else output_file_without_profile
all_data.to_csv(output_file, index=False)

end_time = time.time()
print("Execution time: {:.2f} seconds".format(end_time - start_time))