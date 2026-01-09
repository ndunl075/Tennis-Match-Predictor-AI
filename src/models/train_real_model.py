import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import joblib
import glob # This library helps find files
import os

# Get the project root directory (two levels up from this script)
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))

print("--- Starting AI Model Training on Real ATP Data ---")

# Part 1: Load and Combine All Match Data
# This finds all CSV files in the folder that start with 'atp_matches_'
print("Step 1: Finding and loading all ATP match CSV files...")
data_dir = os.path.join(project_root, 'data', 'raw')
all_files = glob.glob(os.path.join(data_dir, 'atp_matches_*.csv'))
df_list = []
for filename in all_files:
    # We read each file and add it to our list
    df_list.append(pd.read_csv(filename))

# We combine all the individual yearly files into one massive DataFrame
df = pd.concat(df_list, ignore_index=True)
print(f"✅ Loaded {len(df)} matches from {len(all_files)} files.")
print("-" * 50)

# Part 2: Clean and Restructure the Data
print("Step 2: Cleaning the data and preparing it for the model...")
# We select only the columns we need and drop any rows with missing key stats
cols_to_use = [
    'surface', 'winner_name', 'loser_name', 'w_ace', 'l_ace', 
    'w_df', 'l_df', 'w_1stIn', 'l_1stIn', 'w_svpt', 'l_svpt'
]
df_clean = df[cols_to_use].dropna()

# This is the key step: we restructure the data.
# Instead of one row per match, we create one row per PLAYER per match.
# Create a DataFrame for all the winning performances
winners_df = df_clean[['surface', 'winner_name', 'w_ace', 'w_df', 'w_1stIn', 'w_svpt']].copy()
winners_df.columns = ['surface', 'player', 'aces', 'double_faults', 'first_serves_in', 'serve_points']
winners_df['result'] = 1 # 1 means the player won

# Create a DataFrame for all the losing performances
losers_df = df_clean[['surface', 'loser_name', 'l_ace', 'l_df', 'l_1stIn', 'l_svpt']].copy()
losers_df.columns = ['surface', 'player', 'aces', 'double_faults', 'first_serves_in', 'serve_points']
losers_df['result'] = 0 # 0 means the player lost

# Combine them into the final dataset for our model
model_df = pd.concat([winners_df, losers_df], ignore_index=True)
print("✅ Data cleaned and restructured.")
print("-" * 50)

# Part 3: Create the "Smart Stats" (Feature Engineering)
print("Step 3: Creating 'smart stats' to help the model learn...")
model_df['first_serve_percentage'] = (model_df['first_serves_in'] / model_df['serve_points']) * 100
model_df['ace_to_df_ratio'] = model_df['aces'] / (model_df['double_faults'] + 1)

# We need to handle cases where stats might be zero to avoid errors
model_df.replace([np.inf, -np.inf], 0, inplace=True)
model_df.fillna(0, inplace=True)

# Now, we prepare the final data for the model by converting text to numbers
model_df = pd.get_dummies(model_df, columns=['surface'], drop_first=True)
print("✅ 'Smart stats' created.")
print("-" * 50)

# Part 4: Train the Model
print("Step 4: Training the AI model on all the data...")
# Define the features (X) and the target (y)
features = [
    'aces', 'double_faults', 'first_serve_percentage', 
    'ace_to_df_ratio', 'surface_Hard', 'surface_Grass'
]
X = model_df[features]
y = model_df['result']

# We use the Decision Tree model that gave you the best score (82%)
final_model = DecisionTreeClassifier(max_depth=5, random_state=42) # Tweaked depth for potentially better results on real data

# Train the model on ALL the real data
final_model.fit(X, y)
print("✅ Model has been successfully trained.")
print("-" * 50)

# Part 5: Save the Trained Model
print("Step 5: Saving the final model to a file...")
# Save to root directory
model_path = os.path.join(project_root, 'real_tennis_model.joblib')
joblib.dump(final_model, model_path)
print("\n🎉 SUCCESS! Your AI is trained on real matches and saved as 'real_tennis_model.joblib'.")

