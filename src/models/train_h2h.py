import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import joblib
import glob
import os

# Get the project root directory (two levels up from this script)
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))

print("--- Starting Head-to-Head AI Model Training ---")

# Step 1: Load and Combine All Match Data
print("Step 1: Loading all ATP match data...")
data_dir = os.path.join(project_root, 'data', 'raw')
all_files = glob.glob(os.path.join(data_dir, 'atp_matches_*.csv'))
df = pd.concat((pd.read_csv(f) for f in all_files), ignore_index=True)
print(f"✅ Loaded {len(df)} matches.")

# Step 2: Clean and Prepare Data
print("Step 2: Cleaning and preparing data...")
cols_to_use = [
    'surface', 'winner_name', 'loser_name', 'w_ace', 'l_ace', 
    'w_df', 'l_df', 'w_1stIn', 'l_1stIn', 'w_svpt', 'l_svpt'
]
df_clean = df[cols_to_use].dropna()

# Step 3: Create "Difference" Features for Head-to-Head Training
print("Step 3: Engineering 'difference' features...")
# For each stat, calculate the difference: winner's stat - loser's stat
df_clean['ace_diff'] = df_clean['w_ace'] - df_clean['l_ace']
df_clean['df_diff'] = df_clean['w_df'] - df_clean['l_df']
df_clean['serve_pts_diff'] = df_clean['w_svpt'] - df_clean['l_svpt']
df_clean['first_serve_in_diff'] = df_clean['w_1stIn'] - df_clean['l_1stIn']

# Create a balanced dataset: one row for winner-loser, one for loser-winner
# This teaches the model what both winning and losing stat differences look like.
df_winner_first = df_clean.copy()
df_winner_first['outcome'] = 1 # Winner is player 1, so the outcome is a win

df_loser_first = df_clean.copy()
df_loser_first['outcome'] = 0 # Loser is player 1, so the outcome is a loss
# Invert the differences for the loser-first perspective
df_loser_first['ace_diff'] = -df_loser_first['ace_diff']
df_loser_first['df_diff'] = -df_loser_first['df_diff']
df_loser_first['serve_pts_diff'] = -df_loser_first['serve_pts_diff']
df_loser_first['first_serve_in_diff'] = -df_loser_first['first_serve_in_diff']

# Combine both perspectives
h2h_df = pd.concat([df_winner_first, df_loser_first], ignore_index=True)
h2h_df = pd.get_dummies(h2h_df, columns=['surface'], drop_first=True)

# Step 4: Pre-calculate and Save Player Average Stats
# This is a crucial step for our prediction program to work quickly.
print("Step 4: Calculating and saving average stats for all players...")
# Re-structure the original data to be player-focused
winners_df = df_clean[['surface', 'winner_name', 'w_ace', 'w_df', 'w_1stIn', 'w_svpt']].rename(columns={'winner_name': 'player', 'w_ace': 'aces', 'w_df': 'dfs', 'w_1stIn': 'first_in', 'w_svpt': 'serve_pts'})
losers_df = df_clean[['surface', 'loser_name', 'l_ace', 'l_df', 'l_1stIn', 'l_svpt']].rename(columns={'loser_name': 'player', 'l_ace': 'aces', 'l_df': 'dfs', 'l_1stIn': 'first_in', 'l_svpt': 'serve_pts'})
player_df = pd.concat([winners_df, losers_df])
# Group by player and surface to get their average stats
player_avg_stats = player_df.groupby(['player', 'surface']).mean().reset_index()
# Save to root directory
output_path = os.path.join(project_root, 'player_avg_stats.csv')
player_avg_stats.to_csv(output_path, index=False)

# Step 5: Train and Save the Head-to-Head Model
print("Step 5: Training and saving the new H2H model...")
features = ['ace_diff', 'df_diff', 'serve_pts_diff', 'first_serve_in_diff', 'surface_Hard', 'surface_Grass']
X = h2h_df[features]
y = h2h_df['outcome']

h2h_model = DecisionTreeClassifier(max_depth=5, random_state=42)
h2h_model.fit(X, y)

# Save to root directory
model_path = os.path.join(project_root, 'h2h_model.joblib')
joblib.dump(h2h_model, model_path)
print("\n🎉 SUCCESS! Head-to-head model and player stats are saved.")

