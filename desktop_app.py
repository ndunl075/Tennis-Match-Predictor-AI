import tkinter as tk
from tkinter import ttk  # For better-looking widgets
import pandas as pd
import joblib
import numpy as np

# --- Load Model and Data ---
# This part is crucial and assumes the files are in the same folder.
try:
    h2h_model = joblib.load('h2h_model.joblib')
    player_stats_df = pd.read_csv('player_avg_stats.csv')
    PLAYER_NAMES = sorted(player_stats_df['player'].unique())
except FileNotFoundError:
    # This will show an error in the terminal if the files are missing.
    print("Error: Model or stats file not found! Run 'src/models/train_h2h.py' first.")
    exit()

# --- Prediction Function ---
# This function runs when the user clicks the 'Predict' button.
def predict():
    player1 = p1_combo.get()
    player2 = p2_combo.get()
    surface = surface_combo.get()

    if not player1 or not player2 or not surface:
        result_label.config(text="Please fill all fields.", foreground="orange")
        return
    if player1 == player2:
        result_label.config(text="Select two different players.", foreground="red")
        return

    try:
        # Look up stats for both players
        p1_stats = player_stats_df[(player_stats_df['player'] == player1) & (player_stats_df['surface'] == surface)].iloc[0]
        p2_stats = player_stats_df[(player_stats_df['player'] == player2) & (player_stats_df['surface'] == surface)].iloc[0]

        # Calculate the difference in their average stats
        ace_diff = p1_stats['aces'] - p2_stats['aces']
        df_diff = p1_stats['dfs'] - p2_stats['dfs']
        serve_pts_diff = p1_stats['serve_pts'] - p2_stats['serve_pts']
        first_serve_in_diff = p1_stats['first_in'] - p2_stats['first_in']
        
        # Prepare input for the model
        surface_hard = 1 if surface == 'Hard' else 0
        surface_grass = 1 if surface == 'Grass' else 0
        
        features = np.array([[ace_diff, df_diff, serve_pts_diff, first_serve_in_diff, surface_hard, surface_grass]])
        win_prob_p1 = h2h_model.predict_proba(features)[0][1]

        # Determine the winner and display the result
        if win_prob_p1 > 0.5:
            winner, prob = player1, win_prob_p1
        else:
            winner, prob = player2, 1 - win_prob_p1
            
        result_label.config(text=f"Predicted Winner: {winner} ({prob:.1%})", foreground="green")

    except IndexError:
        result_label.config(text="Not enough data for this matchup.", foreground="red")

# --- GUI Setup ---
# This part builds the actual window and its widgets.
root = tk.Tk()
root.title("Tennis Predictor")
root.geometry("400x250")

frame = ttk.Frame(root, padding="10")
frame.pack(fill=tk.BOTH, expand=True)

# Player 1 dropdown
ttk.Label(frame, text="Player 1:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
p1_combo = ttk.Combobox(frame, values=PLAYER_NAMES, width=30)
p1_combo.grid(row=0, column=1, padx=5, pady=5)

# Player 2 dropdown
ttk.Label(frame, text="Player 2:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
p2_combo = ttk.Combobox(frame, values=PLAYER_NAMES, width=30)
p2_combo.grid(row=1, column=1, padx=5, pady=5)

# Surface dropdown
ttk.Label(frame, text="Surface:").grid(row=2, column=0, padx=5, pady=5, sticky="w")
surface_combo = ttk.Combobox(frame, values=["Hard", "Clay", "Grass"], width=30)
surface_combo.grid(row=2, column=1, padx=5, pady=5)

# Predict Button
predict_button = ttk.Button(frame, text="Predict Winner", command=predict)
predict_button.grid(row=3, column=0, columnspan=2, pady=15)

# Result Label
result_label = ttk.Label(frame, text="", font=("Helvetica", 12))
result_label.grid(row=4, column=0, columnspan=2)

# Start the app's main event loop
root.mainloop()