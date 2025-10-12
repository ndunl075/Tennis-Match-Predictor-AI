import pandas as pd
import joblib
import numpy as np

# --- Main function to run the prediction ---
def predict_winner(p1_name, p2_name, surface, model, stats_df):
    try:
        # Look up average stats for both players on the given surface
        p1_stats = stats_df[(stats_df['player'] == p1_name) & (stats_df['surface'] == surface)].iloc[0]
        p2_stats = stats_df[(stats_df['player'] == p2_name) & (stats_df['surface'] == surface)].iloc[0]
    except IndexError:
        # Handle cases where a player is not found or has no data on that surface
        print("Error: One or both players not found, or no match data available on this surface.")
        return

    # Calculate the difference in their average historical stats
    ace_diff = p1_stats['aces'] - p2_stats['aces']
    df_diff = p1_stats['dfs'] - p2_stats['dfs']
    serve_pts_diff = p1_stats['serve_pts'] - p2_stats['serve_pts']
    first_serve_in_diff = p1_stats['first_in'] - p2_stats['first_in']
    
    # Prepare the input for the model
    surface_hard = 1 if surface == 'Hard' else 0
    surface_grass = 1 if surface == 'Grass' else 0
    
    # The input must be in the exact same order as the training features
    match_features = np.array([[
        ace_diff, df_diff, serve_pts_diff, first_serve_in_diff, surface_hard, surface_grass
    ]])
    
    # Get the prediction probability from the model
    win_probability_p1 = model.predict_proba(match_features)[0][1] * 100
    
    # Display the result
    print("\n--------------------------")
    print(f"📊 Prediction for {p1_name} vs. {p2_name} on {surface}:")
    if win_probability_p1 > 50:
        print(f"🏆 Predicted Winner: {p1_name} ({win_probability_p1:.1f}% chance)")
    else:
        print(f"🏆 Predicted Winner: {p2_name} ({(100 - win_probability_p1):.1f}% chance)")
    print("--------------------------")

# --- Main part of the program ---
if __name__ == "__main__":
    try:
        # Load the trained model and the player stats file
        h2h_model = joblib.load('h2h_model.joblib')
        player_stats_df = pd.read_csv('player_avg_stats.csv')
        print("✅ AI model and player stats loaded successfully.")
    except FileNotFoundError:
        print("❌ Error: Model or stats file not found.")
        print("Please run the 'train_h2h_model.py' script first.")
        exit()

    while True:
        # Get user input
        player1 = input("\nEnter Player 1 Name (e.g., Novak Djokovic): ").strip()
        player2 = input("Enter Player 2 Name (e.g., Carlos Alcaraz): ").strip()
        court_surface = input("Enter Surface (Hard, Clay, or Grass): ").strip().title()

        predict_winner(player1, player2, court_surface, h2h_model, player_stats_df)
        
        again = input("\nMake another prediction? (yes/no): ").strip().lower()
        if again != 'yes':
            print("Goodbye!")
            break