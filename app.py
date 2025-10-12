import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Use caching to load the model and data only once
@st.cache_resource
def load_resources():
    """Load the trained model and player stats."""
    try:
        model = joblib.load('h2h_model.joblib')
        stats_df = pd.read_csv('player_avg_stats.csv')
        return model, stats_df
    except FileNotFoundError:
        return None, None

# Load the resources
h2h_model, player_stats_df = load_resources()

# --- Page Configuration ---
st.set_page_config(page_title="Tennis Match Predictor", page_icon="🎾", layout="centered")

# --- User Interface ---
st.title("🎾 Tennis Match Predictor AI")
st.write("Enter two player names and a surface to predict the winner based on historical data.")

if h2h_model is None or player_stats_df is None:
    st.error("Error: Model or stats file not found! Please run the 'train_h2h_model.py' script first.")
else:
    # Get a sorted list of unique player names for the dropdowns
    player_names = sorted(player_stats_df['player'].unique())
    
    col1, col2 = st.columns(2)
    with col1:
        player1 = st.selectbox("Select Player 1", player_names, index=None, placeholder="Choose a player...")
    with col2:
        player2 = st.selectbox("Select Player 2", player_names, index=None, placeholder="Choose a player...")

    surface = st.selectbox("Select Surface", ["Hard", "Clay", "Grass"], index=None, placeholder="Choose a surface...")

    # --- Prediction Logic ---
    if st.button("Predict Winner", type="primary"):
        if not player1 or not player2 or not surface:
            st.warning("Please select both players and a surface.")
        elif player1 == player2:
            st.error("Please select two different players.")
        else:
            try:
                # Look up stats for both players on the selected surface
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
                
                match_features = np.array([[ace_diff, df_diff, serve_pts_diff, first_serve_in_diff, surface_hard, surface_grass]])
                
                # Get prediction probability from the model
                win_probability_p1 = h2h_model.predict_proba(match_features)[0][1]

                st.subheader("Prediction Result:")
                if win_probability_p1 > 0.5:
                    st.success(f"🏆 Predicted Winner: {player1} ({win_probability_p1:.1%})")
                else:
                    st.success(f"🏆 Predicted Winner: {player2} ({(1 - win_probability_p1):.1%})")

            except IndexError:
                st.error(f"Prediction failed. Not enough data for one or both players on a {surface} court.")