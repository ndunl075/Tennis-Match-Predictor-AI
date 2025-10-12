import joblib
import numpy as np

# --- Function to get user input and prepare it for the model ---
def get_user_input_and_predict(model):
    print("\n🎾 Enter Player Stats to Predict Match Outcome 🎾")
    
    # Get stats from the user
    aces = int(input("Enter number of Aces: "))
    double_faults = int(input("Enter number of Double Faults: "))
    first_serve_perc = float(input("Enter First Serve Percentage (e.g., 65.4): "))
    surface = input("Enter court surface (Hard, Clay, or Grass): ").strip().title()
    
    # Calculate the "smart stat" (feature) that the model was trained on
    ace_to_df_ratio = aces / (double_faults + 1)
    
    # Process the surface input into the numerical format the model understands
    surface_hard = 1 if surface == 'Hard' else 0
    surface_grass = 1 if surface == 'Grass' else 0
        
    # Put all the data into a single list, in the exact same order as the training features
    # Note the double brackets [[...]] because the model expects a 2D array (a list of matches)
    user_data = np.array([[
        aces, 
        double_faults, 
        first_serve_perc, 
        ace_to_df_ratio, 
        surface_hard, 
        surface_grass
    ]])
    
    # Use the loaded model to make a prediction
    prediction = model.predict(user_data)
    prediction_proba = model.predict_proba(user_data) # Gets win/loss probabilities
    
    # Display the result to the user
    print("\n--------------------------")
    print("🤖 Analyzing stats...")
    
    if prediction[0] == 1:
        confidence = prediction_proba[0][1] * 100
        print(f"🏆 Prediction: WIN (Confidence: {confidence:.1f}%)")
    else:
        confidence = prediction_proba[0][0] * 100
        print(f"😔 Prediction: LOSS (Confidence: {confidence:.1f}%)")
    print("--------------------------")

# --- Main part of the program ---
if __name__ == "__main__":
    MODEL_FILE = 'real_tennis_model.joblib'
    
    # Load the saved model from the file
    try:
        loaded_model = joblib.load(MODEL_FILE)
        print(f"✅ AI Model '{MODEL_FILE}' loaded successfully.")
    except FileNotFoundError:
        print(f"❌ Error: '{MODEL_FILE}' not found.")
        print("Please run the 'train_real_model.py' script first to create the model file.")
        exit()

    # Loop to allow for multiple predictions
    while True:
        get_user_input_and_predict(loaded_model)
        
        again = input("\nMake another prediction? (yes/no): ").strip().lower()
        if again != 'yes':
            print("Goodbye!")
            break