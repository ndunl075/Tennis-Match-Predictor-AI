import os
import requests

def fetch_tennis_data():
    """
    Automated data acquisition for ATP matches and rankings.
    Ensures the project remains lightweight by fetching data only when needed.
    """
    # Get the project root directory (two levels up from this script)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    
    base_url = "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/"
    
    # Define which categories of files we want to fetch
    file_categories = [
        "atp_matches_{year}.csv",
        "atp_rankings_{decade}s.csv",
        "atp_players.csv",
        "atp_rankings_current.csv"
    ]
    
    years = range(1968, 2026)
    decades = ["70", "80", "90", "00", "10", "20"]
    
    # Create the data directory if it doesn't exist
    target_dir = os.path.join(project_root, 'data', 'raw')
    os.makedirs(target_dir, exist_ok=True)

    def download(filename):
        save_path = os.path.join(target_dir, filename)
        if os.path.exists(save_path):
            return # Skip if already there
            
        print(f"Downloading {filename}...")
        try:
            r = requests.get(f"{base_url}{filename}", timeout=10)
            if r.status_code == 200:
                with open(save_path, 'wb') as f:
                    f.write(r.content)
            else:
                print(f"Failed: {filename}")
        except Exception as e:
            print(f"Error: {e}")

    # Download everything
    for year in years:
        download(f"atp_matches_{year}.csv")
    for decade in decades:
        download(f"atp_rankings_{decade}s.csv")
    download("atp_players.csv")
    download("atp_rankings_current.csv")

if __name__ == "__main__":
    fetch_tennis_data()