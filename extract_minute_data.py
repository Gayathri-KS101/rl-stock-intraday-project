import os
import pandas as pd
from pathlib import Path

def extract_minute_data():
    archive_dir = Path("archive")
    processed_dir = Path("processed_data")

    # Ensure processed directory exists
    processed_dir.mkdir(exist_ok=True)

    # List all files in archive directory
    files = [f for f in os.listdir(archive_dir) if f.endswith('.xlsx')]
    
    print(f"Found {len(files)} Excel files in {archive_dir}")

    for file in files:
        file_path = archive_dir / file
        try:
            # Read the 'minute' sheet
            # using openpyxl engine is standard for xlsx
            df = pd.read_excel(file_path, sheet_name='minute', engine='openpyxl')
            
            # Construct output filename
            # e.g., ZOMATO.xlsx -> zomato-minute.csv
            output_filename = f"{file_path.stem.lower()}-minute.csv"
            output_path = processed_dir / output_filename
            
            # Save to CSV
            df.to_csv(output_path, index=False)
            print(f"Processed: {file} -> {output_filename}")
            
        except ValueError as e:
            if "Worksheet named 'minute' not found" in str(e):
                print(f"Skipping {file}: 'minute' sheet not found")
            else:
                print(f"Error processing {file}: {e}")
        except Exception as e:
             print(f"Error processing {file}: {e}")

if __name__ == "__main__":
    extract_minute_data()
