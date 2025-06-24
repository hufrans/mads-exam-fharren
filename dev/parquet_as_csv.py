
import pandas as pd

# Vervang dit met het juiste pad naar je Parquet-bestand
parquet_bestand = '/home/azureuser/mads-exam-fharren/runs/20250623-183016_all_model_summary.parquet'

# Laad het Parquet-bestand
df = pd.read_parquet(parquet_bestand)

# Sla het op als CSV-bestand
csv_bestand = '/home/azureuser/mads-exam-fharren/runs/20250623-183016_all_model_summary.csv'
df.to_csv(csv_bestand, index=False)

print(f"Het bestand is opgeslagen als {csv_bestand}")

