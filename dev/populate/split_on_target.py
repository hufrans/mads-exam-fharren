import pandas as pd
import os

# Pad naar het inputbestand
input_file = 'data/heart_big_train.parq'  # Vervang dit met het pad naar jouw Parquet-bestand

# Lees het Parquet-bestand in
df = pd.read_parquet(input_file)

# Maak een map aan voor de output-bestanden
output_dir = 'data/split'
os.makedirs(output_dir, exist_ok=True)

# Splits op basis van de kolom 'target' en sla elk deel op
for target_value, group_df in df.groupby('target'):
    # Maak een veilige bestandsnaam
    safe_value = str(target_value).replace(' ', '_').replace('/', '_')
    output_file = os.path.join(output_dir, f'target_{safe_value}.parquet')
    group_df.to_parquet(output_file, index=False)

print(f"Bestanden opgeslagen in map: {output_dir}")
