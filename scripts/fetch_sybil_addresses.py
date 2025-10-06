import pandas as pd
import os

# 建立輸出資料夾
os.makedirs("new_data", exist_ok=True)

# Read txt file, one address per line
with open("new_data/all_addresses_clean.txt", "r") as file:
    addresses = file.read().splitlines()

# 限制為前 5000 筆
addresses = addresses[:5000]

# Create a DataFrame and add source labels
df = pd.DataFrame({
    "address": addresses,
    "source": "bounty_report"
})

# Save to new_data
df.to_csv("new_data/sybil_addresses.csv", index=False)
print(f"✅ Exported {len(df)} addresses to new_data/sybil_addresses.csv")
