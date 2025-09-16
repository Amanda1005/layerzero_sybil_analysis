import pandas as pd

# Read txt file, one address per line
with open("data/all_addresses_clean.txt", "r") as file:
    addresses = file.read().splitlines()

# Create a DataFrame and add source labels
df = pd.DataFrame({
    "address": addresses,
    "source": "bounty_report"
})

# Save as CSV
df.to_csv("data/sybil_addresses.csv", index=False)
print(f"✅ Export successful, a total of {len(df)} addresses have been saved as sybil_addresses.csv")
