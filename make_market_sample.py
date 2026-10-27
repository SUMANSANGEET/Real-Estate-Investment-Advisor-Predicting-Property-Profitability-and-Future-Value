"""Build a small, clean data/market_sample.csv from the full raw dataset (memory-safe, chunked).
Usage:  python make_market_sample.py path/to/india_housing_prices.csv [rows]
"""
import sys, os, csv
import pandas as pd

src = sys.argv[1]
rows = int(sys.argv[2]) if len(sys.argv) > 2 else 20000
keep = ["ID", "State", "City", "Property_Type", "BHK", "Size_in_SqFt", "Price_in_Lakhs", "PPS_INR", "Year_Built",
        "Furnished_Status", "Floor_No", "Total_Floors", "Nearby_Schools", "Nearby_Hospitals",
        "Public_Transport_Accessibility", "Facing", "Good_Investment"]
parts = []
for ch in pd.read_csv(src, chunksize=50_000, usecols=lambda c: c in keep, engine="python",
                      quoting=csv.QUOTE_NONE, on_bad_lines="skip"):
    parts.append(ch)
df = pd.concat(parts, ignore_index=True)
if "PPS_INR" not in df and {"Price_in_Lakhs", "Size_in_SqFt"} <= set(df):
    df["PPS_INR"] = df["Price_in_Lakhs"] * 1e5 / df["Size_in_SqFt"]
df = df.sample(min(rows, len(df)), random_state=1)
os.makedirs("data", exist_ok=True)
df.to_csv("data/market_sample.csv", index=False)
print("Wrote data/market_sample.csv", df.shape, f"{os.path.getsize('data/market_sample.csv')/1e6:.1f} MB")
