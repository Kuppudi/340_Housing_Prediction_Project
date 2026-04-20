import pandas as pd
import os
base_path = os.path.dirname(os.path.abspath(__file__)) + os.sep

ames_df = pd.read_csv(base_path + "AmesHousing.csv")
king_df = pd.read_csv(base_path + "kc_house_data.csv")
housing_df = pd.read_csv(base_path + "Housing.csv")
world_df = pd.read_csv(base_path + "world_real_estate_data(147k).csv")

print("Ames shape:", ames_df.shape)
print("King County shape:", king_df.shape)
print("Housing.csv shape:", housing_df.shape)
print("World dataset shape:", world_df.shape)

print("\nAmes columns:")
print(ames_df.columns.tolist())

print("\nKing County columns:")
print(king_df.columns.tolist())

print("\nHousing.csv columns:")
print(housing_df.columns.tolist())

print("\nWorld dataset columns:")
print(world_df.columns.tolist())

# 1. Ames dataset
ames_clean = ames_df[[
    "SalePrice",
    "Gr Liv Area",
    "Bedroom AbvGr",
    "Full Bath",
    "Half Bath",
    "Year Built"]].copy()

# create bathrooms properly
ames_clean["bathrooms"] = ames_clean["Full Bath"] + 0.5 * ames_clean["Half Bath"]

# rename columns
ames_clean = ames_clean.rename(columns={
    "SalePrice": "price",
    "Gr Liv Area": "area",
    "Bedroom AbvGr": "bedrooms",
    "Year Built": "year_built"
})

# keep only needed columns
ames_clean = ames_clean[["price", "area", "bedrooms", "bathrooms", "year_built"]]


# 2. King County dataset
king_clean = king_df[[
    "price",
    "sqft_living",
    "bedrooms",
    "bathrooms",
    "yr_built"
]].copy()

king_clean = king_clean.rename(columns={
    "sqft_living": "area",
    "yr_built": "year_built"
})


# 3. Housing.csv dataset
housing_clean = housing_df[[
    "price",
    "area",
    "bedrooms",
    "bathrooms"
]].copy()

# add missing column
housing_clean["year_built"] = None


# 4. World dataset

# clean "120 m²" → 120
world_df["area_clean"] = world_df["apartment_total_area"].astype(str).str.replace(" m²", "", regex=False).str.replace(" ", "", regex=False)
world_df["area_clean"] = pd.to_numeric(world_df["area_clean"], errors="coerce")
print("\nWorld area sample after cleaning:")
print(world_df[["apartment_total_area", "area_clean"]].head(10))

world_clean = world_df[[
    "price_in_USD",
    "area_clean",
    "apartment_bedrooms",
    "apartment_bathrooms",
    "building_construction_year"
]].copy()

world_clean = world_clean.rename(columns={
    "price_in_USD": "price",
    "area_clean": "area",
    "apartment_bedrooms": "bedrooms",
    "apartment_bathrooms": "bathrooms",
    "building_construction_year": "year_built"
})


# Check results
print("Ames clean shape:", ames_clean.shape)
print("King clean shape:", king_clean.shape)
print("Housing clean shape:", housing_clean.shape)
print("World clean shape:", world_clean.shape)

print("\nAmes preview:")
print(ames_clean.head())

# Cleaning + Combining datasets

# 1. Drop missing values
ames_clean = ames_clean.dropna(subset=["price", "area", "bedrooms", "bathrooms"])
king_clean = king_clean.dropna(subset=["price", "area", "bedrooms", "bathrooms"])
housing_clean = housing_clean.dropna(subset=["price", "area", "bedrooms", "bathrooms"])
world_clean = world_clean.dropna(subset=["price", "area", "bedrooms", "bathrooms"])

# 2. Add dataset labels (VERY IMPORTANT for analysis later)
ames_clean["dataset"] = "Ames"
king_clean["dataset"] = "KingCounty"
housing_clean["dataset"] = "HousingCSV"
world_clean["dataset"] = "World"

# 3. Combine all datasets
combined_df = pd.concat(
    [ames_clean, king_clean, housing_clean, world_clean],
    ignore_index=True
)

# 4. Final checks
print("Final combined shape:", combined_df.shape)

print("\nDataset distribution:")
print(combined_df["dataset"].value_counts())

print("\nPreview:")
print(combined_df.head())

print("\nSummary statistics:")
print(combined_df[["price", "area", "bedrooms", "bathrooms", "year_built"]].describe())

print("\nMissing values in combined dataset:")
print(combined_df.isnull().sum())

# Remove unrealistic values
combined_df = combined_df[
    (combined_df["price"] > 10000) &          # remove extremely low prices
    (combined_df["area"] >= 10) &             # remove impossible small houses
    (combined_df["area"] <= 20000) &          # remove extremely large outliers
    (combined_df["bedrooms"] >= 0) &          # remove negative bedrooms
    (combined_df["bedrooms"] <= 15) &         # remove unrealistic high bedrooms
    (combined_df["bathrooms"] >= 0) &         # remove negative bathrooms
    (combined_df["bathrooms"] <= 15)          # remove unrealistic high bathrooms
].copy()


# Clean year_built separately
combined_df.loc[ ~combined_df["year_built"].between(1800, 2026),"year_built"] = pd.NA

# Check results
print("\nShape after cleaning:", combined_df.shape)

print("\nCleaned summary statistics:")
print(combined_df[["price", "area", "bedrooms", "bathrooms", "year_built"]].describe())

# Feature Engineering

# 1. Price per square foot
combined_df["price_per_sqft"] = combined_df["price"] / combined_df["area"]

# 2. Log transform of price (VERY IMPORTANT for ML models)
import numpy as np
combined_df["log_price"] = np.log1p(combined_df["price"])

# 3. House age (only where year exists)
current_year = 2026
combined_df["house_age"] = current_year - combined_df["year_built"]

# Check new features
print("\nNew feature preview:")
print(combined_df[["price", "area", "price_per_sqft", "log_price", "house_age"]].head())

print("\nSummary of new features:")
print(combined_df[["price_per_sqft", "log_price", "house_age"]].describe())

# Remove extreme price_per_sqft outliers
combined_df = combined_df[
    (combined_df["price_per_sqft"] >= 50) &
    (combined_df["price_per_sqft"] <= 10000)].copy()

print("\nShape after price_per_sqft cleaning:", combined_df.shape)

print("\nUpdated price_per_sqft stats:")
print(combined_df["price_per_sqft"].describe())

combined_df.to_csv(base_path + "cleaned_housing_data.csv", index=False)

print("\nCleaned dataset saved!")