import pandas as pd
import numpy as np

df = pd.read_csv("data/validated_trials.csv")

print("Dataset carregado:", df.shape)

#GDD = temperatura média − temperatura base
BASE_TEMP = 10

df["gdd"] = df["temperature"] - BASE_TEMP

df["gdd"] = df["gdd"].clip(lower=0)

rain_mean = df.groupby("region")["rainfall"].transform("mean")

df["rainfall_anomaly"] = df["rainfall"] - rain_mean

## positivo → ano chuvoso
## negativo → seca

temp_mean = df.groupby("region")["temperature"].transform("mean")

df["temperature_anomaly"] = df["temperature"] - temp_mean

#Water Stress = temperatura / chuva
df["water_stress_index"] = df["temperature"] / df["rainfall"]

#Genotype × Environment interaction
df["gdd_genotype"] = df["gdd"] * df["genotype"].astype("category").cat.codes

df["rain_temp_interaction"] = df["rainfall"] * df["temperature"]

#Agregation by enviroment
df["environment"] = df["region"] + "_" + df["year"].astype(str)

#normalized productivity
env_mean = df.groupby("environment")["yield"].transform("mean")

df["yield_normalized"] = df["yield"] - env_mean

print(df.head())

df.to_csv("data/feature_engineered_trials.csv", index=False)

print("Dataset com features salvo.")