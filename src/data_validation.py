import pandas as pd
import numpy as np

df = pd.read_csv("data/synthetic_trials.csv")

print("Dataset carregado:", df.shape)

missing = df.isnull().sum()

print("\nValores faltantes por coluna:")
print(missing)

rules = {
    "rainfall": (0, 3000),
    "temperature": (0, 50),
    "soil_ph": (4, 9),
    "yield": (0, 15)
}

for col, (min_val, max_val) in rules.items():

    invalid = df[(df[col] < min_val) | (df[col] > max_val)]

    print(f"\nValores inválidos em {col}: {len(invalid)}")
    
def detect_outliers(df, column):

    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)

    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outliers = df[(df[column] < lower) | (df[column] > upper)]

    return outliers

columns_to_check = ["yield","rainfall","temperature"]

for col in columns_to_check:

    outliers = detect_outliers(df, col)

    print(f"\nOutliers detectados em {col}: {len(outliers)}")
    
regional_stats = df.groupby("region")["temperature"].mean()

print("\nTemperatura média por região:")
print(regional_stats)

for region in df["region"].unique():

    mean_temp = df[df.region == region]["temperature"].mean()

    region_outliers = df[
        (df.region == region) &
        (abs(df.temperature - mean_temp) > 5)
    ]

    print(f"\nPossíveis inconsistências em {region}: {len(region_outliers)}")
    
report = {
    "total_rows": len(df),
    "missing_values": df.isnull().sum().sum()
}

print("\nRelatório de qualidade:")
print(report)

df.to_csv("data/validated_trials.csv", index=False)

print("\nDataset validado salvo.")