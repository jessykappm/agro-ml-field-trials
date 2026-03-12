import numpy as np
import pandas as pd

np.random.seed(42)

years = range(2018, 2025)

regions = {
    "MT": ["MT1","MT2"],
    "GO": ["GO1","GO2"],
    "PR": ["PR1","PR2"],
    "RS": ["RS1","RS2"],
    "BA": ["BA1","BA2"]
}

genotypes = [f"G{i}" for i in range(1,13)]

blocks = [1,2,3,4]

climate_profile = {
    "MT": (900,28),
    "GO": (850,27),
    "PR": (1200,22),
    "RS": (1300,20),
    "BA": (700,29)
}

rows = []

for year in years:

    for region, locations in regions.items():

        for loc in locations:

            rainfall_base, temp_base = climate_profile[region]

            rainfall = np.random.normal(rainfall_base,80)
            temperature = np.random.normal(temp_base,1)
            soil_ph = np.random.normal(6.2,0.3)

            for block in blocks:

                for genotype in genotypes:

                    genetic_effect = np.random.normal(0,0.4)

                    yield_value = (
                        3
                        + rainfall*0.002
                        - temperature*0.03
                        + soil_ph*0.4
                        + genetic_effect
                        + np.random.normal(0,0.2)
                    )

                    rows.append([
                        year,
                        region,
                        loc,
                        block,
                        genotype,
                        rainfall,
                        temperature,
                        soil_ph,
                        yield_value
                    ])

df = pd.DataFrame(rows, columns=[
    "year",
    "region",
    "location",
    "block",
    "genotype",
    "rainfall",
    "temperature",
    "soil_ph",
    "yield"
])

df.to_csv("data/synthetic_trials.csv", index=False)

print("Dataset criado:", df.shape)