import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

df = pd.read_csv("data/feature_engineered_trials.csv")

print("Dataset carregado:", df.shape)

yield_year = df.groupby("year")["yield"].mean()

print(yield_year)

yield_year.index = pd.to_datetime(yield_year.index, format="%Y")

plt.figure(figsize=(8,5))

plt.plot(yield_year)

plt.title("Average Yield Over Time")
plt.xlabel("Year")
plt.ylabel("Yield")

plt.show()

decomposition = seasonal_decompose(
    yield_year,
    model="additive"
)

decomposition.plot()

plt.show()

temp_year = df.groupby("year")["temperature"].mean()

temp_year.index = pd.to_datetime(temp_year.index, format="%Y")

plt.figure(figsize=(8,5))

plt.plot(temp_year)

plt.title("Average Temperature Over Time")

plt.show()

yearly_data = df.groupby("year")[[
    "yield",
    "temperature",
    "rainfall"
]].mean()

corr = yearly_data.corr()

print("\nCorrelação clima-produto:")
print(corr)

import seaborn as sns

sns.heatmap(corr, annot=True)

plt.title("Climate vs Yield Correlation")

plt.show()

yearly_data.to_csv("data/yearly_aggregated.csv")
