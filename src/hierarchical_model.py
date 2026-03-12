import pandas as pd
import numpy as np
import pymc as pm
import arviz as az

df = pd.read_csv("data/feature_engineered_trials.csv")

df["genotype_id"] = df["genotype"].astype("category").cat.codes
df["region_id"] = df["region"].astype("category").cat.codes
df["location_id"] = df["location"].astype("category").cat.codes

n_genotypes = df["genotype_id"].nunique()
n_regions = df["region_id"].nunique()
n_locations = df["location_id"].nunique()

yield_obs = df["yield"].values
rainfall = df["rainfall"].values
temperature = df["temperature"].values
gdd = df["gdd"].values

genotype_idx = df["genotype_id"].values
region_idx = df["region_id"].values
location_idx = df["location_id"].values

with pm.Model() as model:

    intercept = pm.Normal("intercept", mu=0, sigma=5)

    beta_rain = pm.Normal("beta_rain", mu=0, sigma=1)
    beta_temp = pm.Normal("beta_temp", mu=0, sigma=1)
    beta_gdd = pm.Normal("beta_gdd", mu=0, sigma=1)

    sigma_genotype = pm.Exponential("sigma_genotype", 1)

    genotype_effect = pm.Normal(
        "genotype_effect",
        mu=0,
        sigma=sigma_genotype,
        shape=n_genotypes
    )

    sigma_region = pm.Exponential("sigma_region", 1)

    region_effect = pm.Normal(
        "region_effect",
        mu=0,
        sigma=sigma_region,
        shape=n_regions
    )

    sigma_location = pm.Exponential("sigma_location", 1)

    location_effect = pm.Normal(
        "location_effect",
        mu=0,
        sigma=sigma_location,
        shape=n_locations
    )

    mu = (
        intercept
        + beta_rain * rainfall
        + beta_temp * temperature
        + beta_gdd * gdd
        + genotype_effect[genotype_idx]
        + region_effect[region_idx]
        + location_effect[location_idx]
    )

    sigma = pm.Exponential("sigma", 1)

    yield_like = pm.Normal(
        "yield_like",
        mu=mu,
        sigma=sigma,
        observed=yield_obs
    )

    trace = pm.sample(
        1000,
        tune=1000,
        chains=2,
        target_accept=0.9
    )
    
summary = az.summary(trace)
print(summary)