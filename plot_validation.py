
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pymatgen.core import Composition

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error
)

from xgboost import XGBRegressor


RANDOM_STATE = 42

print("1. Loading DFT dataset...")

df = pd.read_csv("perovskite_bulk_energy.csv")

required_columns = {"formula", "e_form"}

if not required_columns.issubset(df.columns):
    raise ValueError("Required columns are missing from the dataset.")


print("\n2. Extracting composition descriptors...")


def get_fractions(formula):
    if pd.isna(formula):
        return {}

    try:
        return (
            Composition(str(formula))
            .fractional_composition
            .get_el_amt_dict()
        )
    except Exception:
        return {}


descriptors = df["formula"].apply(get_fractions)

X = pd.DataFrame(
    descriptors.tolist(),
    index=df.index
).fillna(0.0)

X = X.reindex(sorted(X.columns), axis=1)

y = pd.to_numeric(df["e_form"], errors="coerce")

valid_rows = (
    (X.sum(axis=1) > 0)
    & y.notna()
    & np.isfinite(y)
)

X = X.loc[valid_rows].astype(float)
y = y.loc[valid_rows].astype(float)

if X.empty or X.shape[1] == 0:
    raise ValueError("No valid composition descriptors found.")

print(f" -> Valid records: {len(X)}")


print("\n3. Preparing training and test sets...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=RANDOM_STATE
)


print("\n4. Training XGBoost model...")

model = XGBRegressor(
    n_estimators=100,
    objective="reg:squarederror",
    random_state=RANDOM_STATE,
    n_jobs=1
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)


print("\n5. Calculating validation metrics...")

r2 = r2_score(y_test, y_pred)

rmse = np.sqrt(
    mean_squared_error(y_test, y_pred)
)

mae = mean_absolute_error(y_test, y_pred)

print(f"Test R2:   {r2:.3f}")
print(f"Test RMSE: {rmse:.3f} eV/cell")
print(f"Test MAE:  {mae:.3f} eV/cell")


print("\n6. Generating validation plots...")

fig, (ax1, ax2) = plt.subplots(
    1, 2, figsize=(13, 5)
)

# Parity plot
ax1.scatter(
    y_test,
    y_pred,
    alpha=0.4
)

limits = [
    min(y_test.min(), y_pred.min()),
    max(y_test.max(), y_pred.max())
]

ax1.plot(
    limits,
    limits,
    "r--",
    linewidth=2,
    label="Ideal prediction"
)

ax1.set_xlim(limits)
ax1.set_ylim(limits)
ax1.set_aspect("equal", adjustable="box")

ax1.set_xlabel("DFT Formation Energy (eV/cell)")
ax1.set_ylabel("Predicted Formation Energy (eV/cell)")
ax1.set_title("XGBoost: Predicted vs DFT")
ax1.legend()

ax1.text(
    0.05,
    0.95,
    f"R² = {r2:.3f}\nRMSE = {rmse:.3f} eV/cell\nMAE = {mae:.3f} eV/cell",
    transform=ax1.transAxes,
    verticalalignment="top"
)

# Residual plot
residuals = y_test - y_pred

ax2.hist(
    residuals,
    bins=30,
    edgecolor="black",
    alpha=0.8
)

ax2.axvline(
    0,
    linestyle="--",
    linewidth=2
)

ax2.set_xlabel("Residual Error (eV/cell)")
ax2.set_ylabel("Frequency")
ax2.set_title("Distribution of Prediction Residuals")

plt.tight_layout()

plt.savefig(
    "model_validation.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nSuccess! model_validation.png updated.")
