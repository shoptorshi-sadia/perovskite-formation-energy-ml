
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pymatgen.core import Composition

from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from xgboost import XGBRegressor


RANDOM_STATE = 42

print("1. Loading DFT dataset...")

df = pd.read_csv("perovskite_bulk_energy.csv")


print("2. Extracting composition descriptors...")


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
print(f" -> Elemental features: {len(X.columns)}")


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


print("\n5. Calculating permutation importance...")

result = permutation_importance(
    model,
    X_test,
    y_test,
    n_repeats=20,
    random_state=RANDOM_STATE,
    scoring="r2",
    n_jobs=1
)

importance = pd.Series(
    result.importances_mean,
    index=X.columns
)

importance_std = pd.Series(
    result.importances_std,
    index=X.columns
)

top_10 = importance.sort_values(ascending=False).head(10)
top_10 = top_10.sort_values(ascending=True)

print("\nTop 10 elemental features:")

for element, score in top_10.items():
    print(f" -> {element}: {score:.4f}")


print("\n6. Generating visualization...")

plt.figure(figsize=(9, 6))

plt.barh(
    top_10.index,
    top_10.values,
    xerr=importance_std[top_10.index],
    capsize=3
)

plt.axvline(0, linewidth=0.8, linestyle="--")

plt.title(
    "Permutation Importance of Elemental Composition Features"
)

plt.xlabel("Decrease in Test-Set R² After Permutation")
plt.ylabel("Elemental Fraction")

plt.tight_layout()

plt.savefig(
    "feature_importance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nSuccess! feature_importance.png updated.")
