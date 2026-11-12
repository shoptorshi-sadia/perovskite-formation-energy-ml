
import pandas as pd
import numpy as np

from pymatgen.core import Composition

from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error
)

from sklearn.model_selection import (
    train_test_split,
    KFold,
    cross_validate
)

from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyRegressor
from xgboost import XGBRegressor


RANDOM_STATE = 42

print("1. Loading DFT dataset...")

df = pd.read_csv("perovskite_bulk_energy.csv")

required_columns = {"formula", "e_form"}

if not required_columns.issubset(df.columns):
    raise ValueError(
        f"Dataset must contain columns: {required_columns}"
    )

print(f" -> Original records: {len(df)}")


print("\n2. Extracting composition descriptors...")


def get_fractions(formula):
    if pd.isna(formula):
        return {}

    try:
        composition = Composition(str(formula))
        return (
            composition
            .fractional_composition
            .get_el_amt_dict()
        )
    except Exception:
        return {}


descriptors = df["formula"].apply(get_fractions)

invalid_formulas = sum(not d for d in descriptors)

X = pd.DataFrame(
    descriptors.tolist(),
    index=df.index
).fillna(0.0)

if X.empty or X.shape[1] == 0:
    raise ValueError("No valid elemental descriptors were extracted.")

# Keep feature ordering consistent
X = X.reindex(sorted(X.columns), axis=1)

y = pd.to_numeric(df["e_form"], errors="coerce")

# Remove invalid formulas and missing/non-finite targets
valid_rows = (
    (X.sum(axis=1) > 0)
    & y.notna()
    & np.isfinite(y)
)

dropped_rows = int((~valid_rows).sum())

X = X.loc[valid_rows].astype(float)
y = y.loc[valid_rows].astype(float)

if len(X) < 5:
    raise ValueError("Insufficient valid records for five-fold CV.")

if not np.isfinite(X.to_numpy()).all():
    raise ValueError("Non-finite values found in feature matrix.")

print(f" -> Valid records: {len(X)}")
print(f" -> Invalid formulas: {invalid_formulas}")
print(f" -> Total dropped records: {dropped_rows}")
print(f" -> Unique elemental features: {len(X.columns)}")


print("\n3. Preparing training and test sets...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=RANDOM_STATE
)

kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE
)

scoring = {
    "r2": "r2",
    "mae": "neg_mean_absolute_error",
    "rmse": "neg_root_mean_squared_error"
}


models = {
    "Baseline (Mean)": DummyRegressor(strategy="mean"),

    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        random_state=RANDOM_STATE,
        n_jobs=-1
    ),

    "XGBoost": XGBRegressor(
        n_estimators=100,
        objective="reg:squarederror",
        random_state=RANDOM_STATE,
        n_jobs=1
    )
}


print("\n4. Model evaluation")

print("=" * 65)

for name, model in models.items():

    cv_results = cross_validate(
        model,
        X_train,
        y_train,
        cv=kf,
        scoring=scoring
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    # Independent test-set evaluation
    test_r2 = r2_score(y_test, predictions)

    test_rmse = np.sqrt(
        mean_squared_error(y_test, predictions)
    )

    test_mae = mean_absolute_error(
        y_test,
        predictions
    )

    # Cross-validation results
    cv_r2 = cv_results["test_r2"]

    cv_mae = -cv_results["test_mae"]

    cv_rmse = -cv_results["test_rmse"]

    print(f"\n{name}")

    print(
        f"CV R2:   {cv_r2.mean():.3f} "
        f"(+/- {cv_r2.std():.3f})"
    )

    print(
        f"CV RMSE: {cv_rmse.mean():.3f} "
        f"(+/- {cv_rmse.std():.3f}) eV/cell"
    )

    print(
        f"CV MAE:  {cv_mae.mean():.3f} "
        f"(+/- {cv_mae.std():.3f}) eV/cell"
    )

    print(
        f"Test R2:   {test_r2:.3f}"
    )

    print(
        f"Test RMSE: {test_rmse:.3f} eV/cell"
    )

    print(
        f"Test MAE:  {test_mae:.3f} eV/cell"
    )

    print("-" * 65)

print("\nModel evaluation completed.")
