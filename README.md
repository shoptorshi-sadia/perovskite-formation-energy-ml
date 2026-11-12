
# Composition-Based Machine Learning for DFT Formation-Energy Prediction in Perovskites

## Scientific Objective

This project investigates the use of machine learning to predict the DFT-derived heat of formation of perovskite materials using elemental composition-based descriptors.

The pipeline uses the Matbench perovskites dataset and compares Random Forest and XGBoost regression models to establish a baseline for composition-based formation-energy prediction.

## Dataset

- **Dataset:** `matbench_perovskites` (Matbench v0.1)
- **Source:** Matminer dataset library
- **Number of entries:** 18,928
- **Original data:** DFT-calculated perovskite crystal structures
- **Target variable:** `e_form`
- **Target definition:** Heat of formation of the entire five-atom perovskite unit cell
- **Target unit:** eV per five-atom cell (eV/cell)
- **DFT method:** RPBE-GGA

The dataset originates from the work of Castelli et al. (2012) and contains a combinatorially generated set of perovskite compositions. It is not restricted to oxide and halide perovskites.

The oxygen reference state in the dataset is based on oxygen's chemical potential in water vapor rather than molecular oxygen.

## Methodology

Due to computational resource limitations, this project uses composition-based descriptors rather than full crystallographic structural representations.

The workflow consists of:

1. Retrieving the Matbench perovskites dataset using Matminer.
2. Extracting chemical formulas from the available structures.
3. Parsing chemical compositions using Pymatgen.
4. Generating elemental fractional-composition descriptors.
5. Training and comparing a Dummy Regressor, Random Forest and XGBoost.
6. Evaluating model performance using a held-out test set and five-fold cross-validation.

## Model Evaluation

The models are evaluated using:

- Coefficient of determination (R²)
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)

### XGBoost Test-Set Performance

| Metric | Result |
|---|---:|
| R² | 0.313 |
| RMSE | 0.621 eV/cell |
| MAE | 0.462 eV/cell |

These results represent an exploratory composition-based prediction baseline and are not an official Matbench leaderboard score.

## Feature Importance and Interpretability

Elemental composition features are examined to identify which elements contribute most to the model's predictions.

Feature importance provides information about predictive associations within the trained model and should not be interpreted directly as evidence of physical causation.

## Limitations

- The model uses elemental composition rather than full crystal-structure descriptors.
- Site-specific chemical environments and crystallographic information are not explicitly represented.
- Composition-based predictions may not capture structural differences between materials with similar chemical compositions.
- Formation energy is not equivalent to energy above the convex hull and does not independently establish thermodynamic stability against competing phases.

## Reproducibility

### Installation

```bash
pip install -r requirements.txt
```

### Download the dataset

```bash
python fetch_dft_data.py
```

### Train and evaluate models

```bash
python train_model.py
```

### Generate validation plots

```bash
python plot_validation.py
```

## References

1. Castelli, I. E. et al. (2012). New cubic perovskites for one- and two-photon water splitting using the computational materials repository. *Energy & Environmental Science*.
   https://doi.org/10.1039/C2EE22341D

2. Matbench / Matminer dataset documentation:
   https://hackingmaterials.lbl.gov/matminer/dataset_summary.html
