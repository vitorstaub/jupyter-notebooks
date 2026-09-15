# jupyter-notebooks

Study notebooks on data preprocessing and machine learning with scikit-learn, XGBoost and PyTorch.

## Files

| File | Description |
|---|---|
| `scikitlearn-basics.ipynb` | scikit-learn basics: KNN classification on Iris, linear regression on California Housing, a `StandardScaler` + KNN pipeline and classification metrics. |
| `missingdatahandling.ipynb` | Fills missing values in a small NumPy array with `KNNImputer`. |
| `outliershandling.ipynb` | Finds and removes outliers in a synthetic house-price dataset with the IQR rule, with boxplots before and after. |
| `derivative-plots.ipynb` | Plots position and velocity as a function and its derivative, and the cross-entropy loss with its tangent line. |
| `predict-titanic.ipynb` | Titanic survival prediction with a scikit-learn preprocessing pipeline and XGBoost. Includes the math background, confusion matrix and feature importance. |
| `predict-is-cat-or-not.ipynb` | Cat vs. dog classifier on the Oxford-IIIT Pet dataset, fine-tuning a pretrained ResNet-34 (PyTorch + timm) in two phases. Includes the math background and a prediction on a local image. |

## Running

Each notebook installs its dependencies in the first cell. Datasets are downloaded when the notebooks run.

`data/`, `models/`, `*.png` and `*.pkl` are not versioned (see `.gitignore`). `predict-titanic.ipynb` saves `titanic_model.pkl` when it runs, and `predict-is-cat-or-not.ipynb` expects a local image at `local_img_path`.
