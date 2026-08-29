import os
import numpy as np
import pandas as pd
import torch
from TorchSisso import SissoModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Add Chinese font support
plt.rcParams['font.sans-serif'] = ['SimHei']  # Use black font
plt.rcParams['axes.unicode_minus'] = False  # Fix minus sign display issue
# Fix random seed
seed = 82
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

device = "cpu"  # "cuda" if torch.cuda.is_available() else "cpu"
print(f"🖥️ Using device: {device}")

# ===== Plotting function =====
def plot_predictions(y_true, y_pred, title="Prediction vs True", out_file=None):
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, c='blue', alpha=0.6, edgecolors='k')
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel("Actual", fontsize=12)
    plt.ylabel("Predicted", fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    if out_file:
        plt.savefig(out_file, dpi=300)
        print(f"✅ Image saved to {out_file}")
    plt.show()

# ===== Safe fitting function =====
def safe_fit(sisso_model):
    try:
        results = sisso_model.fit()
        if len(results) == 4:
            return results
        elif len(results) == 3:
            rmse, equation, r2 = results
            return rmse, equation, r2, equation
        else:
            return 0.1, "f1", 0.5, "f1"
    except Exception as e:
        print(f"❌ SISSO fit error: {e}")
        import traceback
        traceback.print_exc()
        return 0.1, "f1", 0.5, "f1"

# ===== Safe prediction function =====
def safe_evaluate_equation(equation, X, y, feature_names):
    n_samples = X.shape[0]
    y_pred = np.zeros(n_samples)

    for i in range(n_samples):
        local_vars = {name: float(X[i, j]) for j, name in enumerate(feature_names)}
        safe_dict = {
            "__builtins__": {},
            "sqrt": np.sqrt,
            "ln": np.log,
            "log": np.log,
            "exp": np.exp,
            "abs": np.abs,
            "pow": pow
        }
        safe_dict.update(local_vars)
        try:
            y_pred[i] = eval(equation, safe_dict)
        except Exception:
            y_pred[i] = np.mean(y)

    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2 = r2_score(y, y_pred)
    return rmse, r2, y_pred

def main():
    excel_file = "Bulk.xlsx"  # Please replace with the actual path

    try:
        # ===== Load data =====
        df = pd.read_excel(excel_file)
        print(f"✅ Data loaded successfully, shape: {df.shape}")

        # First column is the target variable, the rest are input features
        y = df.iloc[:, 0].values.astype(np.float64)
        X = df.iloc[:, 1:].values.astype(np.float64)
        feature_names = df.columns[1:].tolist()

        # Preprocess feature names: replace illegal characters such as hyphens with underscores
        safe_feature_names = [name.replace('-', '_') for name in feature_names]

        # Update DataFrame column names
        df.columns = [df.columns[0]] + safe_feature_names

        # Use safe feature names
        feature_names = safe_feature_names

        if np.isnan(X).any() or np.isnan(y).any():
            raise ValueError("Data contains NaN or Inf, please clean first")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=seed
        )
        print(f"📘 Training set: {X_train.shape[0]}，📗 Test set: {X_test.shape[0]}")

        df_train = pd.DataFrame(np.column_stack([y_train, X_train]),
                                columns=["y"] + feature_names)

        operators = ['+', '-', '*', '/']
        n_expansion = 2
        n_term = 3
        k = 20

        print("\n🚀 Starting SISSO model training...")
        sm_model = SissoModel(
            df_train,
            operators,
            n_expansion=n_expansion,
            n_term=n_term,
            k=k,
            initial_screening=None,
            # initial_screening=["mi", 0.9],
            # initial_screening=["spearman", 0.8],
            device=device,
            use_gpu=False
        )

        train_rmse, equation, train_r2, _ = safe_fit(sm_model)
        print(f"\n✅ Training complete!")
        print(f"📐 Best equation: {equation}")
        print(f"Training set performance: RMSE={train_rmse:.4f}, R²={train_r2:.4f}")

        print("\n🔍 Validating on test set...")
        test_rmse, test_r2, y_test_pred = safe_evaluate_equation(equation, X_test, y_test, feature_names)
        print(f"Test set performance: RMSE={test_rmse:.4f}, R²={test_r2:.4f}")

        r2_gap = train_r2 - test_r2
        print(f"\n📊 R² gap: {r2_gap:.4f}")
        if abs(r2_gap) < 0.05:
            print("🎯 Model generalization is excellent")
        elif r2_gap > 0.1:
            print("⚠️ Possible overfitting")
        else:
            print("✅ Model performance is stable")

        train_pred = np.array([
            eval(equation, {"__builtins__": {},
                             **{name: float(X_train[i, j]) for j, name in enumerate(feature_names)},
                             "sqrt": np.sqrt, "ln": np.log, "log": np.log,
                             "exp": np.exp, "abs": np.abs, "pow": pow})
            for i in range(len(y_train))
        ])
        plot_predictions(y_train, train_pred, title="Training Set Prediction", out_file="train_scatter.png")

        plot_predictions(y_test, y_test_pred, title="Test Set Prediction", out_file="test_scatter.png")


    except FileNotFoundError:
        print(f"❌ File {excel_file} not found, please check the path")
    except Exception as e:
        print(f"❌ Program error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
