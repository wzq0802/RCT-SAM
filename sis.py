import os
import numpy as np
import pandas as pd
import torch
from TorchSisso import SissoModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# 添加中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
# 固定随机种子
seed = 82
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

device = "cpu"  # "cuda" if torch.cuda.is_available() else "cpu"
print(f"🖥️ 使用设备: {device}")

# ===== 绘图函数 =====
def plot_predictions(y_true, y_pred, title="Prediction vs True", out_file=None):
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, c='blue', alpha=0.6, edgecolors='k')
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel("实际值", fontsize=12)
    plt.ylabel("预测值", fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    if out_file:
        plt.savefig(out_file, dpi=300)
        print(f"✅ 图像已保存到 {out_file}")
    plt.show()

# ===== 安全拟合函数 =====
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
        print(f"❌ SISSO 拟合错误: {e}")
        import traceback
        traceback.print_exc()
        return 0.1, "f1", 0.5, "f1"

# ===== 安全预测函数 =====
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
    excel_file = "new/Bulk.xlsx"  # 请替换为实际路径

    try:
        # ===== 加载数据 =====
        df = pd.read_excel(excel_file)
        print(f"✅ 数据加载成功, 形状: {df.shape}")

        # 第一列为目标变量，其余列为输入特征
        y = df.iloc[:, 0].values.astype(np.float64)
        X = df.iloc[:, 1:].values.astype(np.float64)
        feature_names = df.columns[1:].tolist()

        # 预处理特征名：将连字符等非法字符替换为下划线
        safe_feature_names = [name.replace('-', '_') for name in feature_names]

        # 更新 DataFrame 列名
        df.columns = [df.columns[0]] + safe_feature_names

        # 使用安全的特征名
        feature_names = safe_feature_names

        if np.isnan(X).any() or np.isnan(y).any():
            raise ValueError("数据中存在 NaN 或 Inf，请先处理")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=seed
        )
        print(f"📘 训练集: {X_train.shape[0]}，📗 测试集: {X_test.shape[0]}")

        df_train = pd.DataFrame(np.column_stack([y_train, X_train]),
                                columns=["y"] + feature_names)

        operators = ['+', '-', '*', '/']
        # operators = ['+', '-', '*', '/', 'exp', 'pow(1/2)', 'pow(1/3)', 'log', 'ln', '^-1', 'pow(2)', 'pow(3)',
        #                                           'exp(-1)']
        n_expansion = 2
        n_term = 3
        k = 20

        print("\n🚀 开始训练 SISSO 模型...")
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
        print(f"\n✅ 训练完成！")
        print(f"📐 最优方程: {equation}")
        print(f"训练集性能: RMSE={train_rmse:.4f}, R²={train_r2:.4f}")

        print("\n🔍 测试集验证中...")
        test_rmse, test_r2, y_test_pred = safe_evaluate_equation(equation, X_test, y_test, feature_names)
        print(f"测试集性能: RMSE={test_rmse:.4f}, R²={test_r2:.4f}")

        r2_gap = train_r2 - test_r2
        print(f"\n📊 R² 差距: {r2_gap:.4f}")
        if abs(r2_gap) < 0.05:
            print("🎯 模型泛化能力优秀")
        elif r2_gap > 0.1:
            print("⚠️ 可能过拟合")
        else:
            print("✅ 模型性能稳定")

        train_pred = np.array([
            eval(equation, {"__builtins__": {},
                             **{name: float(X_train[i, j]) for j, name in enumerate(feature_names)},
                             "sqrt": np.sqrt, "ln": np.log, "log": np.log,
                             "exp": np.exp, "abs": np.abs, "pow": pow})
            for i in range(len(y_train))
        ])
        plot_predictions(y_train, train_pred, title="训练集预测", out_file="train_scatter.png")

        plot_predictions(y_test, y_test_pred, title="测试集预测", out_file="test_scatter.png")


    except FileNotFoundError:
        print(f"❌ 找不到文件 {excel_file}，请确认路径")
    except Exception as e:
        print(f"❌ 程序错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
