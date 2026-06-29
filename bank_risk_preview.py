from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "源代码" / "data" / "bankriskinfo.csv"
OUTPUT_DIR = BASE_DIR / "outputs"


def load_data() -> pd.DataFrame:
    data = pd.read_csv(DATA_PATH, encoding="utf-8")
    data = data.rename(columns={"Default1": "Default"})
    return data


def build_model_data(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    model_data = data.dropna(subset=["Default"]).copy()
    y = model_data["Default"].astype(int)
    x = model_data.drop(columns=["Default"])

    numeric_cols = x.select_dtypes(include=["number"]).columns
    category_cols = x.columns.difference(numeric_cols)

    x[numeric_cols] = x[numeric_cols].fillna(x[numeric_cols].median())
    x[category_cols] = x[category_cols].fillna("未知")
    x = pd.get_dummies(x, drop_first=False)
    return x, y


def save_default_distribution(data: pd.DataFrame) -> None:
    counts = data["Default"].value_counts().sort_index()
    labels = ["未违约", "违约"]

    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(labels, counts.values, color=["#4c78a8", "#f58518"])
    ax.set_title("银行客户违约分布")
    ax.set_xlabel("客户状态")
    ax.set_ylabel("客户数量")
    ax.bar_label(bars)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "default_distribution.png", dpi=160)
    plt.close(fig)


def save_top_coefficients(model: LogisticRegression, columns: pd.Index) -> None:
    weights = pd.Series(model.coef_[0], index=columns).abs().sort_values().tail(10)

    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(8, 5))
    weights.plot(kind="barh", ax=ax, color="#54a24b")
    ax.set_title("逻辑回归 Top 10 特征权重")
    ax.set_xlabel("系数绝对值")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "top_coefficients.png", dpi=160)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    data = load_data()
    print("银行客户信用风险评估 - 快速预览")
    print(f"数据规模: {data.shape[0]} 行, {data.shape[1]} 列")
    print("\n违约分布:")
    print(data["Default"].value_counts().sort_index().rename(index={0.0: "未违约", 1.0: "违约"}))

    x, y = build_model_data(data)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=33, stratify=y
    )

    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        solver="liblinear",
        random_state=33,
    )
    model.fit(x_train, y_train)
    y_score = model.predict_proba(x_test)[:, 1]
    auc = roc_auc_score(y_test, y_score)
    print(f"\n快速逻辑回归 AUC: {auc:.4f}")

    save_default_distribution(data)
    save_top_coefficients(model, x.columns)
    print(f"\n图表已保存到: {OUTPUT_DIR}")
    print("- default_distribution.png")
    print("- top_coefficients.png")


if __name__ == "__main__":
    main()
