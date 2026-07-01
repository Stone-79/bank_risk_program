from __future__ import annotations

import argparse, hashlib, hmac, html, json, secrets, sqlite3
from datetime import datetime
from http import cookies
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "源代码" / "data" / "bankriskinfo.csv"
DB_PATH = BASE_DIR / "bank_risk_users.db"
SESSION_COOKIE = "bank_risk_session"
SESSIONS: dict[str, int] = {}

PROFILE_DEFAULTS = {
    "age": "35", "employmentYears": "5", "creditHistoryYears": "3",
    "idVerify": "一致", "threeVerify": "一致", "inCourt": "0", "isBlackList": "0",
}
ASSESSMENT_DEFAULTS = {
    "CityId": "二线城市", "education": "高中", "maritalStatus": "已婚", "sex": "男",
    "netLength": "24个月以上", "card_age": "5", "transTotalAmt": "5000",
    "transTotalCnt": "20", "onlineTransAmt": "1200", "cashTotalAmt": "0", "isDue": "0",
    "monthlyIncome": "10000", "monthlyExpense": "4500", "existingMonthlyRepayment": "1500",
    "requestedAmount": "50000", "loanTerm": "12", "overdueCount": "0", "creditCardUsage": "35",
}

RISK_POLICIES = {
    "A": ("A级，低风险", "可直接通过", 1.2, "自动初审 → 资料确认 → 放款", "预计 1-2 个工作日"),
    "B": ("B级，较低风险", "可放贷", 1.0, "自动初审 → 基础人工复核 → 放款", "预计 2-3 个工作日"),
    "C": ("C级，中等风险", "人工审核", 0.7, "系统初审 → 人工复核 → 补充资料 → 额度调整 → 放款", "预计 3-5 个工作日"),
    "D": ("D级，较高风险", "降低额度或提高利率", 0.4, "系统初审 → 严格人工审核 → 补充收入或资产证明 → 降低额度后重新评估", "预计 5-7 个工作日"),
    "E": ("E级，高风险", "建议拒贷", 0.0, "系统初审 → 高风险提示 → 暂不建议申请贷款", "建议先提高信用状况后再申请"),
}


def _onehot() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH, encoding="utf-8").rename(columns={"Default1": "Default"}).dropna(subset=["Default"]).copy()


MODEL_LABELS = {
    "gradient_boosting": "梯度提升模型（推荐）",
    "random_forest": "随机森林模型",
    "extra_trees": "ExtraTrees 模型",
    "adaboost": "AdaBoost 模型",
    "logistic_regression": "逻辑回归模型",
}
MODEL_NOTES = {
    "gradient_boosting": "\u9002\u5408\u7efc\u5408\u5224\u65ad\u591a\u6570\u5ba2\u6237\u60c5\u51b5",
    "random_forest": "\u9002\u5408\u6536\u5165\u3001\u652f\u51fa\u3001\u8d1f\u503a\u8f83\u590d\u6742\u7684\u60c5\u51b5",
    "extra_trees": "\u9002\u5408\u8d44\u6599\u4e0d\u591f\u7a33\u5b9a\u6216\u4fe1\u7528\u5386\u53f2\u8f83\u77ed\u7684\u60c5\u51b5",
    "adaboost": "\u9002\u5408\u8bc6\u522b\u8f7b\u5fae\u903e\u671f\u7b49\u98ce\u9669\u4fe1\u53f7",
    "logistic_regression": "\u9002\u5408\u4fe1\u7528\u72b6\u51b5\u7a33\u5b9a\u3001\u7ed3\u679c\u66f4\u6613\u89e3\u91ca\u7684\u60c5\u51b5",
}
DEFAULT_MODEL_KEY = "gradient_boosting"
AUTO_MODEL_KEY = "auto"


def _preprocessor(num_cols: list[str], cat_cols: list[str]) -> ColumnTransformer:
    return ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), num_cols),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", _onehot())]), cat_cols),
    ])


def train_models() -> tuple[dict[str, Pipeline], dict[str, Any], list[str], dict[str, float]]:
    data = load_data(); y = data["Default"].astype(int); x = data.drop(columns=["Default"])
    num_cols = x.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = [c for c in x.columns if c not in num_cols]
    defaults: dict[str, Any] = {c: float(x[c].median()) for c in num_cols}
    for c in cat_cols:
        mode = x[c].mode(dropna=True); defaults[c] = str(mode.iloc[0]) if not mode.empty else "\u672a\u77e5"
    classifiers = {
        "gradient_boosting": GradientBoostingClassifier(n_estimators=180, learning_rate=0.04, max_depth=3, random_state=33),
        "random_forest": RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=10, class_weight="balanced", random_state=33, n_jobs=-1),
        "extra_trees": ExtraTreesClassifier(n_estimators=300, max_depth=10, min_samples_leaf=8, class_weight="balanced", random_state=33, n_jobs=-1),
        "adaboost": AdaBoostClassifier(n_estimators=180, learning_rate=0.05, random_state=33),
        "logistic_regression": LogisticRegression(class_weight="balanced", solver="liblinear", max_iter=1000, random_state=33),
    }
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=33, stratify=y)
    models: dict[str, Pipeline] = {}; aucs: dict[str, float] = {}
    for key, classifier in classifiers.items():
        model = Pipeline([("preprocessor", _preprocessor(num_cols, cat_cols)), ("classifier", classifier)])
        model.fit(x_train, y_train)
        models[key] = model
        aucs[key] = float(roc_auc_score(y_test, model.predict_proba(x_test)[:, 1]))
    return models, defaults, x.columns.tolist(), aucs


MODELS, DEFAULTS, FEATURE_COLUMNS, MODEL_AUCS = train_models()

def fnum(v: Any, default: float = 0.0) -> float:
    try:
        return default if v in ("", None) else float(v)
    except (TypeError, ValueError):
        return default


def nonneg(v: Any, default: float = 0.0) -> float: return max(fnum(v, default), 0.0)
def posint(v: Any, default: int = 1) -> int: return max(int(fnum(v, default)), 1)
def pct(v: Any, default: float = 0.0) -> float: return min(max(fnum(v, default), 0.0), 100.0)
def clamp(p: float) -> float: return min(max(p, 0.01), 0.99)


def grade_from_probability(p: float) -> str:
    return "E" if p >= 0.80 else "D" if p >= 0.60 else "C" if p >= 0.40 else "B" if p >= 0.20 else "A"


def normalize_form(data: dict[str, Any]) -> dict[str, Any]:
    out = dict(data)
    for key in ["age", "employmentYears", "creditHistoryYears", "monthlyIncome", "monthlyExpense", "existingMonthlyRepayment", "requestedAmount", "overdueCount", "card_age", "transTotalAmt", "transTotalCnt", "onlineTransAmt", "cashTotalAmt"]:
        if key in out: out[key] = str(nonneg(out.get(key)))
    if "loanTerm" in out: out["loanTerm"] = str(posint(out.get("loanTerm"), 12))
    if "creditCardUsage" in out: out["creditCardUsage"] = str(pct(out.get("creditCardUsage")))
    return out


def apply_business_rules(base: float, d: dict[str, Any]) -> tuple[float, float]:
    adj = 0.0
    idv, three = str(d.get("idVerify", "")).strip(), str(d.get("threeVerify", "")).strip()
    if idv == "不一致": adj += 0.12
    elif idv == "未知": adj += 0.06
    if three == "不一致": adj += 0.10
    elif three == "未知": adj += 0.05
    if fnum(d.get("inCourt")) > 0: adj += 0.12
    if fnum(d.get("isBlackList")) > 0: adj += 0.20
    if fnum(d.get("isDue")) > 0: adj += 0.12
    od = nonneg(d.get("overdueCount"))
    if od >= 5: adj += 0.15
    elif od >= 3: adj += 0.10
    elif od > 0: adj += 0.05
    income, expense, debt = nonneg(d.get("monthlyIncome")), nonneg(d.get("monthlyExpense")), nonneg(d.get("existingMonthlyRepayment"))
    requested, term = nonneg(d.get("requestedAmount")), posint(d.get("loanTerm"), 12)
    disposable = income - expense - debt
    debt_ratio = debt / income if income > 0 else 0
    expense_ratio = expense / income if income > 0 else 0
    if income <= 0: adj += 0.12
    if disposable <= 0: adj += 0.15
    elif requested / term > disposable: adj += 0.10
    if debt_ratio > 0.6: adj += 0.12
    elif debt_ratio > 0.4: adj += 0.08
    if expense_ratio > 0.8: adj += 0.08
    elif expense_ratio > 0.65: adj += 0.04
    usage = pct(d.get("creditCardUsage"))
    if usage > 80: adj += 0.10
    elif usage > 50: adj += 0.06
    history = fnum(d.get("creditHistoryYears"))
    if history < 1: adj += 0.08
    elif history < 2: adj += 0.04
    elif history >= 5: adj -= 0.03
    work = fnum(d.get("employmentYears"))
    if work < 1: adj += 0.03
    elif work >= 5: adj -= 0.02
    if idv == "一致" and three == "一致" and od == 0: adj -= 0.03
    if income > 0 and debt_ratio < 0.25 and disposable > 0: adj -= 0.03
    return clamp(base + adj), adj

def loan_amount_advice(d: dict[str, Any], coef: float) -> dict[str, Any]:
    income, expense, debt = nonneg(d.get("monthlyIncome")), nonneg(d.get("monthlyExpense")), nonneg(d.get("existingMonthlyRepayment"))
    term, requested = posint(d.get("loanTerm"), 12), nonneg(d.get("requestedAmount"))
    disposable = income - expense - debt
    base_amount = max(disposable, 0) * term
    recommended = base_amount * coef
    return {
        "monthly_income": round(income, 2), "monthly_expense": round(expense, 2),
        "existing_monthly_repayment": round(debt, 2), "loan_term": term,
        "requested_amount": round(requested, 2), "disposable_income": round(disposable, 2),
        "base_amount": round(base_amount, 2), "risk_coefficient": coef,
        "recommended_amount": round(recommended, 2),
        "amount_comment": "当前申请金额较合理" if requested <= recommended else "申请金额可能偏高，建议降低申请额度",
    }


def improvement_advice(d: dict[str, Any], grade: str) -> list[str]:
    tips: list[str] = []
    od, usage = nonneg(d.get("overdueCount")), pct(d.get("creditCardUsage"))
    income, expense, debt = nonneg(d.get("monthlyIncome")), nonneg(d.get("monthlyExpense")), nonneg(d.get("existingMonthlyRepayment"))
    debt_ratio = debt / income if income > 0 else 0
    expense_ratio = expense / income if income > 0 else 0
    if od >= 3: tips.append("历史逾期次数较多，建议保持按时还款，减少逾期记录。")
    elif od > 0: tips.append("存在少量逾期记录，建议后续保持连续按时还款。")
    if usage > 50: tips.append("信用卡使用率较高，建议降低信用卡额度使用比例，控制在 50% 以下。")
    if debt_ratio > 0.4: tips.append("负债收入比较高，建议先减少已有负债，提高可支配收入。")
    if nonneg(d.get("creditHistoryYears")) < 2: tips.append("信用历史较短，建议保持稳定信用记录，延长信用积累时间。")
    if expense_ratio > 0.7: tips.append("月支出占收入比例较高，建议优化消费结构，提高还款能力。")
    if grade in {"D", "E"}: tips.append("当前风险等级偏高，建议暂缓大额贷款申请，先改善信用状况。")
    return tips or ["当前信用状况较稳定，建议继续保持良好还款习惯和合理负债水平。"]



def recommend_model(form_data: dict[str, Any]) -> tuple[str, str]:
    idv = str(form_data.get("idVerify", "")).strip()
    three = str(form_data.get("threeVerify", "")).strip()
    severe_identity = idv == "\u4e0d\u4e00\u81f4" or three == "\u4e0d\u4e00\u81f4"
    uncertain_identity = idv == "\u672a\u77e5" or three == "\u672a\u77e5"
    court = fnum(form_data.get("inCourt")) > 0
    blacklist = fnum(form_data.get("isBlackList")) > 0
    due = fnum(form_data.get("isDue")) > 0
    overdue = nonneg(form_data.get("overdueCount"))
    usage = pct(form_data.get("creditCardUsage"))
    income = nonneg(form_data.get("monthlyIncome"))
    expense = nonneg(form_data.get("monthlyExpense"))
    debt = nonneg(form_data.get("existingMonthlyRepayment"))
    requested = nonneg(form_data.get("requestedAmount"))
    term = posint(form_data.get("loanTerm"), 12)
    disposable = income - expense - debt
    debt_ratio = debt / income if income > 0 else 1
    expense_ratio = expense / income if income > 0 else 1
    monthly_request = requested / term
    history = nonneg(form_data.get("creditHistoryYears"))

    if blacklist or court or overdue >= 3 or severe_identity:
        return "gradient_boosting", "\u5ba2\u6237\u5b58\u5728\u9ed1\u540d\u5355\u3001\u6cd5\u9662\u8bb0\u5f55\u3001\u591a\u6b21\u903e\u671f\u6216\u9a8c\u8bc1\u4e0d\u4e00\u81f4\u7b49\u9ad8\u98ce\u9669\u7279\u5f81\uff0c\u4f18\u5148\u4f7f\u7528\u533a\u5206\u80fd\u529b\u66f4\u5f3a\u7684\u68af\u5ea6\u63d0\u5347\u6a21\u578b\u3002"
    if income <= 0 or disposable <= 0 or debt_ratio > 0.45 or expense_ratio > 0.75 or monthly_request > disposable:
        return "random_forest", "\u5ba2\u6237\u6536\u652f\u6216\u8d1f\u503a\u538b\u529b\u8f83\u9ad8\uff0c\u4f7f\u7528\u968f\u673a\u68ee\u6797\u6a21\u578b\u66f4\u9002\u5408\u5904\u7406\u591a\u4e2a\u8fd8\u6b3e\u80fd\u529b\u53d8\u91cf\u7684\u7ec4\u5408\u5f71\u54cd\u3002"
    if usage > 70 or history < 2 or uncertain_identity:
        return "extra_trees", "\u5ba2\u6237\u5b58\u5728\u4fe1\u7528\u5361\u4f7f\u7528\u7387\u504f\u9ad8\u3001\u4fe1\u7528\u5386\u53f2\u8f83\u77ed\u6216\u9a8c\u8bc1\u4fe1\u606f\u4e0d\u786e\u5b9a\uff0c\u4f7f\u7528 ExtraTrees \u6a21\u578b\u4fbf\u4e8e\u5bf9\u4e0d\u7a33\u5b9a\u7279\u5f81\u505a\u66f4\u7a33\u5065\u7684\u5224\u65ad\u3002"
    if due or overdue > 0 or usage > 50:
        return "adaboost", "\u5ba2\u6237\u5b58\u5728\u8f7b\u5fae\u903e\u671f\u6216\u4fe1\u7528\u5361\u4f7f\u7528\u7387\u504f\u9ad8\uff0c\u4f7f\u7528 AdaBoost \u6a21\u578b\u66f4\u9002\u5408\u5bf9\u8f7b\u5fae\u98ce\u9669\u4fe1\u53f7\u8fdb\u884c\u5f3a\u5316\u5224\u522b\u3002"
    if history >= 5 and debt_ratio < 0.25 and disposable > 0 and idv == "\u4e00\u81f4" and three == "\u4e00\u81f4":
        return "logistic_regression", "\u5ba2\u6237\u4fe1\u7528\u5386\u53f2\u8f83\u957f\u3001\u8d1f\u503a\u538b\u529b\u8f83\u4f4e\u4e14\u9a8c\u8bc1\u4fe1\u606f\u7a33\u5b9a\uff0c\u4f7f\u7528\u903b\u8f91\u56de\u5f52\u6a21\u578b\u53ef\u4ee5\u5f97\u5230\u66f4\u7a33\u5b9a\u3001\u6613\u89e3\u91ca\u7684\u57fa\u7840\u5224\u65ad\u3002"
    return "gradient_boosting", "\u5ba2\u6237\u4fe1\u606f\u5c5e\u4e8e\u4e00\u822c\u60c5\u51b5\uff0c\u9ed8\u8ba4\u4f7f\u7528\u7efc\u5408\u5224\u65ad\u80fd\u529b\u8f83\u5f3a\u7684\u68af\u5ea6\u63d0\u5347\u6a21\u578b\u3002"

def predict_risk(form_data: dict[str, Any]) -> dict[str, Any]:
    requested_model = str(form_data.get("modelName") or AUTO_MODEL_KEY)
    recommended_key, recommendation_reason = recommend_model(form_data)
    auto_selected = requested_model == AUTO_MODEL_KEY
    if auto_selected:
        model_key = recommended_key
    elif requested_model in MODELS:
        model_key = requested_model
    else:
        model_key = recommended_key
        auto_selected = True
    form_data = normalize_form(form_data)
    row = DEFAULTS.copy()
    for k, v in form_data.items():
        if k in row and v not in ("", None): row[k] = float(v) if isinstance(DEFAULTS[k], float) else str(v)
    base = float(MODELS[model_key].predict_proba(pd.DataFrame([row], columns=FEATURE_COLUMNS))[0, 1])
    prob, adj = apply_business_rules(base, form_data)
    grade = grade_from_probability(prob)
    level, suggestion, coef, process, expected_time = RISK_POLICIES[grade]
    amount = loan_amount_advice(form_data, coef)
    return {
        "probability": round(prob, 4), "base_probability": round(base, 4), "business_adjustment": round(adj, 4),
        "percentage": f"{prob * 100:.2f}%", "credit_score": round(100 - prob * 100, 2),
        "risk_grade": grade, "level": level, "suggestion": suggestion, "loan_amount": amount,
        "model_key": model_key, "model_label": MODEL_LABELS[model_key],
        "recommended_model_key": recommended_key, "recommended_model_label": MODEL_LABELS[recommended_key],
        "model_auto_selected": auto_selected, "model_recommendation_reason": recommendation_reason,
        "improvement_advice": improvement_advice(form_data, grade), "loan_process": process,
        "expected_time": expected_time, "auc": round(MODEL_AUCS[model_key], 4), "normalized_input": form_data,
    }


def conn() -> sqlite3.Connection:
    c = sqlite3.connect(DB_PATH); c.row_factory = sqlite3.Row; return c


def init_db() -> None:
    with conn() as c:
        c.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            created_time TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS user_profiles (
            user_id INTEGER PRIMARY KEY,
            age REAL,
            employment_years REAL,
            credit_history_years REAL,
            identity_verified TEXT,
            three_factor_verified TEXT,
            court_record INTEGER,
            blacklist INTEGER,
            updated_time TEXT NOT NULL,
            FOREIGN KEY(user_id) REFERENCES users(id)
        );
        CREATE TABLE IF NOT EXISTS assessment_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            monthly_income REAL,
            monthly_expense REAL,
            existing_monthly_debt REAL,
            requested_loan_amount REAL,
            loan_term INTEGER,
            default_probability REAL,
            credit_score REAL,
            risk_level TEXT,
            recommended_loan_amount REAL,
            created_time TEXT NOT NULL,
            FOREIGN KEY(user_id) REFERENCES users(id)
        );
        """)


def hash_password(password: str, salt_hex: str | None = None) -> str:
    salt = bytes.fromhex(salt_hex) if salt_hex else secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 120_000)
    return f"pbkdf2_sha256${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try: alg, salt, _ = stored.split("$", 2)
    except ValueError: return False
    return alg == "pbkdf2_sha256" and hmac.compare_digest(hash_password(password, salt), stored)


def create_user(username: str, password: str) -> tuple[bool, str]:
    username = username.strip()
    if len(username) < 3: return False, "用户名至少需要 3 个字符。"
    if len(password) < 6: return False, "密码至少需要 6 个字符。"
    now = datetime.now().isoformat(timespec="seconds")
    try:
        with conn() as c:
            cur = c.execute("INSERT INTO users(username,password_hash,created_time) VALUES (?,?,?)", (username, hash_password(password), now))
            c.execute("INSERT INTO user_profiles(user_id,age,employment_years,credit_history_years,identity_verified,three_factor_verified,court_record,blacklist,updated_time) VALUES (?,?,?,?,?,?,?,?,?)", (cur.lastrowid, 35, 5, 3, "一致", "一致", 0, 0, now))
        return True, "注册成功，请登录。"
    except sqlite3.IntegrityError:
        return False, "用户名已存在，请换一个。"


def authenticate(username: str, password: str) -> int | None:
    with conn() as c: row = c.execute("SELECT id,password_hash FROM users WHERE username=?", (username.strip(),)).fetchone()
    return int(row["id"]) if row and verify_password(password, row["password_hash"]) else None


def get_user(user_id: int | None) -> sqlite3.Row | None:
    if not user_id: return None
    with conn() as c: return c.execute("SELECT id,username FROM users WHERE id=?", (user_id,)).fetchone()


def get_profile(user_id: int) -> dict[str, str]:
    p = PROFILE_DEFAULTS.copy()
    with conn() as c: row = c.execute("SELECT * FROM user_profiles WHERE user_id=?", (user_id,)).fetchone()
    if row:
        p.update({
            "age": str(row["age"] if row["age"] is not None else p["age"]),
            "employmentYears": str(row["employment_years"] if row["employment_years"] is not None else p["employmentYears"]),
            "creditHistoryYears": str(row["credit_history_years"] if row["credit_history_years"] is not None else p["creditHistoryYears"]),
            "idVerify": row["identity_verified"] or p["idVerify"], "threeVerify": row["three_factor_verified"] or p["threeVerify"],
            "inCourt": str(row["court_record"] or 0), "isBlackList": str(row["blacklist"] or 0),
        })
    return p


def update_profile(user_id: int, data: dict[str, Any]) -> None:
    d = normalize_form(data); now = datetime.now().isoformat(timespec="seconds")
    with conn() as c:
        c.execute("""
        INSERT INTO user_profiles(user_id,age,employment_years,credit_history_years,identity_verified,three_factor_verified,court_record,blacklist,updated_time)
        VALUES (?,?,?,?,?,?,?,?,?)
        ON CONFLICT(user_id) DO UPDATE SET age=excluded.age, employment_years=excluded.employment_years,
        credit_history_years=excluded.credit_history_years, identity_verified=excluded.identity_verified,
        three_factor_verified=excluded.three_factor_verified, court_record=excluded.court_record,
        blacklist=excluded.blacklist, updated_time=excluded.updated_time
        """, (user_id, nonneg(d.get("age"), 35), nonneg(d.get("employmentYears"), 5), nonneg(d.get("creditHistoryYears"), 3), str(d.get("idVerify") or "未知"), str(d.get("threeVerify") or "未知"), int(fnum(d.get("inCourt"))), int(fnum(d.get("isBlackList"))), now))


def save_record(user_id: int, result: dict[str, Any]) -> None:
    a = result["loan_amount"]
    with conn() as c:
        c.execute("""
        INSERT INTO assessment_records(user_id,monthly_income,monthly_expense,existing_monthly_debt,requested_loan_amount,loan_term,default_probability,credit_score,risk_level,recommended_loan_amount,created_time)
        VALUES (?,?,?,?,?,?,?,?,?,?,?)
        """, (user_id, a["monthly_income"], a["monthly_expense"], a["existing_monthly_repayment"], a["requested_amount"], a["loan_term"], result["probability"], result["credit_score"], result["level"], a["recommended_amount"], datetime.now().isoformat(timespec="seconds")))


def history(user_id: int) -> list[sqlite3.Row]:
    with conn() as c:
        return c.execute("SELECT * FROM assessment_records WHERE user_id=? ORDER BY created_time DESC,id DESC LIMIT 30", (user_id,)).fetchall()

def esc(v: Any) -> str: return html.escape(str(v), quote=True)
def sel(cur: Any, exp: Any) -> str: return " selected" if str(cur) == str(exp) else ""
def money(v: Any) -> str: return f"¥{fnum(v):,.0f}"

APP_CSS = """
:root {
    --ink: #16202a;
    --muted: #667085;
    --line: #d9e1e8;
    --panel: rgba(255, 255, 255, .66);
    --soft: #f5f7fb;
    --navy: #111827;
    --blue: #2563eb;
    --sky: #0ea5e9;
    --indigo: #4f46e5;
    --pink: #ec4899;
    --shadow: 0 18px 42px rgba(15, 23, 42, .12);
}
* { box-sizing: border-box; }
body {
    margin: 0;
    min-height: 100vh;
    font-family: "Microsoft YaHei", "Segoe UI", Arial, sans-serif;
    color: var(--ink);
    background:
        radial-gradient(circle at 12% 12%, rgba(59, 130, 246, .18), transparent 28%),
        radial-gradient(circle at 88% 10%, rgba(14, 165, 233, .14), transparent 25%),
        linear-gradient(180deg, #eff6ff 0%, #dbeafe 46%, #f8fbff 100%);
}
body.auth-page {
    background:
        radial-gradient(circle at 16% 14%, rgba(96, 165, 250, .20), transparent 30%),
        radial-gradient(circle at 84% 18%, rgba(125, 211, 252, .17), transparent 28%),
        linear-gradient(135deg, #eff6ff 0%, #dbeafe 44%, #f8fbff 100%);
}
body::before {
    content: "";
    position: fixed;
    inset: 0;
    pointer-events: none;
    background-image:
        linear-gradient(rgba(15, 23, 42, .035) 1px, transparent 1px),
        linear-gradient(90deg, rgba(15, 23, 42, .035) 1px, transparent 1px);
    background-size: 34px 34px;
    mask-image: linear-gradient(180deg, rgba(0,0,0,.75), transparent 72%);
}
header {
    position: sticky;
    top: 0;
    z-index: 10;
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 22px;
    padding: 18px 36px;
    color: #1e3a8a;
    background: linear-gradient(135deg, rgba(147, 197, 253, .72), rgba(96, 165, 250, .62), rgba(186, 230, 253, .58));
    border-bottom: 1px solid rgba(255, 255, 255, .34);
    box-shadow: 0 10px 28px rgba(37, 99, 235, .12);
    backdrop-filter: blur(16px);
}
.brand { display: flex; align-items: center; gap: 14px; min-width: 260px; }
.brand-mark {
    width: 46px;
    height: 46px;
    border-radius: 8px;
    display: grid;
    place-items: center;
    font-weight: 900;
    letter-spacing: 0;
    color: #0b1220;
    background: linear-gradient(135deg, #f8fafc, #bfdbfe);
    box-shadow: inset 0 -10px 18px rgba(15, 23, 42, .12);
}
h1 { margin: 0 0 5px; font-size: 22px; line-height: 1.2; letter-spacing: 0; }
header p { margin: 0; color: rgba(30, 58, 138, .68); font-size: 13px; }
nav { display: flex; gap: 8px; flex-wrap: wrap; justify-content: flex-end; align-items: center; }
nav a, nav span {
    color: #1e3a8a;
    text-decoration: none;
    min-height: 34px;
    display: inline-flex;
    align-items: center;
    border-radius: 7px;
    padding: 0 12px;
    font-size: 13px;
    background: rgba(255, 255, 255, .24);
    border: 1px solid rgba(255, 255, 255, .32);
}
nav span { background: rgba(255, 255, 255, .36); font-weight: 700; }
nav a:hover { background: rgba(255, 255, 255, .28); }
main {
    position: relative;
    width: min(1440px, 100%);
    margin: 0 auto;
    padding: 28px 36px 44px;
}
.grid { display: grid; grid-template-columns: minmax(420px, 1.35fr) minmax(330px, .82fr); gap: 22px; align-items: start; }
section, .panel {
    background: var(--panel);
    border: 1px solid rgba(255, 255, 255, .48);
    border-radius: 8px;
    padding: 22px;
    box-shadow: 0 18px 42px rgba(190, 24, 93, .11);
    backdrop-filter: blur(18px);
}
.input-panel { border-top: 4px solid var(--blue); }
.result-panel { position: sticky; top: 98px; border-top: 4px solid var(--pink); }
h2 { margin: 0 0 16px; font-size: 18px; line-height: 1.25; }
h3 { margin: 14px 0 10px; font-size: 15px; }
.form-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 14px; }
.auth-shell { min-height: calc(100vh - 170px); display: grid; place-items: center; }
.auth-card {
    width: min(460px, 100%);
    background: rgba(255, 255, 255, .58);
    border-color: rgba(255, 255, 255, .46);
    box-shadow: 0 22px 52px rgba(190, 24, 93, .14);
    backdrop-filter: blur(18px);
}
.auth-form { display: grid; gap: 16px; }
label { display: grid; gap: 7px; font-size: 13px; color: #475467; font-weight: 700; }
input, select {
    width: 100%;
    min-width: 0;
    height: 40px;
    border: 1px solid #cfd8e3;
    border-radius: 7px;
    padding: 0 11px;
    background: rgba(255, 255, 255, .92);
    color: var(--ink);
    outline: none;
    transition: border-color .16s ease, box-shadow .16s ease, background .16s ease;
}
input:focus, select:focus { border-color: var(--blue); box-shadow: 0 0 0 3px rgba(37, 99, 235, .16); background: white; }
button, .button {
    min-height: 42px;
    border: 0;
    border-radius: 7px;
    background: linear-gradient(135deg, var(--blue), var(--sky));
    color: white;
    font-weight: 800;
    cursor: pointer;
    text-decoration: none;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    padding: 0 18px;
    box-shadow: 0 12px 22px rgba(37, 99, 235, .22);
}
button:hover, .button:hover { filter: brightness(1.04); transform: translateY(-1px); }
.actions { grid-column: 1 / -1; display: flex; gap: 12px; align-items: center; flex-wrap: wrap; }
.note { color: var(--muted); font-size: 13px; line-height: 1.65; font-weight: 400; }
.result { display: grid; gap: 14px; }
.result-empty {
    min-height: 164px;
    display: grid;
    align-content: center;
    gap: 10px;
    padding: 18px;
    border: 1px dashed rgba(236, 72, 153, .32);
    border-radius: 8px;
    background: linear-gradient(135deg, rgba(255,255,255,.40), rgba(252,231,243,.42));
    backdrop-filter: blur(10px);
}
.score-row { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
.score-card {
    border: 1px solid rgba(255, 255, 255, .46);
    background: linear-gradient(180deg, rgba(255,255,255,.58), rgba(252,231,243,.34));
    border-radius: 8px;
    padding: 16px;
    backdrop-filter: blur(10px);
}
.score-card span { color: var(--muted); font-size: 13px; }
.score { margin-top: 7px; font-size: 28px; font-weight: 900; color: #0f172a; letter-spacing: 0; }
.level {
    display: inline-flex;
    width: fit-content;
    align-items: center;
    min-height: 34px;
    padding: 0 12px;
    border-radius: 7px;
    background: rgba(37, 99, 235, .12);
    color: #1d4ed8;
    font-weight: 900;
}
.detail-block { border-top: 1px solid #e4eaf0; padding-top: 12px; }
.detail-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px; }
.detail-item {
    background: rgba(255, 255, 255, .46);
    border: 1px solid rgba(255, 255, 255, .42);
    border-radius: 7px;
    padding: 11px;
    color: #526071;
    font-size: 13px;
    backdrop-filter: blur(10px);
}
.detail-item strong { display: inline-block; margin-top: 4px; color: var(--ink); font-size: 15px; }
.advice-list { margin: 6px 0 0; padding-left: 18px; color: #344256; line-height: 1.75; }
.metrics { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px; margin-top: 16px; }
.metric {
    background: rgba(255, 255, 255, .46);
    border: 1px solid rgba(255, 255, 255, .42);
    border-radius: 8px;
    padding: 12px;
    font-size: 13px;
    color: #506172;
    backdrop-filter: blur(10px);
}
.metric strong { display: block; margin-top: 6px; font-size: 19px; color: #0f172a; }
.flash {
    margin-bottom: 14px;
    padding: 11px 13px;
    background: #fff1f7;
    border: 1px solid #f9a8d4;
    border-radius: 7px;
    color: #9d174d;
}
.table-wrap { overflow-x: auto; border: 1px solid rgba(255, 255, 255, .44); border-radius: 8px; backdrop-filter: blur(12px); }
table { width: 100%; border-collapse: collapse; background: rgba(255, 255, 255, .42); min-width: 760px; }
th, td { border-bottom: 1px solid rgba(255, 255, 255, .42); padding: 12px 10px; text-align: left; font-size: 13px; }
th { background: rgba(255, 255, 255, .48); color: #405060; font-weight: 800; }
tr:hover td { background: rgba(255, 255, 255, .38); }
@media (max-width: 980px) {
    header { align-items: flex-start; flex-direction: column; padding: 18px 20px; position: static; }
    nav { justify-content: flex-start; }
    main { padding: 20px; }
    .grid { grid-template-columns: 1fr; }
    .result-panel { position: static; }
    .form-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
}
@media (max-width: 640px) {
    .brand { align-items: flex-start; min-width: 0; }
    .brand-mark { width: 40px; height: 40px; }
    h1 { font-size: 19px; }
    .form-grid, .score-row, .detail-grid, .metrics { grid-template-columns: 1fr; }
    section, .panel { padding: 16px; }
    nav a, nav span { flex: 1 1 auto; justify-content: center; }
}
"""


def layout(title: str, body: str, user: sqlite3.Row | None = None) -> str:
    nav = (f'<nav><span>{esc(user["username"])}</span><a href="/">信用评估</a><a href="/profile">个人信息</a><a href="/history">历史记录</a><a href="/logout">退出</a></nav>' if user else '<nav><a href="/login">登录</a><a href="/register">注册</a></nav>')
    body_class = "app-page" if user else "auth-page"
    return f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{esc(title)}</title><style>{APP_CSS}</style></head><body class="{body_class}"><header><div class="brand"><div class="brand-mark">Risk</div><div><h1>银行客户信用风险评估系统</h1><p>智能模型推荐、额度测算与风险解释一体化工作台</p></div></div>{nav}</header><main>{body}</main></body></html>'''


def auth_page(kind: str, msg: str = "") -> str:
    is_reg = kind == "register"; title = "用户注册" if is_reg else "用户登录"; action = "/register" if is_reg else "/login"
    switch = '<a href="/login">已有账号，去登录</a>' if is_reg else '<a href="/register">没有账号，去注册</a>'
    return layout(title, f'''<div class="auth-shell"><section class="panel auth-card"><h2>{title}</h2>{'<div class="flash">'+esc(msg)+'</div>' if msg else ''}<form class="auth-form" method="post" action="{action}"><label>用户名<input name="username" required minlength="3" autocomplete="username"></label><label>密码<input name="password" type="password" required minlength="6" autocomplete="current-password"></label><div class="actions"><button type="submit">{title}</button><span class="note">{switch}</span></div></form></section></div>''')


def profile_form(p: dict[str, str], msg: str = "") -> str:
    return f'''<section class="panel"><h2>个人基础信息</h2>{'<div class="flash">'+esc(msg)+'</div>' if msg else ''}<form class="form-grid" method="post" action="/profile">
<label>年龄<input name="age" type="number" min="0" value="{esc(p['age'])}"></label><label>工作年限<input name="employmentYears" type="number" min="0" step="0.5" value="{esc(p['employmentYears'])}"></label><label>信用历史年限<input name="creditHistoryYears" type="number" min="0" step="0.5" value="{esc(p['creditHistoryYears'])}"></label>
<label>身份验证情况<select name="idVerify"><option{sel(p['idVerify'],'一致')}>一致</option><option{sel(p['idVerify'],'不一致')}>不一致</option><option{sel(p['idVerify'],'未知')}>未知</option></select></label><label>三要素验证情况<select name="threeVerify"><option{sel(p['threeVerify'],'一致')}>一致</option><option{sel(p['threeVerify'],'不一致')}>不一致</option><option{sel(p['threeVerify'],'未知')}>未知</option></select></label>
<label>是否有法院记录<select name="inCourt"><option value="0"{sel(p['inCourt'],'0')}>否</option><option value="1"{sel(p['inCourt'],'1')}>是</option></select></label><label>是否在黑名单<select name="isBlackList"><option value="0"{sel(p['isBlackList'],'0')}>否</option><option value="1"{sel(p['isBlackList'],'1')}>是</option></select></label><div class="actions"><button type="submit">保存个人信息</button><a class="button" href="/">进入信用评估</a></div></form></section>'''


def select_options(name: str, current: str, options: list[str]) -> str:
    return f'<select name="{name}">' + ''.join(f'<option{sel(current, o)}>{esc(o)}</option>' for o in options) + '</select>'


def yesno(name: str, current: str) -> str:
    return f'<select name="{name}"><option value="0"{sel(current,"0")}>否</option><option value="1"{sel(current,"1")}>是</option></select>'




def model_select() -> str:
    options = f'<option value="{AUTO_MODEL_KEY}" selected>\u7cfb\u7edf\u81ea\u52a8\u63a8\u8350\u6a21\u578b - \u6839\u636e\u7533\u8bf7\u4eba\u4fe1\u606f\u81ea\u52a8\u9009\u62e9</option>'
    options += ''.join(
        f'<option value="{esc(key)}">{esc(label)} - {esc(MODEL_NOTES[key])}</option>'
        for key, label in MODEL_LABELS.items()
    )
    return f'<select name="modelName">{options}</select>'

def index_page(user: sqlite3.Row, profile: dict[str, str]) -> str:
    v = ASSESSMENT_DEFAULTS.copy(); v.update(profile)
    body = f'''<div class="grid"><section class="input-panel"><h2>信用评估申请信息</h2><form class="form-grid" id="riskForm">
<label>评估模型{model_select()}</label><label>年龄<input name="age" type="number" value="{esc(v['age'])}" min="0"></label><label>工作年限<input name="employmentYears" type="number" value="{esc(v['employmentYears'])}" min="0" step="0.5"></label><label>信用历史年限<input name="creditHistoryYears" type="number" value="{esc(v['creditHistoryYears'])}" min="0" step="0.5"></label>
<label>城市等级{select_options('CityId',v['CityId'],['一线城市','二线城市','三线城市','其他'])}</label><label>学历{select_options('education',v['education'],['高中','本科','硕士及以上','其他'])}</label><label>婚姻状态{select_options('maritalStatus',v['maritalStatus'],['未婚','已婚','其他'])}</label><label>性别{select_options('sex',v['sex'],['男','女'])}</label><label>在网时长{select_options('netLength',v['netLength'],['0-6个月','6-12个月','12-24个月','24个月以上','无效'])}</label>
<label>身份验证{select_options('idVerify',v['idVerify'],['一致','不一致','未知'])}</label><label>三要素验证{select_options('threeVerify',v['threeVerify'],['一致','不一致','未知'])}</label><label>银行卡开卡年限<input name="card_age" type="number" value="{esc(v['card_age'])}" min="0"></label><label>总消费金额<input name="transTotalAmt" type="number" value="{esc(v['transTotalAmt'])}" min="0"></label><label>总消费笔数<input name="transTotalCnt" type="number" value="{esc(v['transTotalCnt'])}" min="0"></label><label>网上消费金额<input name="onlineTransAmt" type="number" value="{esc(v['onlineTransAmt'])}" min="0"></label><label>取现金额<input name="cashTotalAmt" type="number" value="{esc(v['cashTotalAmt'])}" min="0"></label>
<label>是否有法院记录{yesno('inCourt',v['inCourt'])}</label><label>是否在黑名单{yesno('isBlackList',v['isBlackList'])}</label><label>是否逾期{yesno('isDue',v['isDue'])}</label><label>月收入<input name="monthlyIncome" type="number" value="{esc(v['monthlyIncome'])}" min="0"></label><label>月支出<input name="monthlyExpense" type="number" value="{esc(v['monthlyExpense'])}" min="0"></label><label>已有月还款金额<input name="existingMonthlyRepayment" type="number" value="{esc(v['existingMonthlyRepayment'])}" min="0"></label><label>申请贷款金额<input name="requestedAmount" type="number" value="{esc(v['requestedAmount'])}" min="0"></label><label>贷款期限（月）<input name="loanTerm" type="number" value="{esc(v['loanTerm'])}" min="1"></label><label>历史逾期次数<input name="overdueCount" type="number" value="{esc(v['overdueCount'])}" min="0"></label><label>信用卡使用率（%）<input name="creditCardUsage" type="number" value="{esc(v['creditCardUsage'])}" min="0" max="100"></label>
<div class="actions"><button type="submit">评估违约风险</button><span class="note">已从个人信息自动填充长期基础字段，可在本次评估中临时修改。</span></div></form></section><section class="result-panel"><h2>评估结果</h2><div class="result" id="result"><div class="result-empty"><strong>等待提交评估</strong><span class="note">提交申请人信息后，这里会显示风险概率、信用评分、额度建议和改进方案。</span></div></div><div class="metrics"><div class="metric">模型区分能力<strong id="auc">-</strong><span class="note">数值越高，说明模型越能区分高低风险客户</span></div><div class="metric">当前模型<strong id="modelNameDisplay">梯度提升模型</strong></div></div></section></div>
<script>const form=document.getElementById("riskForm"),result=document.getElementById("result"),auc=document.getElementById("auc");const money=v=>Number(v).toLocaleString("zh-CN",{{style:"currency",currency:"CNY",maximumFractionDigits:0}});form.addEventListener("submit",async e=>{{e.preventDefault();const payload=Object.fromEntries(new FormData(form).entries());result.innerHTML='<div class="result-empty"><strong>正在评估...</strong><span class="note">模型正在计算风险概率和推荐额度。</span></div>';const r=await fetch("/predict",{{method:"POST",headers:{{"Content-Type":"application/json"}},body:JSON.stringify(payload)}});if(!r.ok){{result.innerHTML='<div class="result-empty"><strong>评估失败</strong><span class="note">请重新登录后再试。</span></div>';return}}const d=await r.json();auc.textContent=d.auc;document.getElementById("modelNameDisplay").textContent=d.model_label;const a=d.loan_amount,items=d.improvement_advice.map(x=>`<li>${{x}}</li>`).join("");result.innerHTML=`<div><strong>使用模型：</strong>${{d.model_label}}</div><div class="score-row"><div class="score-card"><span>违约风险概率</span><div class="score">${{d.percentage}}</div></div><div class="score-card"><span>信用评分</span><div class="score">${{d.credit_score}}</div></div></div><div class="level">${{d.level}}</div><div><strong>审批建议：</strong>${{d.suggestion}}</div><div class="detail-block"><h3>建议可贷款额度</h3><div class="detail-grid"><div class="detail-item">可支配月收入<br><strong>${{money(a.disposable_income)}}</strong></div><div class="detail-item">推荐可贷款额度<br><strong>${{money(a.recommended_amount)}}</strong></div><div class="detail-item">申请金额<br><strong>${{money(a.requested_amount)}}</strong></div><div class="detail-item">金额判断<br><strong>${{a.amount_comment}}</strong></div></div></div><div class="detail-block"><h3>信用提升建议</h3><ul class="advice-list">${{items}}</ul></div><div class="detail-block"><h3>预计贷款流程和到账时间</h3><div class="note">${{d.loan_process}}</div><div><strong>${{d.expected_time}}</strong></div></div>`}});</script>'''
    return layout("信用评估", body, user)


def history_page(user: sqlite3.Row, rows: list[sqlite3.Row]) -> str:
    trs = ''.join(f"<tr><td>{esc(r['created_time'])}</td><td>{money(r['requested_loan_amount'])}</td><td>{r['loan_term']} 月</td><td>{esc(r['risk_level'])}</td><td>{r['default_probability']*100:.2f}%</td><td>{r['credit_score']:.2f}</td><td>{money(r['recommended_loan_amount'])}</td></tr>" for r in rows) or '<tr><td colspan="7" class="note">暂无历史评估记录。</td></tr>'
    return layout("历史记录", f'<section class="panel"><h2>历史评估记录</h2><div class="table-wrap"><table><thead><tr><th>评估时间</th><th>申请金额</th><th>期限</th><th>风险等级</th><th>违约概率</th><th>信用评分</th><th>推荐额度</th></tr></thead><tbody>{trs}</tbody></table></div></section>', user)

class RiskHandler(BaseHTTPRequestHandler):
    def send_body(self, status: int, body: str | bytes, ctype: str = "text/html; charset=utf-8", headers: dict[str, str] | None = None) -> None:
        payload = body.encode("utf-8") if isinstance(body, str) else body
        self.send_response(status); self.send_header("Content-Type", ctype); self.send_header("Content-Length", str(len(payload)))
        for k, v in (headers or {}).items(): self.send_header(k, v)
        self.end_headers(); self.wfile.write(payload)

    def redirect(self, location: str, headers: dict[str, str] | None = None) -> None:
        self.send_body(302, b"", "text/plain; charset=utf-8", {"Location": location, **(headers or {})})

    def form_data(self) -> dict[str, str]:
        raw = self.rfile.read(int(self.headers.get("Content-Length", "0"))).decode("utf-8")
        return {k: v[-1] for k, v in parse_qs(raw).items()}

    def json_data(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        return json.loads(self.rfile.read(length).decode("utf-8")) if length else {}

    def current_user_id(self) -> int | None:
        jar = cookies.SimpleCookie(self.headers.get("Cookie", "")); morsel = jar.get(SESSION_COOKIE)
        return SESSIONS.get(morsel.value) if morsel else None

    def current_user(self) -> sqlite3.Row | None:
        return get_user(self.current_user_id())

    def require_user(self) -> sqlite3.Row | None:
        user = self.current_user()
        if not user: self.redirect("/login")
        return user

    def set_session_header(self, user_id: int) -> str:
        token = secrets.token_urlsafe(32); SESSIONS[token] = user_id
        return f"{SESSION_COOKIE}={token}; HttpOnly; Path=/; SameSite=Lax"

    def clear_session_header(self) -> str:
        jar = cookies.SimpleCookie(self.headers.get("Cookie", "")); morsel = jar.get(SESSION_COOKIE)
        if morsel: SESSIONS.pop(morsel.value, None)
        return f"{SESSION_COOKIE}=; HttpOnly; Path=/; Max-Age=0; SameSite=Lax"

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/login": self.send_body(200, auth_page("login")); return
        if path == "/register": self.send_body(200, auth_page("register")); return
        if path == "/logout": self.redirect("/login", {"Set-Cookie": self.clear_session_header()}); return
        user = self.require_user()
        if not user: return
        if path == "/": self.send_body(200, index_page(user, get_profile(user["id"]))); return
        if path == "/profile": self.send_body(200, layout("个人信息", profile_form(get_profile(user["id"])), user)); return
        if path == "/history": self.send_body(200, history_page(user, history(user["id"]))); return
        self.send_body(404, "Not Found", "text/plain; charset=utf-8")

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path == "/register":
            data = self.form_data(); ok, msg = create_user(data.get("username", ""), data.get("password", ""))
            self.send_body(200 if ok else 400, auth_page("login" if ok else "register", msg)); return
        if path == "/login":
            data = self.form_data(); user_id = authenticate(data.get("username", ""), data.get("password", ""))
            if user_id: self.redirect("/", {"Set-Cookie": self.set_session_header(user_id)})
            else: self.send_body(401, auth_page("login", "用户名或密码错误。"))
            return
        user = self.require_user()
        if not user: return
        if path == "/profile":
            update_profile(user["id"], self.form_data())
            self.send_body(200, layout("个人信息", profile_form(get_profile(user["id"]), "个人信息已保存。"), user)); return
        if path == "/predict":
            result = predict_risk(self.json_data()); save_record(user["id"], result)
            self.send_body(200, json.dumps(result, ensure_ascii=False), "application/json; charset=utf-8"); return
        self.send_body(404, "Not Found", "text/plain; charset=utf-8")

    def log_message(self, fmt: str, *args: Any) -> None:
        return


def run_server(host: str, port: int) -> None:
    init_db(); server = ThreadingHTTPServer((host, port), RiskHandler)
    print(f"银行客户信用风险评估系统已启动: http://{host}:{port}")
    print("模型测试 AUC:")
    for key, label in MODEL_LABELS.items():
        print(f"- {label}: {MODEL_AUCS[key]:.4f}")
    server.serve_forever()

def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--host", default="127.0.0.1"); parser.add_argument("--port", type=int, default=8000); parser.add_argument("--check", action="store_true")
    args = parser.parse_args(); init_db()
    if args.check:
        sample = {"modelName": AUTO_MODEL_KEY, "age": 35, "employmentYears": 5, "CityId": "二线城市", "education": "高中", "maritalStatus": "已婚", "sex": "男", "idVerify": "一致", "threeVerify": "一致", "inCourt": 0, "isBlackList": 0, "isDue": 0, "transTotalAmt": 5000, "transTotalCnt": 20, "monthlyIncome": 10000, "monthlyExpense": 4500, "existingMonthlyRepayment": 1500, "requestedAmount": 50000, "loanTerm": 12, "overdueCount": 1, "creditCardUsage": 35, "creditHistoryYears": 3}
        print(predict_risk(sample)); return
    run_server(args.host, args.port)


if __name__ == "__main__":
    main()
