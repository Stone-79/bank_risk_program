from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "源代码" / "data" / "bankriskinfo.csv"


def _make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def load_data() -> pd.DataFrame:
    data = pd.read_csv(DATA_PATH, encoding="utf-8").rename(columns={"Default1": "Default"})
    return data.dropna(subset=["Default"]).copy()


def train_model() -> tuple[Pipeline, dict[str, Any], list[str], float]:
    data = load_data()
    y = data["Default"].astype(int)
    x = data.drop(columns=["Default"])

    numeric_cols = x.select_dtypes(include=["number"]).columns.tolist()
    category_cols = [column for column in x.columns if column not in numeric_cols]

    defaults: dict[str, Any] = {}
    for column in numeric_cols:
        defaults[column] = float(x[column].median())
    for column in category_cols:
        mode = x[column].mode(dropna=True)
        defaults[column] = str(mode.iloc[0]) if not mode.empty else "未知"

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numeric_cols),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", _make_one_hot_encoder()),
                    ]
                ),
                category_cols,
            ),
        ]
    )
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    solver="liblinear",
                    max_iter=1000,
                    random_state=33,
                ),
            ),
        ]
    )

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=33, stratify=y
    )
    model.fit(x_train, y_train)
    auc = roc_auc_score(y_test, model.predict_proba(x_test)[:, 1])
    return model, defaults, x.columns.tolist(), float(auc)


MODEL, DEFAULTS, FEATURE_COLUMNS, MODEL_AUC = train_model()


RISK_POLICIES = {
    "A": {
        "level": "A级，低风险",
        "suggestion": "可直接通过",
        "coefficient": 1.2,
        "process": "自动初审 → 资料确认 → 放款",
        "time": "预计 1-2 个工作日",
    },
    "B": {
        "level": "B级，较低风险",
        "suggestion": "可放贷",
        "coefficient": 1.0,
        "process": "自动初审 → 基础人工复核 → 放款",
        "time": "预计 2-3 个工作日",
    },
    "C": {
        "level": "C级，中等风险",
        "suggestion": "人工审核",
        "coefficient": 0.7,
        "process": "系统初审 → 人工复核 → 补充资料 → 额度调整 → 放款",
        "time": "预计 3-5 个工作日",
    },
    "D": {
        "level": "D级，较高风险",
        "suggestion": "降低额度或提高利率",
        "coefficient": 0.4,
        "process": "系统初审 → 严格人工审核 → 补充收入或资产证明 → 降低额度后重新评估",
        "time": "预计 5-7 个工作日",
    },
    "E": {
        "level": "E级，高风险",
        "suggestion": "建议拒贷",
        "coefficient": 0.0,
        "process": "系统初审 → 高风险提示 → 暂不建议申请贷款",
        "time": "建议先提高信用状况后再申请",
    },
}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in ("", None):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _grade_from_probability(probability: float) -> str:
    if probability >= 0.80:
        return "E"
    if probability >= 0.60:
        return "D"
    if probability >= 0.40:
        return "C"
    if probability >= 0.20:
        return "B"
    return "A"


def _clamp_probability(probability: float) -> float:
    return min(max(probability, 0.01), 0.99)


def _apply_business_rules(base_probability: float, form_data: dict[str, Any]) -> tuple[float, float]:
    adjustment = 0.0

    id_verify = str(form_data.get("idVerify", "")).strip()
    three_verify = str(form_data.get("threeVerify", "")).strip()
    if id_verify == "不一致":
        adjustment += 0.12
    elif id_verify == "未知":
        adjustment += 0.06
    if three_verify == "不一致":
        adjustment += 0.10
    elif three_verify == "未知":
        adjustment += 0.05

    if _safe_float(form_data.get("inCourt")) > 0:
        adjustment += 0.12
    if _safe_float(form_data.get("isBlackList")) > 0:
        adjustment += 0.20
    if _safe_float(form_data.get("isDue")) > 0:
        adjustment += 0.12

    overdue_count = _safe_float(form_data.get("overdueCount"))
    if overdue_count >= 5:
        adjustment += 0.15
    elif overdue_count >= 3:
        adjustment += 0.10
    elif overdue_count > 0:
        adjustment += 0.05

    monthly_income = _safe_float(form_data.get("monthlyIncome"))
    monthly_expense = _safe_float(form_data.get("monthlyExpense"))
    existing_repayment = _safe_float(form_data.get("existingMonthlyRepayment"))
    requested_amount = _safe_float(form_data.get("requestedAmount"))
    loan_term = max(_safe_float(form_data.get("loanTerm"), 12), 1)
    disposable_income = monthly_income - monthly_expense - existing_repayment
    debt_income_ratio = existing_repayment / monthly_income if monthly_income > 0 else 0
    expense_income_ratio = monthly_expense / monthly_income if monthly_income > 0 else 0
    requested_monthly_payment = requested_amount / loan_term if loan_term > 0 else requested_amount

    if monthly_income <= 0:
        adjustment += 0.12
    if disposable_income <= 0:
        adjustment += 0.15
    elif requested_monthly_payment > disposable_income:
        adjustment += 0.10
    if debt_income_ratio > 0.6:
        adjustment += 0.12
    elif debt_income_ratio > 0.4:
        adjustment += 0.08
    if expense_income_ratio > 0.8:
        adjustment += 0.08
    elif expense_income_ratio > 0.65:
        adjustment += 0.04

    credit_card_usage = _safe_float(form_data.get("creditCardUsage"))
    if credit_card_usage > 80:
        adjustment += 0.10
    elif credit_card_usage > 50:
        adjustment += 0.06

    credit_history_years = _safe_float(form_data.get("creditHistoryYears"))
    if credit_history_years < 1:
        adjustment += 0.08
    elif credit_history_years < 2:
        adjustment += 0.04
    elif credit_history_years >= 5:
        adjustment -= 0.03

    if id_verify == "一致" and three_verify == "一致" and overdue_count == 0:
        adjustment -= 0.03
    if monthly_income > 0 and debt_income_ratio < 0.25 and disposable_income > 0:
        adjustment -= 0.03

    final_probability = _clamp_probability(base_probability + adjustment)
    return final_probability, adjustment


def _loan_amount_advice(form_data: dict[str, Any], coefficient: float) -> dict[str, Any]:
    monthly_income = _safe_float(form_data.get("monthlyIncome"))
    monthly_expense = _safe_float(form_data.get("monthlyExpense"))
    existing_repayment = _safe_float(form_data.get("existingMonthlyRepayment"))
    loan_term = max(_safe_float(form_data.get("loanTerm"), 12), 0)
    requested_amount = _safe_float(form_data.get("requestedAmount"))

    disposable_income = monthly_income - monthly_expense - existing_repayment
    base_amount = max(disposable_income, 0) * loan_term
    recommended_amount = base_amount * coefficient
    amount_comment = (
        "当前申请金额较合理"
        if requested_amount <= recommended_amount
        else "申请金额可能偏高，建议降低申请额度"
    )

    return {
        "monthly_income": round(monthly_income, 2),
        "monthly_expense": round(monthly_expense, 2),
        "existing_monthly_repayment": round(existing_repayment, 2),
        "loan_term": int(loan_term),
        "requested_amount": round(requested_amount, 2),
        "disposable_income": round(disposable_income, 2),
        "base_amount": round(base_amount, 2),
        "risk_coefficient": coefficient,
        "recommended_amount": round(recommended_amount, 2),
        "amount_comment": amount_comment,
    }


def _credit_improvement_advice(form_data: dict[str, Any], grade: str) -> list[str]:
    suggestions: list[str] = []
    overdue_count = _safe_float(form_data.get("overdueCount"))
    credit_card_usage = _safe_float(form_data.get("creditCardUsage"))
    monthly_income = _safe_float(form_data.get("monthlyIncome"))
    monthly_expense = _safe_float(form_data.get("monthlyExpense"))
    existing_repayment = _safe_float(form_data.get("existingMonthlyRepayment"))
    credit_history_years = _safe_float(form_data.get("creditHistoryYears"))
    debt_income_ratio = existing_repayment / monthly_income if monthly_income > 0 else 0
    expense_income_ratio = monthly_expense / monthly_income if monthly_income > 0 else 0

    if overdue_count >= 3:
        suggestions.append("历史逾期次数较多，建议保持按时还款，减少逾期记录。")
    elif overdue_count > 0:
        suggestions.append("存在少量逾期记录，建议后续保持连续按时还款。")
    if credit_card_usage > 50:
        suggestions.append("信用卡使用率较高，建议降低信用卡额度使用比例，控制在 50% 以下。")
    if debt_income_ratio > 0.4:
        suggestions.append("负债收入比较高，建议先减少已有负债，提高可支配收入。")
    if credit_history_years < 2:
        suggestions.append("信用历史较短，建议保持稳定信用记录，延长信用积累时间。")
    if expense_income_ratio > 0.7:
        suggestions.append("月支出占收入比例较高，建议优化消费结构，提高还款能力。")
    if grade in {"D", "E"}:
        suggestions.append("当前风险等级偏高，建议暂缓大额贷款申请，先改善信用状况。")
    if not suggestions:
        suggestions.append("当前信用状况较稳定，建议继续保持良好还款习惯和合理负债水平。")

    return suggestions


def predict_risk(form_data: dict[str, Any]) -> dict[str, Any]:
    row = DEFAULTS.copy()
    for key, value in form_data.items():
        if key not in row or value in ("", None):
            continue
        if isinstance(DEFAULTS[key], float):
            row[key] = float(value)
        else:
            row[key] = str(value)

    sample = pd.DataFrame([row], columns=FEATURE_COLUMNS)
    base_probability = float(MODEL.predict_proba(sample)[0, 1])
    probability, business_adjustment = _apply_business_rules(base_probability, form_data)
    credit_score = 100 - probability * 100
    grade = _grade_from_probability(probability)
    policy = RISK_POLICIES[grade]
    amount_advice = _loan_amount_advice(form_data, policy["coefficient"])
    improvement_advice = _credit_improvement_advice(form_data, grade)

    return {
        "probability": round(probability, 4),
        "base_probability": round(base_probability, 4),
        "business_adjustment": round(business_adjustment, 4),
        "percentage": f"{probability * 100:.2f}%",
        "credit_score": round(credit_score, 2),
        "level": policy["level"],
        "suggestion": policy["suggestion"],
        "loan_amount": amount_advice,
        "improvement_advice": improvement_advice,
        "loan_process": policy["process"],
        "expected_time": policy["time"],
        "auc": round(MODEL_AUC, 4),
    }


HTML = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>银行客户信用风险评估</title>
  <style>
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Microsoft YaHei", Arial, sans-serif;
      color: #17212b;
      background: #f4f6f8;
    }
    header {
      padding: 22px 32px;
      background: #12324a;
      color: white;
      border-bottom: 4px solid #2c7a7b;
    }
    header h1 { margin: 0 0 6px; font-size: 24px; }
    header p { margin: 0; color: #cfe3ef; }
    main {
      display: grid;
      grid-template-columns: minmax(360px, 1.35fr) minmax(300px, .85fr);
      gap: 20px;
      padding: 22px 32px;
    }
    section {
      background: white;
      border: 1px solid #dde3ea;
      border-radius: 8px;
      padding: 18px;
    }
    h2 { margin: 0 0 14px; font-size: 18px; }
    form {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 14px;
    }
    label { display: grid; gap: 6px; font-size: 13px; color: #435160; }
    input, select {
      width: 100%;
      height: 36px;
      border: 1px solid #c9d2dc;
      border-radius: 6px;
      padding: 0 10px;
      background: #fff;
      color: #17212b;
    }
    button {
      height: 40px;
      border: 0;
      border-radius: 6px;
      background: #216869;
      color: white;
      font-weight: 700;
      cursor: pointer;
    }
    button:hover { background: #174f50; }
    .actions { grid-column: 1 / -1; display: flex; gap: 10px; align-items: center; }
    .result {
      min-height: 190px;
      display: grid;
      align-content: start;
      gap: 12px;
    }
    .score { font-size: 38px; font-weight: 800; color: #12324a; }
    .score-row {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
    }
    .score-card {
      border: 1px solid #e3e8ee;
      border-radius: 8px;
      padding: 12px;
      background: #fafbfc;
    }
    .score-card span {
      display: block;
      color: #66727f;
      font-size: 13px;
      margin-bottom: 4px;
    }
    .level {
      display: inline-block;
      width: fit-content;
      padding: 7px 12px;
      border-radius: 999px;
      background: #e7f3f2;
      color: #185b5d;
      font-weight: 700;
    }
    .note { color: #66727f; line-height: 1.7; }
    .metrics {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
      margin-top: 14px;
    }
    .metric {
      border: 1px solid #e3e8ee;
      border-radius: 6px;
      padding: 12px;
      background: #fafbfc;
    }
    .metric strong { display: block; font-size: 20px; margin-top: 4px; }
    .detail-block {
      border-top: 1px solid #edf1f5;
      padding-top: 12px;
    }
    .detail-block h3 {
      margin: 0 0 8px;
      font-size: 15px;
    }
    .detail-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 8px;
      font-size: 14px;
    }
    .detail-item {
      padding: 9px;
      border-radius: 6px;
      background: #f7f9fb;
      border: 1px solid #e4eaf0;
    }
    .advice-list {
      margin: 0;
      padding-left: 18px;
      display: grid;
      gap: 6px;
      line-height: 1.6;
      color: #435160;
    }
    @media (max-width: 900px) {
      main { grid-template-columns: 1fr; padding: 16px; }
      form { grid-template-columns: 1fr; }
      header { padding: 18px 16px; }
    }
  </style>
</head>
<body>
  <header>
    <h1>银行客户信用风险评估</h1>
    <p>填写贷款申请人的基础信息，系统预测违约风险概率和审批建议。</p>
  </header>
  <main>
    <section>
      <h2>申请人信息</h2>
      <form id="riskForm">
        <label>年龄
          <input name="age" type="number" value="35" min="18" max="80" />
        </label>
        <label>城市级别
          <select name="CityId">
            <option>一线城市</option>
            <option>二线城市</option>
            <option>其它</option>
          </select>
        </label>
        <label>学历
          <select name="education">
            <option>小学</option>
            <option>初中</option>
            <option>高中</option>
            <option>本科以上</option>
          </select>
        </label>
        <label>婚姻状况
          <select name="maritalStatus">
            <option>已婚</option>
            <option>未婚</option>
            <option>未知</option>
          </select>
        </label>
        <label>性别
          <select name="sex">
            <option>男</option>
            <option>女</option>
          </select>
        </label>
        <label>在网时长
          <select name="netLength">
            <option>0-6个月</option>
            <option>6-12个月</option>
            <option>12-24个月</option>
            <option>24个月以上</option>
            <option>无效</option>
          </select>
        </label>
        <label>身份验证
          <select name="idVerify">
            <option>一致</option>
            <option>不一致</option>
            <option>未知</option>
          </select>
        </label>
        <label>三要素验证
          <select name="threeVerify">
            <option>一致</option>
            <option>不一致</option>
            <option>未知</option>
          </select>
        </label>
        <label>银行卡开卡年限
          <input name="card_age" type="number" value="5" min="0" />
        </label>
        <label>总消费金额
          <input name="transTotalAmt" type="number" value="5000" min="0" />
        </label>
        <label>总消费笔数
          <input name="transTotalCnt" type="number" value="20" min="0" />
        </label>
        <label>网上消费金额
          <input name="onlineTransAmt" type="number" value="1200" />
        </label>
        <label>取现金额
          <input name="cashTotalAmt" type="number" value="0" min="0" />
        </label>
        <label>是否法院记录
          <select name="inCourt">
            <option value="0">否</option>
            <option value="1">是</option>
          </select>
        </label>
        <label>是否黑名单
          <select name="isBlackList">
            <option value="0">否</option>
            <option value="1">是</option>
          </select>
        </label>
        <label>是否逾期
          <select name="isDue">
            <option value="0">否</option>
            <option value="1">是</option>
          </select>
        </label>
        <label>月收入
          <input name="monthlyIncome" type="number" value="10000" min="0" />
        </label>
        <label>月支出
          <input name="monthlyExpense" type="number" value="4500" min="0" />
        </label>
        <label>已有月还款金额
          <input name="existingMonthlyRepayment" type="number" value="1500" min="0" />
        </label>
        <label>申请贷款金额
          <input name="requestedAmount" type="number" value="50000" min="0" />
        </label>
        <label>贷款期限（月）
          <input name="loanTerm" type="number" value="12" min="1" />
        </label>
        <label>历史逾期次数
          <input name="overdueCount" type="number" value="0" min="0" />
        </label>
        <label>信用卡使用率（%）
          <input name="creditCardUsage" type="number" value="35" min="0" max="100" />
        </label>
        <label>信用历史年限
          <input name="creditHistoryYears" type="number" value="3" min="0" step="0.5" />
        </label>
        <div class="actions">
          <button type="submit">评估违约风险</button>
          <span class="note">模型输出用于辅助风控判断，不等同于最终贷款审批。</span>
        </div>
      </form>
    </section>
    <section>
      <h2>评估结果</h2>
      <div class="result" id="result">
        <div class="note">提交申请人信息后，这里会显示预测结果。</div>
      </div>
      <div class="metrics">
        <div class="metric">模型测试 AUC<strong id="auc">-</strong></div>
        <div class="metric">模型类型<strong>逻辑回归</strong></div>
      </div>
    </section>
  </main>
  <script>
    const form = document.getElementById("riskForm");
    const result = document.getElementById("result");
    const auc = document.getElementById("auc");
    const money = (value) => Number(value).toLocaleString("zh-CN", {
      style: "currency",
      currency: "CNY",
      maximumFractionDigits: 0,
    });

    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      const payload = Object.fromEntries(new FormData(form).entries());
      result.innerHTML = '<div class="note">正在评估...</div>';
      const response = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await response.json();
      auc.textContent = data.auc;
      const amount = data.loan_amount;
      const adviceItems = data.improvement_advice
        .map((item) => `<li>${item}</li>`)
        .join("");
      result.innerHTML = `
        <div class="score-row">
          <div class="score-card">
            <span>违约风险概率</span>
            <div class="score">${data.percentage}</div>
          </div>
          <div class="score-card">
            <span>信用评分</span>
            <div class="score">${data.credit_score}</div>
          </div>
        </div>
        <div class="level">${data.level}</div>
        <div><strong>审批建议：</strong>${data.suggestion}</div>
        <div class="detail-block">
          <h3>建议可贷款额度</h3>
          <div class="detail-grid">
            <div class="detail-item">可支配月收入<br><strong>${money(amount.disposable_income)}</strong></div>
            <div class="detail-item">推荐可贷款额度<br><strong>${money(amount.recommended_amount)}</strong></div>
            <div class="detail-item">申请金额<br><strong>${money(amount.requested_amount)}</strong></div>
            <div class="detail-item">金额判断<br><strong>${amount.amount_comment}</strong></div>
          </div>
        </div>
        <div class="detail-block">
          <h3>信用提升建议</h3>
          <ul class="advice-list">${adviceItems}</ul>
        </div>
        <div class="detail-block">
          <h3>预计贷款流程和到账时间</h3>
          <div class="note">${data.loan_process}</div>
          <div><strong>${data.expected_time}</strong></div>
        </div>
      `;
    });
  </script>
</body>
</html>
"""


class RiskHandler(BaseHTTPRequestHandler):
    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/":
            self._send(200, HTML.encode("utf-8"), "text/html; charset=utf-8")
            return
        self._send(404, b"Not Found", "text/plain; charset=utf-8")

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path != "/predict":
            self._send(404, b"Not Found", "text/plain; charset=utf-8")
            return

        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        result = predict_risk(payload)
        self._send(
            200,
            json.dumps(result, ensure_ascii=False).encode("utf-8"),
            "application/json; charset=utf-8",
        )

    def log_message(self, format: str, *args: Any) -> None:
        return


def run_server(host: str, port: int) -> None:
    server = ThreadingHTTPServer((host, port), RiskHandler)
    print(f"银行客户信用风险评估前端已启动: http://{host}:{port}")
    print(f"模型测试 AUC: {MODEL_AUC:.4f}")
    server.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    if args.check:
        sample = {
            "age": 35,
            "CityId": "二线城市",
            "education": "高中",
            "maritalStatus": "已婚",
            "sex": "男",
            "transTotalAmt": 5000,
            "transTotalCnt": 20,
            "monthlyIncome": 10000,
            "monthlyExpense": 4500,
            "existingMonthlyRepayment": 1500,
            "requestedAmount": 50000,
            "loanTerm": 12,
            "overdueCount": 1,
            "creditCardUsage": 35,
            "creditHistoryYears": 3,
        }
        print(predict_risk(sample))
        return

    run_server(args.host, args.port)


if __name__ == "__main__":
    main()
