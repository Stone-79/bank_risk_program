# 银行客户信用风险评估系统

本项目基于《第13章 数据分析案例5--银行客户信用风险评估》的银行客户风险数据集，构建个人信用风险评估模型，用于预测贷款申请人是否存在违约风险，并通过本地浏览器页面展示评估结果。

## 项目功能

- 用户注册、登录、退出。
- 用户个人基础信息保存：年龄、工作年限、信用历史年限、身份验证、三要素验证、法院记录、黑名单状态。
- 登录后自动读取个人基础信息并填充到信用评估表单。
- 使用逻辑回归模型输出基础违约概率。
- 在模型基础概率上叠加业务规则评分，让身份验证、三要素验证、法院记录、黑名单、逾期、收入负债、信用卡使用率等输入直接影响最终风险结果。
- 最终违约概率限制在 0.01 到 0.99 之间，避免异常输入导致概率越界。
- 输出五级风险等级：A级低风险、B级较低风险、C级中等风险、D级较高风险、E级高风险。
- 展示信用评分：信用评分 = 100 - 违约概率 × 100。
- 根据风险等级展示审批建议、推荐可贷款额度、申请金额合理性判断、信用提升建议、预计贷款流程和预计到账时间。
- 将每次评估结果保存到 SQLite 数据库，并提供历史评估记录页面。

## 项目结构

```text
D:\bank_program
├── README.md
├── bank_risk_web_app.py          # 主系统：模型训练、登录注册、前端页面、预测接口、历史记录
├── bank_risk_preview.py          # 快速预览脚本：输出数据概况、AUC、预览图
├── bank_risk_users.db            # 运行后自动生成的 SQLite 用户与评估记录数据库
├── outputs/                      # 预览脚本生成的图片
└── 源代码/
    ├── 第13章数据分析案例5--银行客户信用风险评估.ipynb
    ├── data/
    │   └── bankriskinfo.csv
    ├── picture/
    └── ttf/
        └── simkai.ttf
```

## 环境依赖

建议使用 Python 3.10 或 Python 3.11。

必需依赖：

```text
pandas
numpy
scikit-learn
matplotlib
seaborn
```

如果需要运行 Notebook，还需要：

```text
jupyter
nbformat
nbclient
nbconvert
```

## 如何安装依赖

使用 pip：

```powershell
pip install pandas numpy scikit-learn matplotlib seaborn jupyter nbformat nbclient nbconvert
```

使用 conda：

```powershell
conda install pandas numpy scikit-learn matplotlib seaborn jupyter nbformat nbclient nbconvert
```

当前本机已使用过的解释器路径：

```text
D:\Anaconda\envs\test\python.exe
```

## 如何训练模型

模型训练逻辑写在 `bank_risk_web_app.py` 中，系统启动时会自动完成：

1. 读取 `源代码/data/bankriskinfo.csv`。
2. 将 `Default1` 字段重命名为 `Default`。
3. 划分训练集和测试集。
4. 数值字段缺失值用中位数填充。
5. 类别字段缺失值填充后进行 One-Hot 编码。
6. 训练逻辑回归模型。
7. 计算测试集 AUC。
8. 预测时叠加业务规则修正，得到最终违约概率、信用评分和风险等级。

检查模型是否能正常训练和预测：

```powershell
cd D:\bank_program
D:\Anaconda\envs\test\python.exe bank_risk_web_app.py --check
```

## 如何启动系统

在 PowerShell 或 PyCharm Terminal 中运行：

```powershell
cd D:\bank_program
D:\Anaconda\envs\test\python.exe bank_risk_web_app.py --host 127.0.0.1 --port 8000
```

启动成功后，终端会显示类似：

```text
银行客户信用风险评估系统已启动: http://127.0.0.1:8000
模型测试 AUC: 0.8597
```

保持该终端窗口不要关闭，系统服务会持续运行。

## 如何在浏览器中访问

启动系统后，打开 Microsoft Edge，访问：

```text
http://127.0.0.1:8000/
```

首次使用流程：

1. 打开注册页面创建账号。
2. 登录系统。
3. 进入个人信息页面，填写或修改长期基础信息。
4. 返回信用评估页面，系统会自动填充已保存的个人基础信息。
5. 修改本次贷款相关信息，例如月收入、月支出、已有月还款金额、申请金额、贷款期限、逾期情况、信用卡使用率。
6. 点击“评估违约风险”查看结果。
7. 可在“历史记录”页面查看最近 30 次评估结果。

## 当前模型说明

当前方案是“机器学习基础概率 + 业务规则修正”的组合模型。逻辑回归模型负责根据历史数据输出基础违约概率，业务规则负责让新增前端输入直接影响最终违约概率、信用评分和风险等级。该方式适合课程项目展示，也方便解释每类输入对风险结果的影响。
