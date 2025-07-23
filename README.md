# 客户信用风险评估模型 (Credit Risk Prediction Model)

本项目是一个端到端的机器学习解决方案，旨在利用真实的金融信贷数据，预测客户在申请贷款时未来是否会发生违约。该项目完整地覆盖了从数据清洗、探索性数据分析(EDA)、深度特征工程、交叉验证建模到模型评估的全流程。

**项目亮点:**
*   **技术栈**: `Python`, `Pandas`, `Scikit-learn`, `LightGBM`
*   **核心任务**: 二分类问题 (预测客户是否违约 `TARGET=1`)
*   **关键成果**: 在5折交叉验证中取得了 **平均 0.736 的 AUC 分数**，并识别出决定客户风险的核心特征。

---

## 1. 项目流程与核心发现

### 1.1 数据探索与清洗 (EDA & Cleaning)
*   **数据概览**: 对包含7个关联表的复杂数据集进行了分析，并对主表(`application_train`)进行了5%的抽样以适应本地计算。
*   **异常值处理**: 识别并妥善处理了 `DAYS_EMPLOYED` 特征中的明显异常值(365243)。通过对比分析发现，该异常值代表一个风险显著更低的特定客群（如退休人员），因此将其作为一个独立的布尔特征 `DAYS_EMPLOYED_ANOM`，成功将噪声转化为有效信息。
*   **关系探索**: 通过可视化分析，验证了多个核心风控假设：
    *   **年龄**: 年轻客户的违约风险显著高于年长客户。
    *   **学历与收入类型**: 学历较低、收入来源不稳定的客群（如Working Class）违约率更高。
    *   **外部征信评分 (`EXT_SOURCE`)**: 三个外部评分均与违约率呈现强负相关，是区分好坏客户的顶级特征。

<img width="990" height="588" alt="Image" src="https://github.com/user-attachments/assets/9860c879-98b6-4cab-a877-4e5eb23301f8" />

### 1.2 特征工程 (Feature Engineering)
为提升模型性能，本项目进行了两类核心的特征衍生：
*   **金融比率特征**: 基于业务理解，创造了如 `CREDIT_INCOME_PERCENT` (负债收入比), `ANNUITY_INCOME_PERCENT` (月供收入比), `CREDIT_TERM` (还款期限) 等强业务相关性特征。
*   **多项式特征**: 对最重要的 `EXT_SOURCE` 系列和 `DAYS_BIRTH` 特征进行多项式组合，以捕捉它们之间可能存在的复杂非线性关系。

### 1.3 建模与评估 (Modeling & Evaluation)
*   **模型选择**: 选用业界领先的 **LightGBM** 模型，以其高效率和高精度应对海量特征。
*   **验证策略**: 采用 **5折交叉验证 (5-fold Cross-Validation)**，确保模型性能评估的稳健性和可靠性。
*   **模型性能**: 最终在交叉验证中取得了 **平均 AUC = 0.736** 的优异成绩。

<img width="1547" height="722" alt="Image" src="https://github.com/user-attachments/assets/a6d7e106-1e89-4150-81c7-88103637cbf0" />

## 2. 关键洞察与商业价值
模型的特征重要性排序揭示了决定信用风险的关键因素：
1.  **还款能力与压力 (`CREDIT_TERM`, `ANNUITY_INCOME_PERCENT`)**: 客户的债务负担是其风险的核心。
2.  **客户稳定性 (`DAYS_EMPLOYED_PERCENT`, `DAYS_REGISTRATION`)**: 客户工作与身份的稳定性是重要的风险缓释因子。
3.  **外部数据源 (`EXT_SOURCE`系列)**: 权威的外部征信数据具有不可替代的预测价值。

这些洞察可以直接应用于金融机构的贷前审批策略优化，通过量化模型赋能，实现更精准的风险识别与差异化定价。

## 3. 如何运行
1.  克隆本仓库。
2.  确保已安装`Python 3.x`及`pip`。
3.  在项目根目录下运行 `pip install -r requirements.txt` (你需要创建一个`requirements.txt`文件)。
4.  使用Jupyter Notebook打开 `.ipynb` 文件并运行所有单元格。

## 数据来源 (Data Source)

本项目使用的数据集来自Kaggle"Home Credit Default Risk"。由于文件大小限制，数据集未包含在本仓库中。您可以从以下链接下载原始数据：

https://www.kaggle.com/c/home-credit-default-risk

