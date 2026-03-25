# StockAgent A-Share AI

面向中国 A 股的 AI 量化研究与盘后决策工作台，包含：

- 历史样本训练与留出集回测
- 因子候选池筛选
- 次日高开、走强、收涨概率预测
- 国际事件、政策与产业新闻情报聚合
- 可插拔 AI 摘要与交付报告导出

## 功能结构

- `策略训练`：训练随机森林分类模型，输出准确率、召回率、AUC、特征重要性和留出集策略表现。
- `候选筛选`：从本地特征文件中按量价和板块条件生成候选池。
- `次日预测`：对候选股生成次日高开、盘中走强、收盘上涨和触达目标位的概率。
- `市场情报`：调用远程资讯接口聚合热点，并映射潜在受影响板块。
- `交付报告`：把训练结论、预测结果和情报分析导出为 Markdown 报告。

## 数据与接口

- 本地历史数据：`data/features/features_all.parquet`、`data/labels/labels_all.parquet`
- 远程行情/资讯：优先通过 AkShare 接入东方财富系与公开财经接口
- AI 摘要：兼容 OpenAI API 协议

## 本地启动

1. 安装 Python 3.11
2. 创建虚拟环境并安装依赖

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

3. 配置环境变量

```powershell
Copy-Item .env.example .env
```

4. 启动应用

```powershell
streamlit run app.py
```

默认地址：

- `http://localhost:8501`

## Docker 部署

```powershell
docker build -t stockagent-a-share-ai .
docker run --rm -p 8501:8501 --env-file .env stockagent-a-share-ai
```

## 后续可扩展方向

- 加入更丰富的分钟级特征、公告特征、北向资金与板块资金流
- 用滚动训练替代固定切分
- 引入 LightGBM / XGBoost / 时序模型做集成
- 增加任务调度，实现收盘后自动训练、自动推送与自动报告
