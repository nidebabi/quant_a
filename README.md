# quant_a（重构清理态）

本仓库当前仅用于“数据资产保留 + 数据管线重建准备”。

> 约束：不在当前阶段开发新业务功能，不保留旧 UI/旧服务/旧策略主流程代码。

## 主干保留项

### 1) 数据资产目录
- `data/raw/`
- `data/features/`
- `data/labels/`
- `data/metadata/`（证券基础信息、交易日历、板块/行业映射等元资产）

### 2) 有效数据脚本
- `download_data.py`
- `download_sector_data.py`
- `build_features.py`
- `build_labels.py`
- `merge_sector_into_features.py`

### 3) 必要配置模板
- `settings.py`
- `requirements.txt`
- `.env.example`
- `Dockerfile`

### 4) 少量说明文档
- `README.md`
- `data/metadata/README.md`

## 说明

后续重构将基于上述最小集合重新组织项目结构。
