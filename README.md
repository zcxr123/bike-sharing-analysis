# 共享单车大数据分析与强化学习调度项目

## 📖 项目概述

基于华盛顿特区Capital Bikeshare的真实数据，进行共享单车大数据分析和强化学习调度策略研究。项目实现了从数据生成、需求建模、环境模拟到强化学习训练的完整pipeline，并提供基线策略对比和可视化展示。

**项目特色**：
- ✅ 完整的大数据处理流程（Hadoop + Spark）
- ✅ 精确的需求预测模型（Poisson + XGBoost）
- ✅ 标准的Gym强化学习环境
- ✅ 3种基线调度策略（Zero-Action, Proportional, Min-Cost）
- ✅ 全面的评估框架和可视化

---

## 🛠️ 技术栈

### 数据处理
- **大数据平台**: Hadoop HDFS, Spark (PySpark)
- **数据分析**: Pandas, NumPy

### 建模与仿真
- **需求预测**: Poisson回归, XGBoost
- **环境模拟**: Gymnasium (OpenAI Gym)
- **调度策略**: 基线策略 + 强化学习

### 强化学习
- **算法**: PPO, DQN (计划中)
- **框架**: Stable-Baselines3 (计划中)

### 可视化
- **图表**: Matplotlib, Seaborn, Pyecharts
- **Web应用**: Flask (计划中)

### 优化算法
- **网络流**: NetworkX (Min-Cost Flow)

### 开发环境
- **系统**: WSL2 + Ubuntu 24.04
- **语言**: Python 3.12

---

## 📁 目录结构

```
bike-sharing-analysis/
├── README.md                    # 项目说明（本文件）
├── requirements.txt             # Python依赖
├── config/
│   └── env_config.yaml         # 环境配置文件 ⭐
├── data/
│   ├── raw/                    # Kaggle原始数据
│   │   ├── hour.csv           # 17,380条小时级数据
│   │   └── day.csv            # 731条天级数据
│   ├── processed/             # 处理后的数据
│   └── generated/             # 生成的模拟数据
│       ├── orders_100k.csv    # 10万条订单
│       ├── user_info_10k.csv  # 1万用户
│       └── bike_info_5k.csv   # 5千单车
├── scripts/                    # 数据生成和处理脚本
│   ├── generate_bike_data.py  # 数据生成（Day 1）
│   ├── spark_analysis.py      # Spark分析（Day 2）
│   ├── demand_modeling.py     # 需求建模（Day 2）
│   └── visualization.py       # 可视化脚本
├── simulator/                  # Gym调度环境 ⭐
│   ├── __init__.py
│   ├── bike_env.py            # 环境核心逻辑（Day 3）
│   └── demand_sampler.py      # 需求采样器（Day 3）
├── policies/                   # 基线策略模块 ⭐ NEW!
│   ├── __init__.py
│   ├── baseline_policies.py   # 3种基线策略（Day 4）
│   └── evaluate_baselines.py  # 评估框架（Day 4）
├── rl/                        # 强化学习训练（Day 7-9）
├── tests/                     # 单元测试
│   ├── test_env.py           # 环境测试（Day 3）
│   └── test_fix.py
├── results/                   # 分析和评估结果
│   ├── lambda_params.pkl      # 需求参数（Day 2）
│   ├── poisson_model.pkl      # Poisson模型（Day 2）
│   ├── xgboost_model.json     # XGBoost模型（Day 2）
│   ├── baseline_*.csv         # 基线策略结果（Day 4）
│   └── baseline_*.md          # 评估报告（Day 4）
├── web/                       # Flask可视化（Day 10-12）
└── docs/                      # 项目文档
    ├── QUICK_START.md         # 快速开始指南 ⭐
    ├── Day1_完成总结.md
    ├── Day2_完成总结.md
    ├── Day3_完成总结.md
    └── Day4_完成总结与后续计划.md
```

---

## 🚀 快速开始

### 环境准备

```bash
# 1. 安装依赖
pip install -r requirements.txt --break-system-packages

# 2. 验证环境
python3 -c "import numpy, pandas, gymnasium, networkx; print('✅ 所有依赖已安装')"
```

### 运行评估（Day 4 - 基线策略）

```bash
# 评估3种基线策略
cd ~/bike-sharing-analysis/policies
python3 evaluate_baselines.py

# 查看结果
cat ../results/baseline_evaluation_report_*.md
```

### 完整流程

#### **Step 1: 数据生成（Day 1）**

```bash
cd ~/bike-sharing-analysis/scripts
python3 generate_bike_data.py
```

生成文件：
- `data/generated/orders_100k.csv` - 10万条订单
- `data/generated/user_info_10k.csv` - 1万用户
- `data/generated/bike_info_5k.csv` - 5千单车

#### **Step 2: Spark分析（Day 2）**

```bash
# 上传到HDFS
bash scripts/upload_to_hdfs.sh

# Spark分析
python3 scripts/spark_analysis.py

# 需求建模
python3 scripts/demand_modeling.py
```

生成文件：
- `results/lambda_params.pkl` - 需求参数
- `results/poisson_model.pkl` - Poisson模型
- `results/xgboost_model.json` - XGBoost模型

#### **Step 3: 环境测试（Day 3）**

```bash
cd ~/bike-sharing-analysis/tests
python3 test_env.py
```

#### **Step 4: 策略评估（Day 4）**

```bash
cd ~/bike-sharing-analysis/policies
python3 evaluate_baselines.py
```

生成文件：
- `results/baseline_comparison_*.csv` - 策略对比
- `results/baseline_evaluation_report_*.md` - 评估报告

---

## 📊 数据说明

### 原始数据（Kaggle - Capital Bikeshare）

| 文件 | 记录数 | 时间跨度 | 说明 |
|-----|-------|---------|------|
| `hour.csv` | 17,380条 | 2011-2012 | 小时级租赁数据 |
| `day.csv` | 731条 | 2011-2012 | 天级聚合数据 |

**特征变量**：
- 时间特征：season, month, hour, weekday, workingday
- 天气特征：weather, temp, atemp, humidity, windspeed
- 目标变量：casual, registered, cnt (租赁数量)

### 生成数据（模拟）

| 文件 | 记录数 | 说明 |
|-----|-------|------|
| `orders_100k.csv` | 100,000条 | 订单明细（起点、终点、时间） |
| `user_info_10k.csv` | 10,000条 | 用户信息（类型、注册时间） |
| `bike_info_5k.csv` | 5,000条 | 单车信息（类型、状态） |

### 华盛顿特区服务区域（6个区域）

| 区域代码 | 区域名称 | 特征 | 权重 |
|---------|---------|------|------|
| A | Capitol Hill | 政府区，工作日高峰 | 25% |
| B | Downtown | 商务区，全天活跃 | 25% |
| C | Georgetown | 商业/居住混合 | 15% |
| D | Dupont Circle | 交通枢纽 | 15% |
| E | Shaw | 文化区 | 10% |
| F | Navy Yard | 海军船坞 | 10% |

---

## 🎯 核心功能

### 1. 需求预测模型（Day 2）

**模型架构**：
- **Poisson回归**: 建模λ(t)的时间依赖性
- **XGBoost**: 捕捉非线性特征交互
- **Lambda分解**: λ(t) = f(hour, season, workingday, weather)

**性能**：
- Poisson R²: 0.85+
- XGBoost R²: 0.90+

### 2. 调度环境（Day 3）

**Gym环境特性**：
- ✅ 标准Gymnasium接口
- ✅ 6个区域，800辆单车
- ✅ 动态需求采样（基于λ(t)）
- ✅ 还车机制（85%按权重，15%随机）
- ✅ 调度成本计算
- ✅ 多种奖励函数（profit, service_rate, mixed）

**状态空间**：
- 库存: [6] - 各区域单车数量（归一化）
- 时间: [1] - 当前小时（归一化）
- 环境: [3] - season, workingday, weather

**动作空间**：
- 连续: [6×6] - 调度矩阵（从i区到j区的单车数）

### 3. 基线策略（Day 4）⭐

#### **策略1: Zero-Action Policy**
- **算法**: 不进行任何调度
- **复杂度**: O(1)
- **适用**: 成本敏感场景
- **性能**: 89.94%服务率, $92k净利润（Mock环境）

#### **策略2: Proportional Refill Policy**
- **算法**: 按区域权重维持库存比例，贪心匹配
- **复杂度**: O(n²log n)
- **适用**: 快速响应库存失衡
- **性能**: 89.94%服务率, $78k净利润（Mock环境）

#### **策略3: Min-Cost Flow Policy**
- **算法**: 网络流优化，求解最小成本调度
- **复杂度**: O(n³)
- **适用**: 离线规划，理论上限
- **性能**: 需真实环境验证

**评估框架**：
- 多轮评估（5 episodes）
- 多种指标（服务率、成本、利润）
- 自动报告生成（CSV + Markdown）

---

## 📈 评估结果（截至Day 4）

### 基线策略对比（Mock环境）

| 策略 | 服务率 | 净利润 | 调度成本 |
|-----|--------|--------|---------|
| Zero-Action | 89.94% ± 2.60% | $92,116 ± $1,345 | $0 |
| Proportional-Refill | 89.94% ± 2.60% | $78,067 ± $3,503 | $14,050 ± $4,490 |
| Min-Cost-Flow | 需真实环境验证 | - | - |

**关键发现**：
1. Day 3的还车机制使Zero-Action表现优异
2. Proportional策略在Mock环境中价值未体现
3. 需要在真实环境中重新评估

---

## 📅 开发计划与进度

### 里程碑进度

```
✅ M1 阶段 (Day 1-3) - 数据与分析 【100%】
   ✅ Day 1: 环境搭建与数据生成 (10-26)
   ✅ Day 2: 需求模型与Spark分析 (10-27)  
   ✅ Day 3: 采样模块与Gym环境 (10-28)

🚀 M2 阶段 (Day 4-6) - 调度模拟器 【33%】
   ✅ Day 4: 基线策略实现 (10-29) ⭐
   ⏳ Day 5: 策略参数优化 (10-30)
   ⏳ Day 6: 多场景评估 (10-31)

⏳ M3 阶段 (Day 7-9) - RL训练 【0%】
   ⭕ Day 7: PPO算法接入 (11-01)
   ⭕ Day 8: 超参数调优 (11-02)
   ⭕ Day 9: 策略对比 (11-03)

⏳ M4 阶段 (Day 10-12) - Flask集成 【0%】
   ⭕ Day 10: Flask应用开发 (11-04)
   ⭕ Day 11: What-if仿真页面 (11-05)
   ⭕ Day 12: 文档与PPT (11-06)
```

**当前进度**: 4/12天 (33.3%)  
**状态**: 🟢 按计划推进（超预期完成）

### 详细计划

查看 `docs/共享单车数据分析与强化学习调度_项目计划_v_1.md`

---

## 📖 使用文档

### 快速上手
- **基线策略使用**: `docs/QUICK_START.md`
- **完整教程**: 各Day的完成总结文档

### 配置文件
- **环境配置**: `config/env_config.yaml`
  - 12大配置模块
  - 5种预定义场景
  - 灵活参数调整

### API文档
- **Gym环境**: `simulator/bike_env.py` (文档字符串)
- **基线策略**: `policies/baseline_policies.py` (文档字符串)
- **评估框架**: `policies/evaluate_baselines.py` (文档字符串)

---

## 🔬 技术亮点

### 1. 数据生成真实性
- 基于真实Capital Bikeshare数据
- 保留时间/天气/季节模式
- OD矩阵符合城市交通特征

### 2. 需求建模准确性
- Poisson回归 + XGBoost组合
- Lambda分解（4个维度）
- R² > 0.90

### 3. 环境模拟完整性
- 标准Gym接口
- 动态需求采样
- 还车机制（Day 3修复）
- 多种奖励函数

### 4. 策略多样性
- 3种基线策略（算法复杂度递增）
- 强化学习策略（计划中）
- 完整评估框架

### 5. 工程实践
- 模块化设计
- 配置驱动
- 完整文档
- 单元测试

---

## 🧪 测试

### 运行测试

```bash
# 环境测试
cd ~/bike-sharing-analysis/tests
python3 test_env.py

# 策略测试
cd ~/bike-sharing-analysis/policies
python3 -m pytest  # 如果有pytest
```

---

## 📦 依赖管理

### 核心依赖

```
numpy>=1.21.0         # 数值计算
pandas>=1.3.0         # 数据处理
pyyaml>=5.4.0         # 配置文件
networkx>=2.6.0       # 网络流优化
matplotlib>=3.4.0     # 可视化
gymnasium>=0.28.0     # 强化学习环境
pyspark>=3.2.0        # 大数据处理
tabulate>=0.8.0       # 表格格式化
```

查看完整依赖: `requirements.txt`

---

## 🤝 贡献指南

### 开发流程
1. Fork项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 提交Pull Request

### 代码规范
- 遵循PEP 8
- 添加类型提示
- 编写文档字符串
- 添加单元测试

---

## 📄 许可证

MIT License - 详见 `LICENSE` 文件

---

## 👨‍💻 作者

**renr**

- 项目开始: 2025-10-26
- 当前进度: Day 4/12

---

## 🙏 致谢

- **数据来源**: Capital Bikeshare (Washington D.C.)
- **数据集**: Kaggle - Bike Sharing Dataset
- **技术支持**: Claude (Anthropic)

---

## 📞 联系方式

如有问题，请：
- 查看 `docs/QUICK_START.md`
- 阅读各Day的完成总结
- 提交GitHub Issue（如果有仓库）

---

## 📚 参考资料

### 论文
- [Deep Reinforcement Learning for Bike Rebalancing]
- [Demand Forecasting in Bike-Sharing Systems]

### 文档
- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [NetworkX Documentation](https://networkx.org/)
- [PySpark Documentation](https://spark.apache.org/docs/latest/api/python/)

---

*最后更新: 2025-10-29 (Day 4完成)*  
*项目版本: v0.4.0*