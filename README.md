# 共享单车大数据分析与强化学习调度项目

## 📖 项目概述

基于华盛顿特区Capital Bikeshare的真实数据，进行共享单车大数据分析和强化学习调度策略研究。项目实现了从数据生成、需求建模、环境模拟到强化学习训练的完整pipeline，并提供基线策略对比、Dashboard可视化展示和ROI分析。

**项目特色**：
- ✅ 完整的大数据处理流程（Hadoop + Spark）
- ✅ 精确的需求预测模型（Poisson + XGBoost）
- ✅ 标准的Gym强化学习环境
- ✅ 3种基线调度策略（Zero-Action, Proportional, Min-Cost）
- ✅ 强化学习策略训练与优化（PPO）
- ✅ 交互式Dashboard展示平台（Streamlit）
- ✅ 全面的评估框架和可视化

---

## 🎉 核心成果

### **突破性成果（Day 8）**

通过PPO强化学习算法的超参数优化，取得了显著的性能提升：

| 指标 | Day 7 | Day 8 | 改进 |
|-----|-------|-------|------|
| **调度成本** | $2,172/周 | $520/周 | ↓ 76% |
| **ROI** | 56.7 | 244.2 | ↑ 4.3倍 |
| **服务率** | 95.3% | 98.1% | ↑ 2.8% |
| **净利润** | $123,197/周 | $127,045/周 | ↑ 3.1% |

**年度经济效益**: $283,660（单城市）

**关键发现**:
- 🧠 **高频低成本策略**: 调度频率提升18倍，但成本仅增加14%
- 🎯 **98%的智慧**: 自动找到服务率-成本的最优平衡点
- ⏰ **预测性调度**: 在需求高峰前主动布局
- 📈 **规模效应**: 高需求期成本反而更低

### **Dashboard平台（Day 10）**

基于Streamlit开发的交互式Dashboard，包含：

1. **项目概览页** 🏠
   - 4个核心指标卡片
   - 项目进度展示
   - 关键洞察展示
   - 技术架构说明

2. **策略对比页** 📈
   - 多策略选择与对比
   - 3种交互式图表（柱状图、箱线图、散点图）
   - 详细统计分析
   - 数据下载功能

3. **ROI计算器** 💰
   - 参数化效益计算
   - 敏感性分析
   - 回本期分析
   - 多年效益展望

---

## 🛠️ 技术栈

### 数据处理
- **大数据平台**: Hadoop HDFS, Spark (PySpark)
- **数据分析**: Pandas, NumPy

### 建模与仿真
- **需求预测**: Poisson回归, XGBoost
- **环境模拟**: Gymnasium (OpenAI Gym)
- **调度策略**: 基线策略 + 强化学习（PPO）

### 强化学习
- **算法**: PPO (Proximal Policy Optimization)
- **框架**: Stable-Baselines3
- **超参数优化**: 成本感知奖励函数

### 可视化与Dashboard
- **图表**: Matplotlib, Seaborn, Plotly
- **Dashboard**: Streamlit
- **交互式分析**: 参数化计算、敏感性分析

### 优化算法
- **网络流**: NetworkX (Min-Cost Flow)

### 开发环境
- **系统**: WSL2 + Ubuntu 24.04
- **语言**: Python 3.12

---

## 📁 目录结构

```
bike-sharing-analysis/
├── README.md                    # 项目说明（本文件）⭐
├── requirements.txt             # Python依赖
├── config/
│   └── env_config.yaml         # 环境配置文件
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
│   ├── day10_prepare_data.py  # Dashboard数据准备（Day 10）⭐
│   └── visualization.py       # 可视化脚本
├── simulator/                  # Gym调度环境
│   ├── __init__.py
│   ├── bike_env.py            # 环境核心逻辑（Day 3）
│   └── demand_sampler.py      # 需求采样器（Day 3）
├── policies/                   # 基线策略模块
│   ├── __init__.py
│   ├── baseline_policies.py   # 3种基线策略（Day 4）
│   └── evaluate_baselines.py  # 评估框架（Day 4）
├── rl/                        # 强化学习训练（Day 7-9）
│   ├── train_ppo.py          # PPO训练
│   └── evaluate_policy.py    # 策略评估
├── dashboard/                  # Dashboard应用（Day 10）⭐ NEW!
│   ├── app.py                # 主应用（项目概览）
│   ├── pages/                # 多页面
│   │   ├── 2_📈_策略对比.py
│   │   ├── 3_🔍_决策分析.py
│   │   └── 4_💰_ROI计算器.py
│   ├── data/                 # Dashboard数据
│   │   ├── comparison.csv
│   │   ├── decisions.csv
│   │   └── summary.pkl
│   └── config.json           # Dashboard配置
├── tests/                     # 单元测试
│   ├── test_env.py           # 环境测试（Day 3）
│   └── test_fix.py
├── results/                   # 分析和评估结果
│   ├── lambda_params.pkl      # 需求参数（Day 2）
│   ├── poisson_model.pkl      # Poisson模型（Day 2）
│   ├── xgboost_model.json     # XGBoost模型（Day 2）
│   ├── baseline_*.csv         # 基线策略结果（Day 4）
│   ├── day8_comparison/       # Day 8对比数据⭐
│   ├── day9_analysis/         # Day 9决策分析⭐
│   ├── day9_visualizations/   # Day 9可视化图表⭐
│   └── day9_reports/          # Day 9业务/技术报告⭐
└── docs/                      # 项目文档
    ├── QUICK_START.md         # 快速开始指南
    ├── Day1_完成总结.md
    ├── Day2_完成总结.md
    ├── Day3_完成总结.md
    ├── Day4_完成总结与后续计划.md
    ├── Day9_Quick-Start.md
    └── Day10_完成总结与后续计划.md  ⭐ NEW!
```

---

## 🚀 快速开始

### 方式A: 直接体验Dashboard（推荐）

```bash
# 1. 启动Dashboard
cd ~/bike-sharing-analysis/dashboard
streamlit run app.py

# 2. 浏览器访问
# http://localhost:8501

# 3. 探索功能
# - 查看项目概览和核心成果
# - 对比不同策略的性能
# - 使用ROI计算器评估经济效益
```

### 方式B: 完整流程体验

#### **环境准备**

```bash
# 1. 安装依赖
pip install -r requirements.txt --break-system-packages

# 或使用虚拟环境
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. 验证环境
python3 -c "import numpy, pandas, gymnasium, networkx, streamlit; print('✅ 所有依赖已安装')"
```

#### **Step 1: 数据生成（Day 1）**

```bash
cd ~/bike-sharing-analysis/scripts
python3 generate_bike_data.py
```

生成文件：
- `data/generated/orders_100k.csv` - 10万条订单
- `data/generated/user_info_10k.csv` - 1万用户
- `data/generated/bike_info_5k.csv` - 5千单车

#### **Step 2: 需求建模（Day 2）**

```bash
# 上传到HDFS（如果使用Spark）
bash scripts/upload_to_hdfs.sh

# Spark分析
python3 scripts/spark_analysis.py

# 需求建模
python3 scripts/demand_modeling.py
```

#### **Step 3: 环境测试（Day 3）**

```bash
cd ~/bike-sharing-analysis/tests
python3 test_env.py
```

#### **Step 4: 基线策略评估（Day 4）**

```bash
cd ~/bike-sharing-analysis/policies
python3 evaluate_baselines.py
```

#### **Step 5: 强化学习训练（Day 7-8）**

```bash
cd ~/bike-sharing-analysis/rl

# Day 7: 初始训练
python3 train_ppo.py --cost-weight 1.0

# Day 8: 超参数优化
python3 train_ppo.py --cost-weight 2.0
```

#### **Step 6: 决策分析与可视化（Day 9）**

```bash
cd ~/bike-sharing-analysis

# 决策分析
python3 scripts/day9_analyze_decisions.py

# 生成可视化图表
python3 scripts/day9_generate_plots.py

# 生成业务报告
python3 scripts/day9_generate_reports.py
```

#### **Step 7: Dashboard准备与启动（Day 10）**

```bash
# 准备Dashboard数据
python3 scripts/day10_prepare_data.py

# 启动Dashboard
cd dashboard
streamlit run app.py
```

---

## 📊 核心功能详解

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

### 3. 基线策略（Day 4）

#### **策略1: Zero-Action Policy**
- **算法**: 不进行任何调度
- **复杂度**: O(1)
- **适用**: 成本敏感场景

#### **策略2: Proportional Refill Policy**
- **算法**: 按区域权重维持库存比例，贪心匹配
- **复杂度**: O(n²log n)
- **适用**: 快速响应库存失衡

#### **策略3: Min-Cost Flow Policy**
- **算法**: 网络流优化，求解最小成本调度
- **复杂度**: O(n³)
- **适用**: 离线规划，理论上限

### 4. 强化学习策略（Day 7-8）

**PPO算法配置**：
- 学习率: 3e-4
- 批次大小: 64
- Epoch数: 10
- Gamma: 0.99
- **成本权重**: 1.0 → 2.0（关键优化）

**Day 8突破**：
- 通过调整cost_weight从1.0到2.0
- 成本降低76%，ROI提升4.3倍
- 发现高频低成本创新策略
- 自动学会98%的最优平衡点

### 5. Dashboard平台（Day 10）⭐

#### **页面1: 项目概览**
**功能**：
- 核心指标展示（成本降低、ROI提升、年度效益、服务率）
- 项目进度追踪
- 关键洞察卡片（可折叠）
- 技术架构说明
- 快速导航链接

**技术**：
- Streamlit多列布局
- 自定义CSS样式
- 动态指标卡片
- 进度条组件

#### **页面2: 策略对比**
**功能**：
- 多策略选择（多选框）
- 场景筛选（全部/单个）
- 指标选择（服务率/利润/成本）
- 3种交互式图表：
  - 柱状图（平均值对比）
  - 箱线图（分布分析）
  - 散点图（成本-服务率权衡）
- 详细统计表格
- 关键洞察自动生成
- 数据下载（CSV）

**技术**：
- Plotly交互式图表
- Pandas数据分组聚合
- 动态筛选与更新
- 统计分析计算

#### **页面3: 决策分析**（可选）
**功能**：
- 时间模式分析（每小时调度频率）
- 成本分布分析
- 调度效率分析
- 原始数据查看

**适用场景**：
- 深入理解PPO决策机制
- 识别调度模式和规律
- 优化策略改进

#### **页面4: ROI计算器**
**功能**：
- 参数调整（城市数量、周需求量）
- 实时效益计算：
  - 周/年/5年效益
  - 投资回报率
  - 回本周期
- 敏感性分析：
  - 城市数量影响曲线
  - 需求量影响曲线
- 回本期分析（累计净效益曲线）
- 多年效益展望（柱状图）
- 下载分析报告

**技术**：
- 滑块组件（参数输入）
- 实时计算与更新
- Plotly多种图表类型
- 场景化效益模型

---

## 📈 评估结果

### Day 8核心成果对比

| 策略 | 服务率 | 调度成本 | 净利润 | ROI |
|-----|--------|---------|--------|-----|
| Baseline (Proportional) | 99.95% | $3,890/周 | $120,350/周 | 30.9 |
| PPO-Day7 | 95.32% | $2,172/周 | $123,197/周 | 56.7 |
| **PPO-Day8** | **98.12%** | **$520/周** | **$127,045/周** | **244.2** |

**改进幅度**：
- 成本: Day7 → Day8 降低 76%
- ROI: Day7 → Day8 提升 4.3倍
- 服务率: Day7 → Day8 提升 2.8%

### 经济效益分析（单城市）

| 时间周期 | 成本节省 | 利润增加 | 总效益 |
|---------|---------|---------|--------|
| 周 | $1,652 | $3,848 | $5,500 |
| 年 | $85,904 | $200,096 | $286,000 |
| 5年 | $429,520 | $1,000,480 | $1,430,000 |

**规模效应**（10个城市）：
- 年效益: $2.86M
- 5年效益: $14.3M
- ROI: 567%

---

## 📅 开发计划与进度

### 里程碑进度

```
✅ M1 阶段 (Day 1-3) - 数据与分析 【100%】
   ✅ Day 1: 环境搭建与数据生成 (10-26)
   ✅ Day 2: 需求模型与Spark分析 (10-27)  
   ✅ Day 3: 采样模块与Gym环境 (10-28)

✅ M2 阶段 (Day 4-6) - 调度模拟器 【100%】
   ✅ Day 4: 基线策略实现 (10-29)
   ✅ Day 5: 策略参数优化
   ✅ Day 6: 多场景评估

✅ M3 阶段 (Day 7-9) - RL训练 【100%】
   ✅ Day 7: PPO算法接入与初始训练
   ✅ Day 8: 超参数调优（突破性成果）
   ✅ Day 9: 决策分析与可视化

🚀 M4 阶段 (Day 10-12) - 项目收尾 【33%】
   ✅ Day 10: Dashboard开发 (10-29) ⭐
   ⏳ Day 11: 文档完善与优化 (10-30)
   ⏳ Day 12: PPT与最终交付 (10-31)
```

**当前进度**: 10/12天 (83.3%)  
**状态**: 🟢 按计划推进

### Day 10完成情况

**已完成** ✅：
- [x] Dashboard主应用（项目概览页）
- [x] 策略对比页面（交互式图表）
- [x] ROI计算器页面（效益分析）
- [x] 决策分析页面（可选）
- [x] 数据准备脚本
- [x] Dashboard部署与测试

**成果**：
- 3个核心页面，功能完整
- 基于Streamlit的专业界面
- 交互式图表和参数调整
- 实时计算和敏感性分析

### 剩余任务（Day 11-12）

**Day 11: 文档完善** ⏳
- [ ] 更新README（本文件）✅
- [ ] 创建Day 10完成总结
- [ ] 完善API文档
- [ ] 更新快速开始指南
- [ ] 项目归档整理

**Day 12: PPT与交付** ⏳
- [ ] 制作演示PPT（10-15页）
- [ ] 准备演示视频（可选）
- [ ] 最终测试与验收
- [ ] 打包交付

---

## 📖 使用文档

### 快速上手
- **Dashboard使用**: 启动后浏览器访问 http://localhost:8501
- **基线策略使用**: `docs/QUICK_START.md`
- **Day 9快速启动**: `docs/Day9_Quick-Start.md`
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

### 4. 策略多样性与创新
- 3种基线策略（算法复杂度递增）
- PPO强化学习策略
- 成本感知奖励函数（Day 8创新）
- 高频低成本策略发现

### 5. Dashboard可视化
- 响应式布局设计
- 交互式图表（Plotly）
- 参数化计算
- 实时敏感性分析
- 专业配色与样式

### 6. 工程实践
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

# Dashboard测试
cd ~/bike-sharing-analysis/dashboard
streamlit run app.py
# 在浏览器中手动测试各页面功能
```

---

## 📦 依赖管理

### 核心依赖

```
# 数值计算与数据处理
numpy>=1.21.0
pandas>=1.3.0
scipy>=1.7.0

# 强化学习
gymnasium>=0.28.0
stable-baselines3>=2.0.0

# 可视化
matplotlib>=3.4.0
seaborn>=0.11.0
plotly>=5.0.0

# Dashboard
streamlit>=1.28.0

# 其他
networkx>=2.6.0
pyyaml>=5.4.0
tabulate>=0.8.0
```

查看完整依赖: `requirements.txt`

### 安装说明

```bash
# 标准安装
pip install -r requirements.txt

# WSL/Linux环境
pip install -r requirements.txt --break-system-packages

# 使用虚拟环境（推荐）
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## 🎯 使用场景

### 1. 数据分析师
- 使用Dashboard查看项目成果
- 分析不同策略的性能
- 下载数据进行二次分析

### 2. 业务决策者
- 使用ROI计算器评估投资回报
- 查看经济效益分析
- 制定部署计划

### 3. 技术团队
- 研究PPO决策机制
- 优化调度策略
- 复现训练过程

### 4. 学术研究
- 强化学习算法研究
- 共享单车调度问题
- 需求预测建模

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
- 当前进度: Day 10/12
- Dashboard完成: 2025-10-29

---

## 🙏 致谢

- **数据来源**: Capital Bikeshare (Washington D.C.)
- **数据集**: Kaggle - Bike Sharing Dataset
- **技术支持**: Claude (Anthropic)
- **框架**: Stable-Baselines3, Streamlit, Plotly

---

## 📞 联系方式

如有问题，请：
- 查看 `docs/` 目录下的完整文档
- 阅读各Day的完成总结
- 启动Dashboard体验功能
- 提交GitHub Issue（如果有仓库）

---

## 📚 参考资料

### 论文
- Deep Reinforcement Learning for Bike Rebalancing
- Demand Forecasting in Bike-Sharing Systems
- Proximal Policy Optimization Algorithms

### 文档
- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [Stable-Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Plotly Documentation](https://plotly.com/python/)
- [NetworkX Documentation](https://networkx.org/)

### 相关项目
- OpenAI Gym
- Capital Bikeshare Open Data

---

## 🔮 未来展望

### 短期计划（Day 11-12）
- [ ] 完善项目文档
- [ ] 制作演示PPT
- [ ] 项目归档与交付

### 中期扩展
- [ ] 更多RL算法（DQN, A2C）
- [ ] Dashboard增加更多页面
- [ ] 实时数据接入
- [ ] 移动端适配

### 长期规划
- [ ] 多城市联合调度
- [ ] 真实路网集成
- [ ] 时序预测升级（Prophet, TFT）
- [ ] Offline RL探索
- [ ] Docker化部署

---

*最后更新: 2025-10-29 (Day 10完成)*  
*项目版本: v1.0*  
*Dashboard版本: v1.0*