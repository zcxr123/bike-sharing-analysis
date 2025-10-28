# 共享单车调度 - 基线策略快速开始指南

> 📅 **Day 4 交付物** - 基线策略实现与评估  
> 📝 **作者**: renr  
> 🗓️ **日期**: 2025-10-29

---

## 📚 目录

1. [项目概述](#项目概述)
2. [文件结构](#文件结构)
3. [快速开始](#快速开始)
4. [策略介绍](#策略介绍)
5. [使用示例](#使用示例)
6. [配置说明](#配置说明)
7. [常见问题](#常见问题)

---

## 项目概述

本项目实现了3种共享单车调度基线策略，用于与强化学习策略进行对比评估：

| 策略名称 | 描述 | 优点 | 缺点 |
|---------|------|------|------|
| **Zero-Action** | 不调度策略 | 零成本，简单 | 无法优化库存 |
| **Proportional Refill** | 按比例补货 | 维持平衡，适应性强 | 未考虑需求预测 |
| **Min-Cost Flow** | 最小成本流 | 理论最优，全局优化 | 计算复杂度高 |

---

## 文件结构

```
bike-sharing-analysis/
├── config/
│   └── env_config.yaml              # 环境配置文件 ⭐
├── policies/
│   ├── baseline_policies.py         # 基线策略实现 (690行) ⭐
│   └── evaluate_baselines.py        # 评估脚本 (600行) ⭐
├── simulator/
│   ├── demand_sampler.py            # 需求采样器 (Day 3)
│   └── bike_env.py                  # Gym环境 (Day 3)
├── results/
│   ├── lambda_params.pkl            # 需求参数 (Day 2)
│   ├── baseline_comparison_*.csv    # 策略对比结果
│   ├── baseline_detail_*.csv        # 详细评估数据
│   ├── baseline_summary_*.csv       # 汇总统计
│   └── baseline_evaluation_report_*.md  # 评估报告
└── tests/
    └── test_env.py                  # 环境测试 (Day 3)
```

---

## 快速开始

### 1. 环境准备

确保已安装依赖：

```bash
pip install numpy pandas pyyaml networkx --break-system-packages
```

### 2. 运行评估

**方法1: 使用真实环境（推荐）**

如果您已完成Day 1-3的任务，在WSL环境中运行：

```bash
cd ~/bike-sharing-analysis
python3 policies/evaluate_baselines.py
```

**方法2: 测试模式**

如果环境模块不可用，脚本会自动使用Mock环境：

```bash
cd /path/to/bike-sharing-analysis
python3 policies/evaluate_baselines.py
# 输出: ⚠️ 使用模拟环境进行测试
```

### 3. 查看结果

评估完成后，结果文件自动保存到 `results/` 目录：

```bash
# 对比表格
cat results/baseline_comparison_*.csv

# 评估报告
cat results/baseline_evaluation_report_*.md
```

---

## 策略介绍

### 🔵 策略1: Zero-Action Policy

**算法描述**:
- 不进行任何调度操作
- 依靠自然流动和还车机制维持库存

**使用场景**:
- 作为最简单的基准
- 评估"无干预"的表现
- 了解调度的价值

**代码示例**:
```python
from policies.baseline_policies import ZeroActionPolicy

policy = ZeroActionPolicy(config)
action = policy.select_action(observation)
# action = 零矩阵 (num_zones × num_zones)
```

### 🟢 策略2: Proportional Refill Policy

**算法描述**:
1. 根据区域权重计算目标库存
2. 识别富余区（库存 > 目标）和缺口区（库存 < 目标）
3. 按成本从低到高贪心匹配调度

**数学模型**:
```
target_inventory[z] = zone_weight[z] × total_inventory
surplus[z] = max(0, current_inventory[z] - target_inventory[z])
deficit[z] = max(0, target_inventory[z] - current_inventory[z])
```

**参数**:
- `threshold`: 触发调度的阈值（default: 0.1）
- `rebalance_ratio`: 每次调度的比例（default: 0.5）

**代码示例**:
```python
from policies.baseline_policies import ProportionalRefillPolicy

policy = ProportionalRefillPolicy(
    config,
    threshold=0.1,        # 偏差>10%时触发
    rebalance_ratio=0.5   # 调度50%的缺口/富余
)
action = policy.select_action(observation)
```

### 🟣 策略3: Min-Cost Flow Policy

**算法描述**:
1. 将调度问题建模为网络流
2. 使用NetworkX求解最小成本最大流
3. 在满足需求前提下最小化调度成本

**数学模型**:
```
Minimize: Σ cost[i,j] × flow[i,j]
Subject to:
  Σ flow[i,j] - Σ flow[j,i] = supply[i]    (富余区)
  Σ flow[i,j] - Σ flow[j,i] = -demand[i]   (缺口区)
  0 ≤ flow[i,j] ≤ capacity[i,j]
```

**参数**:
- `threshold`: 触发调度的阈值（default: 0.15）
- `use_expected_demand`: 是否使用期望需求（default: False）

**代码示例**:
```python
from policies.baseline_policies import MinCostFlowPolicy

policy = MinCostFlowPolicy(
    config,
    threshold=0.15,
    use_expected_demand=False
)
action = policy.select_action(observation)
```

---

## 使用示例

### 示例1: 单个策略评估

```python
from policies.baseline_policies import create_policy
from policies.evaluate_baselines import load_config, evaluate_policy
from simulator.bike_env import BikeRebalancingEnv

# 加载配置
config = load_config('config/env_config.yaml')

# 创建环境
env = BikeRebalancingEnv(config)

# 创建策略
policy = create_policy('proportional', config)

# 评估策略
results = evaluate_policy(policy, env, num_episodes=10)

print(f"平均服务率: {results['mean_service_rate']*100:.2f}%")
print(f"平均净利润: ${results['mean_net_profit']:.2f}")
```

### 示例2: 多策略对比

```python
from policies.evaluate_baselines import compare_policies, load_config

# 加载配置
config = load_config()

# 对比所有策略
comparison_df, results = compare_policies(
    config=config,
    policy_names=['zero', 'proportional', 'mincost'],
    num_episodes=5,
    scenario='sunny_weekday'
)

# 查看对比结果
print(comparison_df)
```

### 示例3: 多场景评估

```python
from policies.evaluate_baselines import compare_policies, load_config

config = load_config()

scenarios = ['sunny_weekday', 'rainy_weekend', 'summer_peak', 'winter_low']

for scenario in scenarios:
    print(f"\n{'='*60}")
    print(f"评估场景: {scenario}")
    print(f"{'='*60}")
    
    comparison_df, results = compare_policies(
        config=config,
        policy_names=['zero', 'proportional'],
        num_episodes=3,
        scenario=scenario
    )
```

### 示例4: 自定义策略参数

```python
from policies.baseline_policies import ProportionalRefillPolicy

# 创建多个变体
variants = [
    ('保守', 0.15, 0.3),   # 高阈值，低调度比例
    ('标准', 0.10, 0.5),   # 默认参数
    ('激进', 0.05, 0.7)    # 低阈值，高调度比例
]

for name, threshold, ratio in variants:
    policy = ProportionalRefillPolicy(
        config,
        threshold=threshold,
        rebalance_ratio=ratio
    )
    print(f"{name}策略 (threshold={threshold}, ratio={ratio})")
    # ... 评估 ...
```

---

## 配置说明

### 环境配置 (`config/env_config.yaml`)

关键配置项：

```yaml
# 区域配置
zones:
  num_zones: 6
  zone_weights: [0.25, 0.25, 0.15, 0.15, 0.10, 0.10]
  zone_capacity: [200, 200, 120, 120, 80, 80]

# 经济参数
economics:
  revenue_per_trip: 4.0
  penalty_per_unmet: 2.0
  rebalance_budget: 500.0
  max_rebalance_qty: 50
  cost_matrix:  # 6×6 调度成本矩阵
    - [0.0, 1.5, 2.0, ...]
    - [1.5, 0.0, 2.5, ...]
    - ...

# 评估配置
evaluation:
  num_episodes: 5
  random_seeds: [42, 43, 44, 45, 46]
  metrics:
    - service_rate
    - unmet_demand
    - total_revenue
    - net_profit
```

### 修改配置

**方法1: 编辑YAML文件**

```bash
nano config/env_config.yaml
# 修改参数后保存
```

**方法2: 代码中覆盖**

```python
config = load_config()
config['economics']['rebalance_budget'] = 1000.0  # 增加预算
config['evaluation']['num_episodes'] = 10          # 增加评估轮数
```

---

## 常见问题

### Q1: 如何添加新策略？

**A**: 继承 `BasePolicy` 类并实现 `select_action` 方法：

```python
from policies.baseline_policies import BasePolicy

class MyPolicy(BasePolicy):
    def select_action(self, observation):
        # 实现你的策略逻辑
        action = ...
        return action

# 在factory中注册
def create_policy(policy_name, config, **kwargs):
    # ...
    elif policy_name == 'my':
        return MyPolicy(config, **kwargs)
```

### Q2: 评估报告在哪里？

**A**: 报告自动保存到 `results/` 目录：

```bash
# 最新报告
ls -t results/baseline_evaluation_report_*.md | head -1

# 查看报告
cat results/baseline_evaluation_report_default_*.md
```

### Q3: 如何修改评估指标？

**A**: 修改配置文件中的 `evaluation.metrics`:

```yaml
evaluation:
  metrics:
    - service_rate           # 服务率
    - unmet_demand          # 未满足需求
    - total_revenue         # 总收入
    - total_cost            # 调度成本
    - net_profit            # 净利润
    - avg_inventory         # 平均库存
    - inventory_std         # 库存标准差
    - custom_metric         # 自定义指标（需要在代码中实现）
```

### Q4: Min-Cost Flow警告"infeasible"怎么办？

**A**: 这通常是因为：
1. 网络流问题无可行解（供需不平衡）
2. 阈值设置过严格

**解决方法**:
- 降低 `threshold` 参数（如从0.15→0.10）
- 检查环境中的库存是否合理
- 查看是否有容量约束冲突

### Q5: 如何并行评估多个策略？

**A**: 可以使用 `multiprocessing`:

```python
from multiprocessing import Pool

def eval_wrapper(args):
    policy_name, config, num_episodes = args
    # ... 评估逻辑 ...
    return results

with Pool(3) as p:
    args = [
        ('zero', config, 5),
        ('proportional', config, 5),
        ('mincost', config, 5)
    ]
    results = p.map(eval_wrapper, args)
```

### Q6: 如何可视化评估结果？

**A**: 使用 pandas 和 matplotlib:

```python
import pandas as pd
import matplotlib.pyplot as plt

# 读取结果
df = pd.read_csv('results/baseline_comparison_*.csv')

# 绘制对比图
df.plot(x='策略', y='服务率(%)', kind='bar')
plt.title('策略服务率对比')
plt.show()
```

---

## 下一步

完成基线策略评估后，下一步工作（Day 5-6）：

1. **优化基线策略参数** - 调整threshold和ratio
2. **扩展评估场景** - 测试更多天气/季节组合
3. **深入分析** - 分析策略优劣和适用场景
4. **准备RL训练** - 建立benchmark，为Day 7 PPO训练做准备

---

## 技术支持

- 📧 **问题反馈**: 提交Issue到项目仓库
- 📖 **详细文档**: 查看 `README.md`
- 💬 **讨论交流**: 加入项目讨论组

---

*本指南由 Day 4 任务生成*  
*更新日期: 2025-10-29*  
*项目: 共享单车数据分析与强化学习调度*
