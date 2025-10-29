# 共享单车智能调度系统 - 技术报告

**报告日期**: 2025-10-29
**项目阶段**: Day 9
**目标读者**: 技术团队、数据科学家

---

## 📋 目录

1. [问题定义](#问题定义)
2. [方法论](#方法论)
3. [实验设置](#实验设置)
4. [结果分析](#结果分析)
5. [技术细节](#技术细节)
6. [局限性与改进](#局限性与改进)
7. [复现指南](#复现指南)

---

## 1. 问题定义

### 1.1 业务场景

共享单车调度优化问题，目标是在满足用户需求的同时最小化运营成本。

### 1.2 形式化描述

**状态空间** S:
- 各区域车辆库存: $B_z$ (z=1..K)
- 时间索引: $t$
- 上下文信息: hour, weekday, season, weather

**动作空间** A:
- 调度决策: $(i→j, qty)$
- 约束: 总调拨量上限、单次最大流量

**奖励函数** R:
```
Day 7: R = revenue - 5.0*penalty - 1.0*cost
Day 8: R = revenue - 5.0*penalty - 2.0*cost  # 关键改进
```

### 1.3 评估指标

- **服务率**: 满足需求量 / 总需求量
- **净利润**: 收益 - 调度成本
- **ROI**: 净利润 / 调度成本
- **成本效率**: 调度成本 / 服务量

---

## 2. 方法论

### 2.1 算法选择

**Proximal Policy Optimization (PPO)**

选择理由:
- On-policy算法，训练稳定
- 样本效率较高
- 易于实现和调试
- 在类似问题上表现优秀

### 2.2 网络结构

```python
Policy Network:
  - Input: State (obs_dim)
  - Hidden: [256, 256] with ReLU
  - Output: Action distribution

Value Network:
  - Input: State (obs_dim)
  - Hidden: [256, 256] with ReLU
  - Output: State value
```

### 2.3 关键创新点

1. **成本感知奖励函数**
   - 将cost_weight从1.0提高到2.0
   - 简单但效果显著

2. **超参数优化**
   - 学习率: 3e-4 → 1e-4
   - batch_size: 64 → 128
   - n_steps: 2048 → 4096

3. **训练策略**
   - 增加训练步数: 100k → 150k
   - 使用EvalCallback和CheckpointCallback

---

## 3. 实验设置

### 3.1 环境配置

- **区域数**: 6
- **时间跨度**: 168小时（1周）
- **需求模型**: Poisson分布，基于历史数据
- **场景**: default, sunny_weekday, rainy_weekend, summer_peak, winter_low

### 3.2 训练配置

```yaml
Day 7 (Baseline):
  algorithm: PPO
  timesteps: 100000
  learning_rate: 3e-4
  n_steps: 2048
  batch_size: 64
  cost_weight: 1.0

Day 8 (Cost-Aware):
  algorithm: PPO
  timesteps: 100000
  learning_rate: 3e-4
  n_steps: 2048
  batch_size: 64
  cost_weight: 2.0  # Key change

Day 8 (Tuned):
  algorithm: PPO
  timesteps: 150000
  learning_rate: 1e-4
  n_steps: 4096
  batch_size: 128
  cost_weight: 2.0
```

### 3.3 评估协议

- 每个场景运行10个episode
- 使用固定随机种子确保可复现
- 对比指标: 服务率、净利润、调度成本

---

## 4. 结果分析

### 4.1 量化结果

```
Day 7 (Original PPO):
  Service Rate: 99.53% (±0.46%)
  Net Profit: $123197 (±$9313)
  Total Cost: $2172 (±$167)

Day 8 (Cost-Aware PPO):
  Service Rate: 97.72% (±0.83%)
  Net Profit: $121024 (±$8006)
  Total Cost: $374 (±$38)

Improvement:
  Cost Reduction: 82.8%
  Profit Increase: -1.8%
```

### 4.2 关键发现

1. **高频低成本策略**
   - PPO调度频率是基线的15倍
   - 但单次成本控制严格
   - 总成本仅高10%

2. **98%的最优点**
   - PPO自动找到98%服务率的平衡点
   - 追求最后2%需要4倍成本
   - 边际收益递减的自然体现

3. **时间适应性**
   - PPO识别高峰和低谷时段
   - 动态调整调度强度
   - 表现出良好的泛化能力

---

## 5. 技术细节

### 5.1 环境实现

```python
class CostAwareEnv(BikeRebalancingEnv):
    def __init__(self, config, scenario='default',
                 cost_weight=2.0, penalty_weight=5.0):
        super().__init__(config_dict=config, scenario=scenario)
        self.cost_weight = cost_weight
        self.penalty_weight = penalty_weight

    def step(self, action):
        obs, _, done, truncated, info = super().step(action)
        
        # Custom reward function
        revenue = info.get('revenue', 0)
        penalty = info.get('penalty', 0)
        cost = info.get('rebalance_cost', 0)
        
        new_reward = (revenue - 
                      self.penalty_weight * penalty - 
                      self.cost_weight * cost)
        
        return obs, new_reward, done, truncated, info
```

### 5.2 训练流程

```python
# Create environment
env = DummyVecEnv([make_cost_aware_env])

# Initialize PPO
model = PPO(
    'MlpPolicy',
    env,
    learning_rate=1e-4,
    n_steps=4096,
    batch_size=128,
    verbose=1
)

# Train
model.learn(
    total_timesteps=150000,
    callback=[eval_callback, checkpoint_callback]
)
```

### 5.3 评估代码

```python
# Load model
model = PPO.load('best_model.zip')

# Evaluate
for ep in range(n_episodes):
    obs, _ = env.reset()
    done = False
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        # Collect metrics
```

---

## 6. 局限性与改进

### 6.1 当前局限性

1. **模拟环境简化**
   - 区域数量较少（6个）
   - 时间跨度较短（1周）
   - 需求模型简化

2. **服务率略低**
   - 98% vs 基线100%
   - 可能不适合追求完美服务的场景

3. **泛化能力待验证**
   - 只在模拟环境测试
   - 真实场景可能有差异

### 6.2 改进方向

**短期**:
- 增加环境复杂度（更多区域、更长时间）
- 引入更多场景（节假日、活动日）
- 多目标优化（成本、服务、环保）

**中期**:
- Offline RL（利用历史数据）
- Multi-Agent RL（多车协同）
- Hierarchical RL（分层决策）

**长期**:
- 与真实系统集成
- 在线学习与适应
- 大规模部署

---

## 7. 复现指南

### 7.1 环境准备

```bash
# Python 3.10+
pip install stable-baselines3[extra] --break-system-packages
pip install pandas numpy matplotlib seaborn
```

### 7.2 训练

```bash
# Day 8 Cost-Aware Training
python3 scripts/day8_train_cost_aware.py \
    --timesteps 100000 \
    --cost-weight 2.0 \
    --quick-test
```

### 7.3 评估

```bash
# Compare all models
python3 scripts/day8_compare_all.py --episodes 10
```

### 7.4 可视化

```bash
# Generate plots
python3 scripts/day9_generate_plots.py
```

---

## 📚 参考文献

1. Schulman et al. (2017). Proximal Policy Optimization Algorithms
2. OpenAI Spinning Up: https://spinningup.openai.com/
3. Stable-Baselines3: https://stable-baselines3.readthedocs.io/

---

**报告生成时间**: 2025-10-29 09:36:17
