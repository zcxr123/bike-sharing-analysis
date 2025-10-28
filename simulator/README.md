# 共享单车调度模拟器 - Gym环境

基于Gymnasium框架的共享单车调度强化学习环境

## 📁 项目结构

```
bike-sharing-rl/
├── config/
│   └── env_config.yaml          # 环境配置文件 ⭐
├── simulator/
│   ├── demand_sampler.py        # 需求采样模块 ⭐
│   └── bike_env.py              # Gym环境实现 ⭐
├── tests/
│   └── test_env.py              # 测试脚本 ⭐
└── README.md                    # 本文档
```

## 🎯 核心模块说明

### 1. **环境配置 (`env_config.yaml`)**

完整的YAML配置文件，包含：

- **区域配置**: 6个区域，权重，容量
- **时间配置**: 时间跨度，步长，调度频率
- **库存配置**: 总数量，初始分布策略
- **需求配置**: lambda参数路径，放大系数
- **经济参数**: 收益，惩罚，成本矩阵，预算
- **奖励配置**: 奖励类型，归一化，折扣因子
- **场景配置**: 预定义场景（晴天/雨天/工作日/周末等）

### 2. **需求采样器 (`demand_sampler.py`)**

基于Day2生成的`lambda_params.pkl`进行需求采样。

**主要功能**:
- `get_lambda_t()`: 根据时间、天气、季节等计算需求强度
- `sample_demand()`: 泊松采样各区域需求
- `sample_batch_demands()`: 批量采样（向量化）
- `get_expected_demand()`: 获取期望需求（不采样）
- `get_demand_statistics()`: 统计信息

**输入参数**:
- hour: 小时 (0-23)
- season: 季节 (1-4)
- workingday: 是否工作日 (0/1)
- weather: 天气 (1-4)

**输出**:
- demands: 各区域需求数组 `(num_zones,)`

### 3. **Gym环境 (`bike_env.py`)**

标准Gymnasium环境实现。

**状态空间** (`observation_space`):
```python
{
    'inventory': Box(shape=(6,)),    # 各区库存
    'hour': Box(shape=(1,)),         # 当前小时
    'season': Discrete(4),           # 季节
    'workingday': Discrete(2),       # 工作日
    'weather': Discrete(4)           # 天气
}
```

**动作空间** (`action_space`):
```python
Box(shape=(6, 6), low=0, high=max_qty)  # 连续动作（调度矩阵）
# 或
Discrete(n)  # 离散动作（预定义模板）
```

**奖励函数**:
```python
# 利润模式（默认）
reward = revenue - penalty - rebalance_cost

# 服务率模式
reward = served / total_demand * 100

# 混合模式
reward = alpha * profit + beta * service_rate
```

**核心方法**:
- `reset(seed, options)`: 重置环境
- `step(action)`: 执行一步
- `render()`: 渲染当前状态
- `close()`: 关闭环境

## 🚀 快速开始

### 前置条件

1. **lambda_params.pkl文件** (来自Day2的需求模型)
   
   文件应位于: `../results/lambda_params.pkl`
   
   如果文件不存在，需要先运行Day2的需求模型拟合代码。

2. **Python依赖**:
   ```bash
   pip install numpy gymnasium pyyaml --break-system-packages
   ```

### 测试环境

```bash
cd tests
python test_env.py
```

这将运行3个测试:
1. ✅ 需求采样器测试
2. ✅ Gym环境测试
3. ✅ 完整Episode集成测试

### 使用示例

#### 示例1: 创建和使用环境

```python
from bike_env import BikeRebalancingEnv

# 创建环境
env = BikeRebalancingEnv(
    config_path="config/env_config.yaml",
    scenario="sunny_weekday"
)

# 重置
obs, info = env.reset(seed=42)

# 运行episode
for _ in range(24):  # 24小时
    action = env.action_space.sample()  # 随机动作
    obs, reward, terminated, truncated, info = env.step(action)
    
    if terminated:
        break

env.close()
```

#### 示例2: Zero-Action基线

```python
import numpy as np

env = BikeRebalancingEnv(scenario="sunny_weekday")
obs, info = env.reset()

episode_reward = 0

while True:
    # 不调度（zero-action基线）
    action = np.zeros(env.action_space.shape)
    
    obs, reward, terminated, truncated, info = env.step(action)
    episode_reward += reward
    
    if terminated:
        print(f"Service Rate: {info['service_rate']*100:.1f}%")
        print(f"Net Profit: ${info['net_profit']:.2f}")
        break
```

#### 示例3: 需求采样器单独使用

```python
from demand_sampler import DemandSampler

sampler = DemandSampler(
    lambda_params_path="results/lambda_params.pkl",
    zone_weights=[0.25, 0.25, 0.15, 0.15, 0.10, 0.10],
    demand_scale=1.0,
    random_seed=42
)

# 采样夏季晴天工作日17:00的需求
demands = sampler.sample_demand(
    hour=17,
    season=3,
    workingday=1,
    weather=1
)

print(f"各区域需求: {demands}")
print(f"总需求: {demands.sum():.0f} 单")
```

## 📊 配置说明

### 修改场景

编辑 `config/env_config.yaml`:

```yaml
scenarios:
  my_custom_scenario:
    season: 3      # 夏季
    weather: 1     # 晴天
    workingday: 1  # 工作日
```

然后使用:

```python
env = BikeRebalancingEnv(scenario="my_custom_scenario")
```

### 调整时间跨度

```yaml
time:
  time_horizon: 168  # 7天 (默认)
  # 或
  time_horizon: 24   # 1天 (快速测试)
```

### 修改调度频率

```yaml
time:
  rebalance_frequency: 24  # 夜间集中调度（每天一次）
  # 或
  rebalance_frequency: 1   # 逐小时滚动调度
```

### 调整奖励函数

```yaml
reward:
  reward_type: "profit"       # 利润模式
  # reward_type: "service_rate"  # 服务率模式
  # reward_type: "mixed"         # 混合模式
  
  normalize: true
  normalization_factor: 1000.0
```

## 🔧 扩展开发

### 添加新的基线策略

在单独的文件中实现:

```python
# baseline_policies.py

def proportional_refill_policy(obs, env):
    """按比例补货策略"""
    inventory = obs['inventory']
    target = env.zone_weights * env.total_bikes
    
    # 计算调度矩阵
    action = compute_rebalancing(inventory, target)
    return action
```

### 集成RL算法

```python
from stable_baselines3 import PPO

# 创建环境
env = BikeRebalancingEnv(scenario="sunny_weekday")

# 训练PPO
model = PPO("MultiInputPolicy", env, verbose=1)
model.learn(total_timesteps=100000)

# 保存模型
model.save("ppo_bike_rebalancing")

# 评估
obs, info = env.reset()
for _ in range(24):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated:
        break
```

## 📈 性能优化

### 向量化环境（多进程训练）

```python
from stable_baselines3.common.vec_env import SubprocVecEnv

def make_env(scenario):
    def _init():
        return BikeRebalancingEnv(scenario=scenario)
    return _init

# 创建4个并行环境
envs = SubprocVecEnv([make_env("sunny_weekday") for _ in range(4)])

# 训练
model = PPO("MultiInputPolicy", envs, verbose=1)
model.learn(total_timesteps=400000)  # 4倍加速
```

## 📝 下一步工作

- [ ] 实现3种基线策略（Zero/Proportional/MinCost）
- [ ] 编写策略评估脚本（对比服务率、成本、收益）
- [ ] 训练PPO/DQN模型
- [ ] 生成评估报告和可视化
- [ ] 集成到Flask Dashboard

## 🐛 常见问题

### Q1: lambda_params.pkl文件不存在

**解决**: 先运行Day2的需求模型拟合代码生成该文件。

### Q2: 导入错误 "No module named 'gymnasium'"

**解决**:
```bash
pip install gymnasium --break-system-packages
```

### Q3: YAML配置加载失败

**解决**:
```bash
pip install pyyaml --break-system-packages
```

### Q4: 状态空间不匹配

**解决**: 检查配置文件中的`normalize_state`设置是否与代码一致。

## 📚 参考资料

- [Gymnasium官方文档](https://gymnasium.farama.org/)
- [Stable-Baselines3文档](https://stable-baselines3.readthedocs.io/)
- 项目Day2需求模型报告

---

**Author**: renr  
**Date**: 2025-10-28  
**Version**: 1.0