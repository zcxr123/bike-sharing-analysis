"""
共享单车调度环境 (Bike Rebalancing Environment)
基于Gymnasium框架的强化学习环境

Author: renr
Date: 2025-10-28
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Tuple, Optional, Any
import yaml
from pathlib import Path

try:
    # 包方式导入（推荐）：python -m ... 或 from simulator.bike_env import ...
    from .demand_sampler import DemandSampler  # type: ignore
except ImportError:
    # 兼容顶层运行/导入：python tests/test_env.py 或 from bike_env import ...
    from demand_sampler import DemandSampler  # type: ignore



class BikeRebalancingEnv(gym.Env):
    """
    共享单车调度环境
    
    状态: 各区库存 + 时间上下文
    动作: 6×6调度矩阵 (from_zone, to_zone, quantity)
    奖励: revenue - penalty - rebalance_cost
    """
    
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 4}
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        config_dict: Optional[Dict] = None,
        scenario: str = "default"
    ):
        """
        初始化环境
        
        Args:
            config_path: 配置文件路径（yaml）
            config_dict: 配置字典（如果提供则不读取文件）
            scenario: 场景名称（default/sunny_weekday/rainy_weekend等）
        """
        super().__init__()
        
        # 加载配置
        if config_dict is not None:
            self.config = config_dict
        elif config_path is not None:
            self.config = self._load_config(config_path)
        else:
            # 使用默认配置路径
            default_path = Path(__file__).parent.parent / "config" / "env_config.yaml"
            self.config = self._load_config(str(default_path))
        
        # 提取配置参数
        self._extract_config()
        
        # 初始化需求采样器
        self.demand_sampler = DemandSampler(
            lambda_params_path=self.lambda_params_path,
            zone_weights=self.zone_weights,
            demand_scale=self.demand_scale,
            random_seed=self.random_seed
        )
        
        # 定义动作空间和观测空间
        self._define_spaces()
        
        # 初始化状态变量
        self.current_step = 0
        self.current_hour = 0
        self.inventory = None
        self.total_revenue = 0.0
        self.total_cost = 0.0
        self.total_served = 0
        self.total_demand = 0
        self.total_unmet = 0
        
        # 场景配置
        self.scenario = scenario
        self._load_scenario(scenario)
        
        print(f"[BikeEnv] 环境初始化完成")
        print(f"  - 区域数: {self.num_zones}")
        print(f"  - 时间跨度: {self.time_horizon}小时")
        print(f"  - 场景: {scenario}")
    
    def _load_config(self, path: str) -> Dict:
        """加载配置文件"""
        with open(path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    
    def _extract_config(self):
        """提取配置参数"""
        # 区域配置
        self.num_zones = self.config['zones']['num_zones']
        self.zone_names = self.config['zones']['zone_names']
        self.zone_weights = np.array(self.config['zones']['zone_weights'])
        self.zone_capacity = np.array(self.config['zones']['zone_capacity'])
        
        # 时间配置
        self.time_horizon = self.config['time']['time_horizon']
        self.step_size = self.config['time']['step_size']
        self.rebalance_freq = self.config['time']['rebalance_frequency']
        
        # 库存配置
        self.total_bikes = self.config['inventory']['total_bikes']
        self.init_dist = self.config['inventory']['initial_distribution']
        
        # 需求配置
        self.lambda_params_path = self.config['demand']['lambda_params_path']
        self.demand_scale = self.config['demand']['demand_scale']
        self.random_seed = self.config['demand']['random_seed']
        
        # 经济参数
        self.revenue_per_trip = self.config['economics']['revenue_per_trip']
        self.penalty_per_unmet = self.config['economics']['penalty_per_unmet']
        self.cost_matrix = np.array(self.config['economics']['cost_matrix'])
        self.rebalance_budget = self.config['economics']['rebalance_budget']
        self.max_rebalance_qty = self.config['economics']['max_rebalance_qty']
        
        # 奖励配置
        self.reward_type = self.config['reward']['reward_type']
        self.normalize_reward = self.config['reward']['normalize']
        self.norm_factor = self.config['reward']['normalization_factor']
        self.gamma = self.config['reward']['gamma']
        
        # 环境行为
        self.normalize_state = self.config['environment']['normalize_state']
        self.action_space_type = self.config['environment']['action_space_type']
        self.clip_actions = self.config['environment']['clip_actions']
    
    def _load_scenario(self, scenario: str):
        """加载场景配置"""
        scenarios = self.config['scenarios']
        if scenario not in scenarios:
            print(f"⚠️  场景'{scenario}'不存在，使用default")
            scenario = 'default'
        
        self.season = scenarios[scenario]['season']
        self.weather = scenarios[scenario]['weather']
        self.workingday = scenarios[scenario]['workingday']
    
    def _define_spaces(self):
        """定义动作空间和观测空间"""
        # 观测空间：字典格式
        # - inventory: 各区库存 (num_zones,)
        # - hour: 当前小时 (标量)
        # - season: 季节 (标量)
        # - workingday: 工作日标识 (标量)
        # - weather: 天气 (标量)
        
        if self.normalize_state:
            # 归一化后的范围 [0, 1]
            self.observation_space = spaces.Dict({
                'inventory': spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(self.num_zones,),
                    dtype=np.float32
                ),
                'hour': spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
                'season': spaces.Discrete(4),  # 1-4
                'workingday': spaces.Discrete(2),  # 0-1
                'weather': spaces.Discrete(4),  # 1-4
            })
        else:
            # 原始范围
            self.observation_space = spaces.Dict({
                'inventory': spaces.Box(
                    low=0.0,
                    high=self.zone_capacity.max(),
                    shape=(self.num_zones,),
                    dtype=np.float32
                ),
                'hour': spaces.Box(low=0, high=23, shape=(1,), dtype=np.float32),
                'season': spaces.Discrete(4),
                'workingday': spaces.Discrete(2),
                'weather': spaces.Discrete(4),
            })
        
        # 动作空间：连续（6×6矩阵）或离散（预定义模板）
        if self.action_space_type == "continuous":
            # 6×6矩阵，每个元素表示从i区调往j区的数量
            # 范围 [0, max_rebalance_qty]
            self.action_space = spaces.Box(
                low=0.0,
                high=float(self.max_rebalance_qty),
                shape=(self.num_zones, self.num_zones),
                dtype=np.float32
            )
        else:
            # 离散动作：预定义若干调度模板
            # 这里简化为N个离散选项（后续可扩展）
            num_discrete_actions = 10
            self.action_space = spaces.Discrete(num_discrete_actions)
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict, Dict]:
        """
        重置环境
        
        Returns:
            observation: 初始观测
            info: 额外信息
        """
        super().reset(seed=seed)
        
        # 重置状态变量
        self.current_step = 0
        self.current_hour = 0
        self.total_revenue = 0.0
        self.total_cost = 0.0
        self.total_served = 0
        self.total_demand = 0
        self.total_unmet = 0
        
        # 初始化库存分布
        self.inventory = self._initialize_inventory()
        
        # 如果提供了options，可以覆盖场景参数
        if options is not None:
            if 'scenario' in options:
                self._load_scenario(options['scenario'])
            if 'season' in options:
                self.season = options['season']
            if 'weather' in options:
                self.weather = options['weather']
            if 'workingday' in options:
                self.workingday = options['workingday']
        
        # 获取初始观测
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, info
    
    def _initialize_inventory(self) -> np.ndarray:
        """初始化各区库存"""
        if self.init_dist == "uniform":
            # 均匀分配
            base = self.total_bikes // self.num_zones
            remainder = self.total_bikes % self.num_zones
            inventory = np.full(self.num_zones, base, dtype=np.float32)
            inventory[:remainder] += 1
        
        elif self.init_dist == "proportional":
            # 按区域权重分配
            inventory = (self.zone_weights * self.total_bikes).astype(np.float32)
        
        elif self.init_dist == "random":
            # 随机分配（Dirichlet分布）
            alpha = np.ones(self.num_zones)
            weights = np.random.dirichlet(alpha)
            inventory = (weights * self.total_bikes).astype(np.float32)
        
        else:
            raise ValueError(f"未知的初始分布策略: {self.init_dist}")
        
        return inventory
    
    def step(self, action: np.ndarray) -> Tuple[Dict, float, bool, bool, Dict]:
        """
        执行一步
        
        Args:
            action: 动作（调度矩阵或离散索引）
        
        Returns:
            observation: 新观测
            reward: 奖励
            terminated: 是否终止
            truncated: 是否截断
            info: 额外信息
        """
        # 1. 处理动作（调度）
        rebalance_matrix = self._process_action(action)
        rebalance_cost = self._apply_rebalancing(rebalance_matrix)
        
        # 2. 需求采样与满足
        demands = self.demand_sampler.sample_demand(
            hour=self.current_hour % 24,
            season=self.season,
            workingday=self.workingday,
            weather=self.weather
        )
        
        served, unmet = self._serve_demands(demands)
        
        # 3. 计算奖励
        revenue = self.revenue_per_trip * served
        penalty = self.penalty_per_unmet * unmet
        
        if self.reward_type == "profit":
            reward = revenue - penalty - rebalance_cost
        elif self.reward_type == "service_rate":
            service_rate = served / (demands.sum() + 1e-6)
            reward = service_rate * 100  # 转为百分比
        elif self.reward_type == "mixed":
            profit = revenue - penalty - rebalance_cost
            service_rate = served / (demands.sum() + 1e-6)
            alpha = self.config['reward']['alpha']
            beta = self.config['reward']['beta']
            reward = alpha * profit + beta * service_rate * 100
        else:
            reward = 0.0
        
        # 归一化奖励
        if self.normalize_reward:
            reward /= self.norm_factor
        
        # 4. 更新统计量
        self.total_revenue += revenue
        self.total_cost += rebalance_cost
        self.total_served += served
        self.total_demand += demands.sum()
        self.total_unmet += unmet
        
        # 5. 推进时间
        self.current_step += 1
        self.current_hour += self.step_size
        
        # 6. 判断终止条件
        terminated = (self.current_step >= self.time_horizon)
        truncated = False
        
        # 7. 获取新观测和信息
        obs = self._get_observation()
        info = self._get_info()
        info['demands'] = demands
        info['served'] = served
        info['unmet'] = unmet
        info['rebalance_cost'] = rebalance_cost
        info['revenue'] = revenue
        
        return obs, reward, terminated, truncated, info
    
    def _process_action(self, action: np.ndarray) -> np.ndarray:
        """
        处理动作，转换为调度矩阵
        
        Returns:
            rebalance_matrix: (num_zones, num_zones) 调度矩阵
        """
        if self.action_space_type == "continuous":
            matrix = action.copy()
            
            # 裁剪到合理范围
            if self.clip_actions:
                matrix = np.clip(matrix, 0, self.max_rebalance_qty)
            
            # 对角线置零（不允许同区调度）
            np.fill_diagonal(matrix, 0)
            
            # 确保出库量不超过库存
            for i in range(self.num_zones):
                outflow = matrix[i, :].sum()
                if outflow > self.inventory[i]:
                    # 按比例缩减
                    scale = self.inventory[i] / (outflow + 1e-6)
                    matrix[i, :] *= scale
            
            return matrix
        
        else:
            # 离散动作：映射到预定义模板
            # TODO: 实现离散动作映射
            return np.zeros((self.num_zones, self.num_zones), dtype=np.float32)
    
    def _apply_rebalancing(self, matrix: np.ndarray) -> float:
        """
        应用调度，更新库存并计算成本
        
        Returns:
            total_cost: 调度总成本
        """
        total_cost = 0.0
        
        for i in range(self.num_zones):
            for j in range(self.num_zones):
                qty = matrix[i, j]
                if qty > 0:
                    # 更新库存
                    self.inventory[i] -= qty
                    self.inventory[j] += qty
                    
                    # 计算成本
                    cost = self.cost_matrix[i, j] * qty
                    total_cost += cost
        
        # 检查预算约束
        if self.rebalance_budget is not None and total_cost > self.rebalance_budget:
            # 超出预算，额外惩罚
            penalty = (total_cost - self.rebalance_budget) * 2
            total_cost += penalty
        
        # 确保库存非负（理论上不应发生）
        self.inventory = np.maximum(self.inventory, 0)
        
        # 确保不超过容量上限
        self.inventory = np.minimum(self.inventory, self.zone_capacity)
        
        return total_cost
    
    def _serve_demands(self, demands: np.ndarray) -> Tuple[float, float]:
        """
        满足需求，更新库存（包含还车机制）
        
        还车逻辑：
        - 用户从区域i取车
        - 骑行后还车到区域j
        - 还车目的地按区域权重分布
        
        Returns:
            served: 已满足需求量
            unmet: 未满足需求量
        """
        served_per_zone = np.minimum(demands, self.inventory)
        unmet_per_zone = demands - served_per_zone
        
        # 更新库存（扣除已服务的单车）
        self.inventory -= served_per_zone
        
        # ⭐ 还车机制：已服务的单车会还回到某个区域
        # 简化假设：还车目的地按区域权重分布（可以理解为流动矩阵）
        total_served = served_per_zone.sum()
        
        if total_served > 0:
            # 按区域权重分配还车量
            returned_per_zone = total_served * self.zone_weights
            
            # 添加随机扰动（模拟真实的不确定性）
            # 85%按权重，15%随机
            deterministic_returns = 0.85 * returned_per_zone
            random_returns = 0.15 * total_served * np.random.dirichlet(np.ones(self.num_zones))
            
            returned_per_zone = deterministic_returns + random_returns
            
            # 还车入库
            self.inventory += returned_per_zone
            
            # 确保不超过容量
            self.inventory = np.minimum(self.inventory, self.zone_capacity)
        
        served = served_per_zone.sum()
        unmet = unmet_per_zone.sum()
        
        return served, unmet
    
    def _get_observation(self) -> Dict:
        """获取当前观测"""
        if self.normalize_state:
            # 归一化
            inventory_norm = self.inventory / self.zone_capacity
            hour_norm = (self.current_hour % 24) / 23.0
            
            obs = {
                'inventory': inventory_norm.astype(np.float32),
                'hour': np.array([hour_norm], dtype=np.float32),
                'season': self.season - 1,  # 转为0-3
                'workingday': self.workingday,
                'weather': self.weather - 1,  # 转为0-3
            }
        else:
            obs = {
                'inventory': self.inventory.astype(np.float32),
                'hour': np.array([self.current_hour % 24], dtype=np.float32),
                'season': self.season - 1,
                'workingday': self.workingday,
                'weather': self.weather - 1,
            }
        
        return obs
    
    def _get_info(self) -> Dict:
        """获取额外信息"""
        service_rate = self.total_served / (self.total_demand + 1e-6)
        
        info = {
            'step': self.current_step,
            'hour': self.current_hour,
            'total_inventory': self.inventory.sum(),
            'service_rate': service_rate,
            'total_revenue': self.total_revenue,
            'total_cost': self.total_cost,
            'net_profit': self.total_revenue - self.total_cost,
            'total_served': self.total_served,
            'total_demand': self.total_demand,
            'total_unmet': self.total_unmet,
        }
        
        return info
    
    def render(self):
        """渲染环境（简单文本输出）"""
        print(f"\n{'='*60}")
        print(f"Step: {self.current_step} | Hour: {self.current_hour % 24:02d}:00")
        print(f"{'='*60}")
        print("库存分布:")
        for i, (name, inv) in enumerate(zip(self.zone_names, self.inventory)):
            print(f"  {name:20s}: {inv:6.1f} / {self.zone_capacity[i]:6.1f}")
        print(f"\n服务率: {self._get_info()['service_rate']*100:.1f}%")
        print(f"净利润: ${self._get_info()['net_profit']:.2f}")
        print(f"{'='*60}\n")
    
    def close(self):
        """关闭环境"""
        pass


# ==================== 辅助函数 ====================

def create_env_from_config(config_path: str, scenario: str = "default") -> BikeRebalancingEnv:
    """从配置文件创建环境"""
    return BikeRebalancingEnv(config_path=config_path, scenario=scenario)


# ==================== 测试代码 ====================

if __name__ == "__main__":
    print("="*60)
    print("Gym环境测试")
    print("="*60)
    
    # 创建环境
    try:
        env = BikeRebalancingEnv(
            config_path="../config/env_config.yaml",
            scenario="sunny_weekday"
        )
        
        # 测试reset
        print("\n[测试1] 环境重置")
        print("-" * 40)
        obs, info = env.reset(seed=42)
        print(f"初始库存: {obs['inventory']}")
        print(f"初始信息: {info}")
        
        # 测试step
        print("\n[测试2] 执行步骤")
        print("-" * 40)
        
        # 随机动作
        for i in range(5):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            print(f"\nStep {i+1}:")
            print(f"  Reward: {reward:.2f}")
            print(f"  Service Rate: {info['service_rate']*100:.1f}%")
            print(f"  Served: {info['served']:.0f} / {info['demands'].sum():.0f}")
            
            if terminated:
                print("\n✅ Episode 结束!")
                break
        
        # 测试render
        print("\n[测试3] 渲染环境")
        print("-" * 40)
        env.render()
        
        print("\n✅ Gym环境测试通过！")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()