"""
基线策略模块 (Baseline Policies)
======================================

本模块实现了3种基线调度策略，用于与强化学习策略对比：
1. ZeroActionPolicy - 不调度策略
2. ProportionalRefillPolicy - 按比例补货策略
3. MinCostFlowPolicy - 最小成本流策略

作者: renr
日期: 2025-10-29 (Day 4)
项目: 共享单车数据分析与强化学习调度
"""

import numpy as np
import networkx as nx
from typing import Dict, Any, Tuple, Optional
from abc import ABC, abstractmethod


class BasePolicy(ABC):
    """策略基类"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化策略
        
        Args:
            config: 环境配置字典
        """
        self.config = config
        self.num_zones = config['zones']['num_zones']
        self.zone_names = config['zones']['zone_names']
        self.zone_weights = np.array(config['zones']['zone_weights'])
        self.zone_capacity = np.array(config['zones']['zone_capacity'])
        self.cost_matrix = np.array(config['economics']['cost_matrix'])
        self.max_rebalance_qty = config['economics']['max_rebalance_qty']
        self.rebalance_budget = config['economics'].get('rebalance_budget', float('inf'))
        
        # 统计信息
        self.total_rebalances = 0
        self.total_cost = 0.0
        
    @abstractmethod
    def select_action(self, observation: Dict[str, np.ndarray]) -> np.ndarray:
        """
        根据观测选择动作
        
        Args:
            observation: 环境观测 (包含 inventory, hour, season, workingday, weather)
            
        Returns:
            action: 调度动作矩阵 (num_zones × num_zones)
        """
        pass
    
    def reset_stats(self):
        """重置统计信息"""
        self.total_rebalances = 0
        self.total_cost = 0.0
    
    def get_stats(self) -> Dict[str, float]:
        """获取统计信息"""
        return {
            'total_rebalances': self.total_rebalances,
            'total_cost': self.total_cost,
            'avg_cost_per_rebalance': self.total_cost / max(1, self.total_rebalances)
        }


class ZeroActionPolicy(BasePolicy):
    """
    策略1: 不调度策略 (Zero-Action)
    ====================================
    
    **策略描述**:
    - 不进行任何调度操作
    - 单车自然流动，依靠还车机制维持库存
    
    **适用场景**:
    - 作为最简单的基准策略
    - 评估"不干预"场景的表现
    
    **优点**:
    - 零调度成本
    - 实施简单
    
    **缺点**:
    - 无法主动优化库存分布
    - 服务率可能较低
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.name = "Zero-Action"
        
    def select_action(self, observation: Dict[str, np.ndarray]) -> np.ndarray:
        """
        返回零动作矩阵
        
        Args:
            observation: 环境观测
            
        Returns:
            零矩阵 (num_zones × num_zones)
        """
        return np.zeros((self.num_zones, self.num_zones), dtype=np.float32)


class ProportionalRefillPolicy(BasePolicy):
    """
    策略2: 按比例补货策略 (Proportional Refill)
    ================================================
    
    **策略描述**:
    - 根据区域权重计算目标库存
    - 将富余区的库存调往缺口区
    - 使各区库存尽量接近目标比例
    
    **数学模型**:
    ```
    target_inventory[z] = zone_weight[z] × total_inventory
    surplus[z] = max(0, current_inventory[z] - target_inventory[z])
    deficit[z] = max(0, target_inventory[z] - current_inventory[z])
    ```
    
    **调度规则**:
    1. 识别富余区 (surplus > 0) 和缺口区 (deficit > 0)
    2. 按成本从低到高匹配调度
    3. 调度量受库存、预算、容量约束
    
    **参数**:
    - threshold: 触发调度的阈值比例 (default: 0.1)
    - rebalance_ratio: 每次调度的比例 (default: 0.5)
    
    **优点**:
    - 维持库存平衡
    - 适应性强
    - 计算简单
    
    **缺点**:
    - 未考虑需求预测
    - 可能过度调度
    """
    
    def __init__(self, config: Dict[str, Any], threshold: float = 0.1, rebalance_ratio: float = 0.5):
        """
        初始化按比例补货策略
        
        Args:
            config: 环境配置
            threshold: 触发调度的阈值（相对于目标库存的偏差比例）
            rebalance_ratio: 每次调度目标缺口/富余的比例
        """
        super().__init__(config)
        self.name = "Proportional-Refill"
        self.threshold = threshold
        self.rebalance_ratio = rebalance_ratio
        
    def select_action(self, observation: Dict[str, np.ndarray]) -> np.ndarray:
        """
        计算按比例补货的调度动作
        
        Args:
            observation: 环境观测
            
        Returns:
            调度动作矩阵
        """
        # 1. 获取当前库存（反归一化）
        inventory_norm = observation['inventory']
        current_inventory = inventory_norm * self.zone_capacity
        total_inventory = current_inventory.sum()
        
        # 2. 计算目标库存（按区域权重分配）
        target_inventory = self.zone_weights * total_inventory
        
        # 3. 计算偏差
        deviation = current_inventory - target_inventory
        
        # 4. 识别富余区和缺口区
        surplus_zones = []
        deficit_zones = []
        
        for z in range(self.num_zones):
            relative_deviation = abs(deviation[z]) / max(target_inventory[z], 1.0)
            
            if relative_deviation > self.threshold:
                if deviation[z] > 0:  # 富余
                    surplus_zones.append((z, deviation[z]))
                else:  # 缺口
                    deficit_zones.append((z, -deviation[z]))
        
        # 5. 如果没有需要调度的，返回零动作
        if not surplus_zones or not deficit_zones:
            return np.zeros((self.num_zones, self.num_zones), dtype=np.float32)
        
        # 6. 初始化调度矩阵
        action = np.zeros((self.num_zones, self.num_zones), dtype=np.float32)
        
        # 7. 贪心匹配：按成本从低到高调度
        # 生成所有可能的调度对 (from_zone, to_zone, cost, quantity)
        rebalance_pairs = []
        for from_z, surplus_qty in surplus_zones:
            for to_z, deficit_qty in deficit_zones:
                if from_z != to_z:
                    cost = self.cost_matrix[from_z, to_z]
                    # 调度量为富余和缺口的最小值，再乘以调度比例
                    qty = min(surplus_qty, deficit_qty) * self.rebalance_ratio
                    rebalance_pairs.append((from_z, to_z, cost, qty))
        
        # 按成本排序
        rebalance_pairs.sort(key=lambda x: x[2])
        
        # 8. 执行调度（考虑约束）
        remaining_budget = self.rebalance_budget
        surplus_remaining = {z: qty for z, qty in surplus_zones}
        deficit_remaining = {z: qty for z, qty in deficit_zones}
        
        for from_z, to_z, cost, qty in rebalance_pairs:
            # 检查剩余富余和缺口
            available_surplus = surplus_remaining.get(from_z, 0)
            available_deficit = deficit_remaining.get(to_z, 0)
            
            if available_surplus <= 0 or available_deficit <= 0:
                continue
            
            # 实际调度量
            actual_qty = min(qty, available_surplus, available_deficit, self.max_rebalance_qty)
            
            # 检查预算约束
            actual_cost = cost * actual_qty
            if actual_cost > remaining_budget:
                actual_qty = remaining_budget / max(cost, 0.001)
            
            if actual_qty > 0.5:  # 至少调度0.5辆（避免无意义的小额调度）
                action[from_z, to_z] = actual_qty
                
                # 更新剩余量
                surplus_remaining[from_z] -= actual_qty
                deficit_remaining[to_z] -= actual_qty
                remaining_budget -= actual_cost
                
                # 更新统计
                self.total_rebalances += 1
                self.total_cost += actual_cost
            
            # 如果预算耗尽，停止
            if remaining_budget <= 0:
                break
        
        return action


class MinCostFlowPolicy(BasePolicy):
    """
    策略3: 最小成本流策略 (Min-Cost Flow)
    ==========================================
    
    **策略描述**:
    - 将调度问题建模为网络流优化问题
    - 求解最小成本最大流
    - 在满足需求的前提下最小化调度成本
    
    **数学模型**:
    ```
    Min: Σ c_ij × f_ij
    s.t.:
        Σ_j f_ij - Σ_j f_ji = supply[i]  (for surplus zones)
        Σ_j f_ij - Σ_j f_ji = -demand[i] (for deficit zones)
        0 ≤ f_ij ≤ capacity_ij
    ```
    
    **网络构建**:
    - 源点 (source): 连接所有富余区
    - 汇点 (sink): 连接所有缺口区
    - 边容量: 库存限制
    - 边成本: cost_matrix
    
    **求解方法**:
    - 使用 NetworkX 的 min_cost_flow 算法
    - 基于 Successive Shortest Path 算法
    
    **参数**:
    - threshold: 触发调度的阈值
    - use_expected_demand: 是否使用期望需求预测
    
    **优点**:
    - 理论最优（最小成本）
    - 全局优化
    
    **缺点**:
    - 计算复杂度较高
    - 需要准确的需求预测
    """
    
    def __init__(self, config: Dict[str, Any], threshold: float = 0.15, 
                 use_expected_demand: bool = False):
        """
        初始化最小成本流策略
        
        Args:
            config: 环境配置
            threshold: 触发调度的阈值
            use_expected_demand: 是否使用期望需求预测（需要demand_sampler）
        """
        super().__init__(config)
        self.name = "Min-Cost-Flow"
        self.threshold = threshold
        self.use_expected_demand = use_expected_demand
        
    def select_action(self, observation: Dict[str, np.ndarray]) -> np.ndarray:
        """
        使用最小成本流算法计算调度动作
        
        Args:
            observation: 环境观测
            
        Returns:
            调度动作矩阵
        """
        # 1. 获取当前库存
        inventory_norm = observation['inventory']
        current_inventory = inventory_norm * self.zone_capacity
        total_inventory = current_inventory.sum()
        
        # 2. 计算目标库存（基于权重或期望需求）
        if self.use_expected_demand:
            # TODO: 集成 demand_sampler 获取期望需求
            # 当前使用权重作为替代
            target_inventory = self.zone_weights * total_inventory
        else:
            target_inventory = self.zone_weights * total_inventory
        
        # 3. 计算供需
        supply_demand = current_inventory - target_inventory
        
        # 4. 识别富余区和缺口区
        surplus_zones = []
        deficit_zones = []
        
        for z in range(self.num_zones):
            relative_deviation = abs(supply_demand[z]) / max(target_inventory[z], 1.0)
            
            if relative_deviation > self.threshold:
                if supply_demand[z] > 0:
                    surplus_zones.append((z, supply_demand[z]))
                else:
                    deficit_zones.append((z, -supply_demand[z]))
        
        # 5. 如果没有需要调度的，返回零动作
        if not surplus_zones or not deficit_zones:
            return np.zeros((self.num_zones, self.num_zones), dtype=np.float32)
        
        # 6. 构建网络流图
        G = nx.DiGraph()
        
        # 添加源点和汇点
        source = 'source'
        sink = 'sink'
        G.add_node(source, demand=-sum(qty for _, qty in surplus_zones))  # 源点供应
        G.add_node(sink, demand=sum(qty for _, qty in deficit_zones))     # 汇点需求
        
        # 添加区域节点
        for z in range(self.num_zones):
            G.add_node(f'zone_{z}', demand=0)
        
        # 添加边：source → 富余区
        for z, surplus_qty in surplus_zones:
            G.add_edge(source, f'zone_{z}', 
                      weight=0,  # 从源点到富余区无成本
                      capacity=int(surplus_qty))
        
        # 添加边：缺口区 → sink
        for z, deficit_qty in deficit_zones:
            G.add_edge(f'zone_{z}', sink,
                      weight=0,  # 从缺口区到汇点无成本
                      capacity=int(deficit_qty))
        
        # 添加边：富余区 → 缺口区
        for from_z, _ in surplus_zones:
            for to_z, _ in deficit_zones:
                if from_z != to_z:
                    cost = int(self.cost_matrix[from_z, to_z] * 100)  # 放大100倍避免浮点数
                    capacity = int(min(current_inventory[from_z], 
                                     self.max_rebalance_qty,
                                     self.zone_capacity[to_z] - current_inventory[to_z]))
                    
                    if capacity > 0:
                        G.add_edge(f'zone_{from_z}', f'zone_{to_z}',
                                  weight=cost,
                                  capacity=capacity)
        
        # 7. 求解最小成本流
        try:
            flow_dict = nx.min_cost_flow(G)
            
            # 8. 提取调度动作
            action = np.zeros((self.num_zones, self.num_zones), dtype=np.float32)
            
            for from_z, _ in surplus_zones:
                from_node = f'zone_{from_z}'
                if from_node in flow_dict:
                    for to_z, _ in deficit_zones:
                        to_node = f'zone_{to_z}'
                        if to_node in flow_dict[from_node]:
                            flow = flow_dict[from_node][to_node]
                            if flow > 0:
                                action[from_z, to_z] = float(flow)
                                
                                # 更新统计
                                cost = self.cost_matrix[from_z, to_z] * flow
                                self.total_rebalances += 1
                                self.total_cost += cost
            
            return action
            
        except nx.NetworkXUnfeasible:
        # 减少警告输出
            if not hasattr(self, '_infeasible_count'):
                self._infeasible_count = 0
                print("Warning: Min-cost flow is infeasible. This is normal in Mock environment.")
                print("Subsequent warnings will be suppressed...")
            self._infeasible_count += 1
            return np.zeros((self.num_zones, self.num_zones), dtype=np.float32)
        
        except Exception as e:
            print(f"Error in min-cost flow: {e}")
            return np.zeros((self.num_zones, self.num_zones), dtype=np.float32)


# ============================================================================
# 工厂函数
# ============================================================================

def create_policy(policy_name: str, config: Dict[str, Any], **kwargs) -> BasePolicy:
    """
    策略工厂函数
    
    Args:
        policy_name: 策略名称 ('zero', 'proportional', 'mincost')
        config: 环境配置
        **kwargs: 策略特定参数
        
    Returns:
        策略实例
        
    Example:
        >>> policy = create_policy('proportional', config, threshold=0.1)
        >>> action = policy.select_action(observation)
    """
    policy_name = policy_name.lower()
    
    if policy_name in ['zero', 'zero-action', 'zeroaction']:
        return ZeroActionPolicy(config)
    
    elif policy_name in ['proportional', 'prop', 'proportional-refill']:
        return ProportionalRefillPolicy(config, **kwargs)
    
    elif policy_name in ['mincost', 'min-cost', 'mincostflow', 'mcf']:
        return MinCostFlowPolicy(config, **kwargs)
    
    else:
        raise ValueError(f"Unknown policy: {policy_name}. "
                        f"Available: 'zero', 'proportional', 'mincost'")


# ============================================================================
# 辅助函数
# ============================================================================

def get_available_policies() -> list:
    """返回可用的策略列表"""
    return ['zero', 'proportional', 'mincost']


def print_policy_info(policy_name: str):
    """
    打印策略详细信息
    
    Args:
        policy_name: 策略名称
    """
    policies_info = {
        'zero': {
            'name': 'Zero-Action Policy',
            'description': '不进行任何调度操作，依靠自然流动',
            'pros': '零成本、简单',
            'cons': '服务率可能较低',
            'use_case': '基准对比'
        },
        'proportional': {
            'name': 'Proportional Refill Policy',
            'description': '按区域权重维持库存比例，贪心匹配调度',
            'pros': '维持平衡、适应性强',
            'cons': '未考虑需求预测',
            'use_case': '快速响应库存失衡'
        },
        'mincost': {
            'name': 'Min-Cost Flow Policy',
            'description': '网络流优化，求解最小成本调度方案',
            'pros': '理论最优、全局优化',
            'cons': '计算复杂度高',
            'use_case': '离线规划、理论上限'
        }
    }
    
    policy_name = policy_name.lower()
    if policy_name not in policies_info:
        print(f"Unknown policy: {policy_name}")
        return
    
    info = policies_info[policy_name]
    print(f"\n{'='*60}")
    print(f"策略: {info['name']}")
    print(f"{'='*60}")
    print(f"描述: {info['description']}")
    print(f"优点: {info['pros']}")
    print(f"缺点: {info['cons']}")
    print(f"适用场景: {info['use_case']}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    """测试代码"""
    
    # 模拟配置
    test_config = {
        'zones': {
            'num_zones': 6,
            'zone_names': ['A', 'B', 'C', 'D', 'E', 'F'],
            'zone_weights': [0.25, 0.25, 0.15, 0.15, 0.10, 0.10],
            'zone_capacity': [200, 200, 120, 120, 80, 80]
        },
        'economics': {
            'cost_matrix': [
                [0.0, 1.5, 2.0, 1.5, 1.0, 2.5],
                [1.5, 0.0, 2.5, 1.0, 1.5, 2.0],
                [2.0, 2.5, 0.0, 2.0, 2.5, 3.0],
                [1.5, 1.0, 2.0, 0.0, 1.0, 1.5],
                [1.0, 1.5, 2.5, 1.0, 0.0, 2.0],
                [2.5, 2.0, 3.0, 1.5, 2.0, 0.0]
            ],
            'max_rebalance_qty': 50,
            'rebalance_budget': 500
        }
    }
    
    # 模拟观测
    test_obs = {
        'inventory': np.array([0.9, 0.3, 0.8, 0.4, 0.7, 0.5]),  # 归一化库存
        'hour': np.array([8.0/23.0]),
        'season': 2,
        'workingday': 1,
        'weather': 1
    }
    
    print("="*60)
    print("基线策略测试")
    print("="*60)
    
    # 测试所有策略
    for policy_name in ['zero', 'proportional', 'mincost']:
        print(f"\n测试策略: {policy_name}")
        print("-"*60)
        
        policy = create_policy(policy_name, test_config)
        action = policy.select_action(test_obs)
        
        print(f"调度动作矩阵:")
        print(action)
        print(f"调度总量: {action.sum():.2f} 辆")
        print(f"非零调度: {np.count_nonzero(action)} 条路径")
        
        if action.sum() > 0:
            # 计算总成本
            cost_matrix = np.array(test_config['economics']['cost_matrix'])
            total_cost = (action * cost_matrix).sum()
            print(f"估计成本: ${total_cost:.2f}")
    
    print("\n" + "="*60)
    print("测试完成！")
    print("="*60)

# ============================================================================
# Day 5: 参数优化器
# ============================================================================

class ParameterOptimizer:
    """
    参数优化器 - Day 5新增
    使用网格搜索优化策略参数
    """
    
    def __init__(self, config: Dict[str, Any]):
        """初始化优化器"""
        self.config = config
        self.optimization_config = config.get('optimization', {})
        self.method = self.optimization_config.get('method', 'grid_search')
        self.target_metric = self.optimization_config.get('target_metric', 'net_profit')
        
    def optimize_proportional_policy(self, env, scenarios: list = None) -> Dict:
        """
        优化ProportionalRefillPolicy参数
        
        Args:
            env: 环境实例或环境工厂函数
            scenarios: 评估场景列表
            
        Returns:
            优化结果字典
        """
        if self.method == 'grid_search':
            return self._grid_search_optimization(env, scenarios)
        else:
            raise NotImplementedError(f"Method {self.method} not implemented")
    
    def _grid_search_optimization(self, env_factory, scenarios: list) -> Dict:
        """网格搜索优化"""
        # 获取参数网格
        param_grid = self.config['baseline_policies']['proportional_refill']['parameter_grid']
        thresholds = param_grid['threshold']
        ratios = param_grid['rebalance_ratio']
        
        scenarios = scenarios or self.optimization_config.get('scenarios', ['default'])
        
        best_params = None
        best_score = float('-inf')
        results = []
        
        print(f"\n{'='*60}")
        print(f"网格搜索优化 - Proportional策略")
        print(f"{'='*60}")
        print(f"参数空间: threshold={thresholds}, ratio={ratios}")
        print(f"评估场景: {scenarios}")
        print(f"目标指标: {self.target_metric}\n")
        
        total_combinations = len(thresholds) * len(ratios)
        current = 0
        
        for threshold in thresholds:
            for ratio in ratios:
                current += 1
                print(f"[{current}/{total_combinations}] 测试: threshold={threshold:.2f}, ratio={ratio:.2f}")
                
                # 在多个场景下评估
                scores = []
                for scenario in scenarios:
                    score = self._evaluate_parameters(
                        env_factory, scenario, threshold, ratio
                    )
                    scores.append(score)
                
                avg_score = np.mean(scores)
                std_score = np.std(scores)
                
                print(f"  → {self.target_metric}: {avg_score:.2f} ± {std_score:.2f}")
                
                # 记录结果
                results.append({
                    'threshold': threshold,
                    'rebalance_ratio': ratio,
                    'avg_score': avg_score,
                    'std_score': std_score,
                    'scores': scores
                })
                
                # 更新最优参数
                if avg_score > best_score:
                    best_score = avg_score
                    best_params = {'threshold': threshold, 'rebalance_ratio': ratio}
                    print(f"  ✨ 新的最优参数！")
        
        print(f"\n{'='*60}")
        print(f"优化完成！")
        print(f"最优参数: {best_params}")
        print(f"最优得分: {best_score:.2f}")
        print(f"{'='*60}\n")
        
        return {
            'best_params': best_params,
            'best_score': best_score,
            'all_results': results,
            'method': self.method,
            'target_metric': self.target_metric
        }
    
    def _evaluate_parameters(self, env_factory, scenario: str, 
                            threshold: float, ratio: float) -> float:
        """评估单个参数组合"""
        # 创建环境和策略
        env = env_factory(config_path=None, scenario=scenario) if callable(env_factory) else env_factory
        policy = ProportionalRefillPolicy(self.config, threshold=threshold, rebalance_ratio=ratio)
        
        # 运行3个episode求平均
        seeds = [42, 43, 44]
        scores = []
        
        for seed in seeds:
            obs, _ = env.reset(seed=seed)
            policy.reset_stats()
            
            done = False
            while not done:
                action = policy.select_action(obs)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
            
            # 计算得分
            if self.target_metric == 'net_profit':
                score = info['net_profit']
            elif self.target_metric == 'service_rate':
                score = info['service_rate'] * 10000
            else:
                score = info.get(self.target_metric, 0)
            
            scores.append(score)
        
        return np.mean(scores)