"""
基线策略评估脚本 (Baseline Policies Evaluation)
================================================

本脚本用于评估和对比不同调度策略的性能：
- Zero-Action Policy
- Proportional Refill Policy
- Min-Cost Flow Policy

评估指标：
- 服务率 (Service Rate)
- 未满足需求 (Unmet Demand)
- 调度成本 (Rebalance Cost)
- 总收入 (Total Revenue)
- 净利润 (Net Profit)

作者: renr
日期: 2025-10-29 (Day 4)
项目: 共享单车数据分析与强化学习调度
"""

import sys
import os
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 导入策略模块
from policies.baseline_policies import create_policy, get_available_policies

# 尝试导入环境模块（如果存在）
try:
    from simulator.bike_env import BikeRebalancingEnv
    from simulator.demand_sampler import DemandSampler
    ENV_AVAILABLE = True
except ImportError:
    ENV_AVAILABLE = False
    print("Warning: Environment modules not found. Using mock environment for testing.")


# ============================================================================
# Mock环境（用于测试，实际使用时会被真实环境替换）
# ============================================================================

class MockBikeEnv:
    """模拟环境（仅用于测试评估脚本）"""
    
    def __init__(self, config):
        self.config = config
        self.num_zones = config['zones']['num_zones']
        self.zone_capacity = np.array(config['zones']['zone_capacity'])
        self.current_step = 0
        self.max_steps = config['time']['time_horizon']
        
    def reset(self, seed=None, options=None):
        """重置环境"""
        if seed is not None:
            np.random.seed(seed)
        
        self.current_step = 0
        self.inventory = np.array([150, 160, 90, 85, 60, 55], dtype=np.float32)
        
        obs = {
            'inventory': self.inventory / self.zone_capacity,
            'hour': np.array([0.0]),
            'season': 2,
            'workingday': 1,
            'weather': 1
        }
        
        info = {'step': 0}
        return obs, info
    
    def step(self, action):
        """执行一步"""
        self.current_step += 1
        
        # 简化的奖励计算
        reward = np.random.uniform(50, 200)
        
        # 更新库存（简化）
        self.inventory += np.random.uniform(-10, 10, self.num_zones)
        self.inventory = np.clip(self.inventory, 0, self.zone_capacity)
        
        obs = {
            'inventory': self.inventory / self.zone_capacity,
            'hour': np.array([float(self.current_step % 24) / 23.0]),
            'season': 2,
            'workingday': 1,
            'weather': 1
        }
        
        terminated = self.current_step >= self.max_steps
        truncated = False
        
        info = {
            'step': self.current_step,
            'service_rate': np.random.uniform(0.7, 0.95),
            'total_demand': np.random.randint(100, 300),
            'served_demand': np.random.randint(80, 280),
            'unmet_demand': np.random.randint(0, 50),
            'rebalance_cost': (action * self.config['economics']['cost_matrix']).sum() if action is not None else 0,
            'revenue': np.random.uniform(300, 800)
        }
        
        return obs, reward, terminated, truncated, info


# ============================================================================
# 评估函数
# ============================================================================

def load_config(config_path: str = None) -> Dict[str, Any]:
    """
    加载配置文件
    
    Args:
        config_path: 配置文件路径，默认为 config/env_config.yaml
        
    Returns:
        配置字典
    """
    if config_path is None:
        config_path = project_root / "config" / "env_config.yaml"
    else:
        config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config


def evaluate_policy(policy, env, num_episodes: int = 5, 
                   random_seeds: List[int] = None) -> Dict[str, Any]:
    """
    评估单个策略
    
    Args:
        policy: 策略实例
        env: 环境实例
        num_episodes: 评估轮数
        random_seeds: 随机种子列表
        
    Returns:
        评估结果字典
    """
    if random_seeds is None:
        random_seeds = list(range(42, 42 + num_episodes))
    
    # 结果容器
    episode_results = []
    
    print(f"\n评估策略: {policy.name}")
    print("-" * 60)
    
    for episode_idx, seed in enumerate(random_seeds):
        # 重置环境
        obs, info = env.reset(seed=seed)
        
        # 重置策略统计
        policy.reset_stats()
        
        # Episode统计
        total_reward = 0.0
        total_demand = 0
        total_served = 0
        total_unmet = 0
        total_revenue = 0.0
        total_rebalance_cost = 0.0
        step_count = 0
        
        terminated = False
        truncated = False
        
        # 运行Episode
        while not (terminated or truncated):
            # 选择动作
            action = policy.select_action(obs)
            
            # 执行动作
            obs, reward, terminated, truncated, info = env.step(action)
            
            # 累积统计
            total_reward += reward
            total_demand += info.get('total_demand', 0)
            total_served += info.get('served_demand', 0)
            total_unmet += info.get('unmet_demand', 0)
            total_revenue += info.get('revenue', 0)
            total_rebalance_cost += info.get('rebalance_cost', 0)
            step_count += 1
        
        # Episode结果
        service_rate = total_served / max(total_demand, 1) if total_demand > 0 else 0
        net_profit = total_revenue - total_rebalance_cost
        
        episode_result = {
            'episode': episode_idx + 1,
            'seed': seed,
            'total_reward': total_reward,
            'service_rate': service_rate,
            'total_demand': total_demand,
            'served_demand': total_served,
            'unmet_demand': total_unmet,
            'total_revenue': total_revenue,
            'rebalance_cost': total_rebalance_cost,
            'net_profit': net_profit,
            'steps': step_count
        }
        
        episode_results.append(episode_result)
        
        # 打印进度
        print(f"Episode {episode_idx+1}/{num_episodes}: "
              f"Service Rate={service_rate*100:.1f}%, "
              f"Net Profit=${net_profit:.2f}, "
              f"Cost=${total_rebalance_cost:.2f}")
    
    # 计算统计量
    df = pd.DataFrame(episode_results)
    
    summary = {
        'policy_name': policy.name,
        'num_episodes': num_episodes,
        'mean_service_rate': df['service_rate'].mean(),
        'std_service_rate': df['service_rate'].std(),
        'mean_net_profit': df['net_profit'].mean(),
        'std_net_profit': df['net_profit'].std(),
        'mean_rebalance_cost': df['rebalance_cost'].mean(),
        'std_rebalance_cost': df['rebalance_cost'].std(),
        'mean_unmet_demand': df['unmet_demand'].mean(),
        'total_served': df['served_demand'].sum(),
        'total_unmet': df['unmet_demand'].sum(),
        'episodes': episode_results
    }
    
    print(f"\n{'='*60}")
    print(f"策略: {policy.name} - 汇总统计")
    print(f"{'='*60}")
    print(f"服务率:     {summary['mean_service_rate']*100:.2f}% ± {summary['std_service_rate']*100:.2f}%")
    print(f"净利润:     ${summary['mean_net_profit']:.2f} ± ${summary['std_net_profit']:.2f}")
    print(f"调度成本:   ${summary['mean_rebalance_cost']:.2f} ± ${summary['std_rebalance_cost']:.2f}")
    print(f"未满足需求: {summary['mean_unmet_demand']:.1f}")
    print(f"{'='*60}\n")
    
    return summary


def compare_policies(config: Dict[str, Any], 
                    policy_names: List[str] = None,
                    num_episodes: int = 5,
                    scenario: str = 'default') -> pd.DataFrame:
    """
    对比多个策略
    
    Args:
        config: 环境配置
        policy_names: 策略名称列表，默认为所有可用策略
        num_episodes: 每个策略的评估轮数
        scenario: 场景名称
        
    Returns:
        对比结果DataFrame
    """
    if policy_names is None:
        policy_names = get_available_policies()
    
    print("\n" + "="*80)
    print(f"策略对比评估 - 场景: {scenario}")
    print("="*80)
    
    # 创建环境
    if ENV_AVAILABLE:
        env = BikeRebalancingEnv(config)
    else:
        env = MockBikeEnv(config)
        print("  使用模拟环境进行测试\n")
    
    # 评估每个策略
    results = []
    
    for policy_name in policy_names:
        try:
            # 创建策略
            policy = create_policy(policy_name, config)
            
            # 评估策略
            summary = evaluate_policy(policy, env, num_episodes)
            
            results.append(summary)
            
        except Exception as e:
            print(f"Error evaluating policy {policy_name}: {e}")
            continue
    
    # 创建对比表
    comparison_data = []
    for result in results:
        comparison_data.append({
            '策略': result['policy_name'],
            '服务率(%)': f"{result['mean_service_rate']*100:.2f} ± {result['std_service_rate']*100:.2f}",
            '净利润($)': f"{result['mean_net_profit']:.2f} ± {result['std_net_profit']:.2f}",
            '调度成本($)': f"{result['mean_rebalance_cost']:.2f} ± {result['std_rebalance_cost']:.2f}",
            '未满足需求': f"{result['mean_unmet_demand']:.1f}",
            '总服务量': result['total_served'],
            '总未满足': result['total_unmet']
        })
    
    df_comparison = pd.DataFrame(comparison_data)
    
    print("\n" + "="*80)
    print("策略对比结果")
    print("="*80)
    print(df_comparison.to_string(index=False))
    print("="*80 + "\n")
    
    return df_comparison, results


def save_results(comparison_df: pd.DataFrame, 
                results: List[Dict],
                output_dir: str = "results",
                scenario: str = "default"):
    """
    保存评估结果
    
    Args:
        comparison_df: 对比结果DataFrame
        results: 详细结果列表
        output_dir: 输出目录
        scenario: 场景名称
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 保存对比表
    comparison_file = output_path / f"baseline_comparison_{scenario}_{timestamp}.csv"
    comparison_df.to_csv(comparison_file, index=False, encoding='utf-8-sig')
    print(f" 对比结果已保存: {comparison_file}")
    
    # 保存详细结果
    for result in results:
        policy_name = result['policy_name'].replace('-', '_').replace(' ', '_').lower()
        
        # 创建详细结果DataFrame
        episodes_df = pd.DataFrame(result['episodes'])
        
        detail_file = output_path / f"baseline_detail_{policy_name}_{scenario}_{timestamp}.csv"
        episodes_df.to_csv(detail_file, index=False, encoding='utf-8-sig')
        print(f" 详细结果已保存: {detail_file}")
    
    # 保存汇总统计
    summary_data = []
    for result in results:
        summary_data.append({
            'policy_name': result['policy_name'],
            'mean_service_rate': result['mean_service_rate'],
            'std_service_rate': result['std_service_rate'],
            'mean_net_profit': result['mean_net_profit'],
            'std_net_profit': result['std_net_profit'],
            'mean_rebalance_cost': result['mean_rebalance_cost'],
            'std_rebalance_cost': result['std_rebalance_cost'],
            'mean_unmet_demand': result['mean_unmet_demand'],
            'total_served': result['total_served'],
            'total_unmet': result['total_unmet']
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_file = output_path / f"baseline_summary_{scenario}_{timestamp}.csv"
    summary_df.to_csv(summary_file, index=False, encoding='utf-8-sig')
    print(f" 汇总统计已保存: {summary_file}")


def generate_report(comparison_df: pd.DataFrame, 
                   results: List[Dict],
                   config: Dict[str, Any],
                   scenario: str = "default",
                   output_dir: str = "results"):
    """
    生成Markdown评估报告
    
    Args:
        comparison_df: 对比结果DataFrame
        results: 详细结果列表
        config: 环境配置
        scenario: 场景名称
        output_dir: 输出目录
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = output_path / f"baseline_evaluation_report_{scenario}_{timestamp}.md"
    
    with open(report_file, 'w', encoding='utf-8') as f:
        # 标题
        f.write("# 基线策略评估报告\n\n")
        f.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**评估场景**: {scenario}\n\n")
        f.write("---\n\n")
        
        # 环境配置
        f.write("## 1. 环境配置\n\n")
        f.write(f"- **区域数量**: {config['zones']['num_zones']}\n")
        f.write(f"- **总单车数**: {config['inventory']['total_bikes']}\n")
        f.write(f"- **模拟时长**: {config['time']['time_horizon']} 小时\n")
        f.write(f"- **每单收入**: ${config['economics']['revenue_per_trip']}\n")
        f.write(f"- **未满足惩罚**: ${config['economics']['penalty_per_unmet']}\n")
        f.write(f"- **调度预算**: ${config['economics']['rebalance_budget']}\n\n")
        
        # 策略对比
        f.write("## 2. 策略对比\n\n")
        f.write(comparison_df.to_markdown(index=False))
        f.write("\n\n")
        
        # 详细分析
        f.write("## 3. 详细分析\n\n")
        
        for result in results:
            f.write(f"### 3.{results.index(result)+1} {result['policy_name']}\n\n")
            f.write(f"**评估轮数**: {result['num_episodes']}\n\n")
            f.write("**性能指标**:\n\n")
            f.write(f"- 平均服务率: {result['mean_service_rate']*100:.2f}% ± {result['std_service_rate']*100:.2f}%\n")
            f.write(f"- 平均净利润: ${result['mean_net_profit']:.2f} ± ${result['std_net_profit']:.2f}\n")
            f.write(f"- 平均调度成本: ${result['mean_rebalance_cost']:.2f} ± ${result['std_rebalance_cost']:.2f}\n")
            f.write(f"- 平均未满足需求: {result['mean_unmet_demand']:.1f}\n")
            f.write(f"- 总服务量: {result['total_served']}\n")
            f.write(f"- 总未满足: {result['total_unmet']}\n\n")
        
        # 结论
        f.write("## 4. 结论\n\n")
        
        # 找出最佳策略
        best_service_rate = max(results, key=lambda x: x['mean_service_rate'])
        best_profit = max(results, key=lambda x: x['mean_net_profit'])
        best_cost = min(results, key=lambda x: x['mean_rebalance_cost'])
        
        f.write(f"- **最高服务率**: {best_service_rate['policy_name']} ({best_service_rate['mean_service_rate']*100:.2f}%)\n")
        f.write(f"- **最高净利润**: {best_profit['policy_name']} (${best_profit['mean_net_profit']:.2f})\n")
        f.write(f"- **最低成本**: {best_cost['policy_name']} (${best_cost['mean_rebalance_cost']:.2f})\n\n")
        
        f.write("---\n\n")
        f.write("*报告由 evaluate_baselines.py 自动生成*\n")
    
    print(f" 评估报告已生成: {report_file}")


# ============================================================================
# 主函数
# ============================================================================

def main():
    """主函数"""
    
    print("\n" + "="*80)
    print("基线策略评估系统")
    print("="*80 + "\n")
    
    # 1. 加载配置
    print(" 加载配置文件...")
    try:
        config = load_config()
        print(" 配置加载成功\n")
    except Exception as e:
        print(f" 配置加载失败: {e}")
        return
    
    # 2. 选择场景
    scenario = 'default'  # 可以修改为其他场景
    print(f" 评估场景: {scenario}\n")
    
    # 3. 选择策略
    policy_names = ['zero', 'proportional', 'mincost']
    print(f" 评估策略: {', '.join(policy_names)}\n")
    
    # 4. 运行评估
    print(" 开始评估...\n")
    try:
        comparison_df, results = compare_policies(
            config=config,
            policy_names=policy_names,
            num_episodes=5,
            scenario=scenario
        )
    except Exception as e:
        print(f" 评估失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 5. 保存结果
    print("\n 保存结果...")
    try:
        save_results(comparison_df, results, output_dir="../results", scenario=scenario)
    except Exception as e:
        print(f" 保存失败: {e}")
    
    # 6. 生成报告
    print("\n 生成报告...")
    try:
        generate_report(comparison_df, results, config, scenario=scenario, output_dir="../results")
        print("报告生成完成")
    except Exception as e:
        print(f" 报告生成失败: {e}")
        import traceback
        traceback.print_exc()

    print(" 评估完成！")


if __name__ == "__main__":
    main()
