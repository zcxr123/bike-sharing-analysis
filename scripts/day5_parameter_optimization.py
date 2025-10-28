"""
Day 5: 策略参数优化
主要任务：
1. Proportional策略参数优化（网格搜索）
2. Min-Cost策略在真实环境中测试
3. 多场景策略评估
"""
import os
import sys
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from simulator.bike_env import BikeRebalancingEnv
from policies.baseline_policies import (
    ZeroActionPolicy, ProportionalRefillPolicy, 
    MinCostFlowPolicy, ParameterOptimizer
)


def load_config(config_path: str = None) -> dict:
    """加载配置文件"""
    if config_path is None:
        config_path = project_root / "config" / "env_config.yaml"
    
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def task1_parameter_optimization(config: dict):
    """任务1: Proportional策略参数优化"""
    print("\n" + "="*70)
    print("任务1: Proportional策略参数优化")
    print("="*70)
    
    # 创建优化器
    optimizer = ParameterOptimizer(config)
    
    # 定义环境工厂函数
    def env_factory(config_path=None, scenario='default'):
        return BikeRebalancingEnv(
            config_dict=config,
            scenario=scenario
        )
    
    # 执行优化
    scenarios = ['default', 'sunny_weekday', 'rainy_weekend']
    results = optimizer.optimize_proportional_policy(env_factory, scenarios)
    
    # 保存结果
    results_dir = project_root / "results"
    results_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 保存最优参数
    best_params_file = results_dir / f"best_proportional_params_{timestamp}.yaml"
    with open(best_params_file, 'w') as f:
        yaml.dump(results['best_params'], f)
    
    # 保存详细结果
    results_df = pd.DataFrame(results['all_results'])
    results_csv = results_dir / f"proportional_optimization_{timestamp}.csv"
    results_df.to_csv(results_csv, index=False)
    
    print(f"✅ 最优参数已保存: {best_params_file}")
    print(f"✅ 详细结果已保存: {results_csv}")
    
    return results


def task2_mincost_debugging(config: dict):
    """任务2: Min-Cost策略调试"""
    print("\n" + "="*70)
    print("任务2: Min-Cost Flow策略调试")
    print("="*70)
    
    scenarios = ['default', 'sunny_weekday', 'rainy_weekend']
    results = {}
    
    for scenario in scenarios:
        print(f"\n测试场景: {scenario}")
        
        # 创建环境和策略
        env = BikeRebalancingEnv(config_dict=config, scenario=scenario)
        policy = MinCostFlowPolicy(config, threshold=0.05)
        
        # 运行3个episode
        episode_results = []
        for seed in [42, 43, 44]:
            obs, _ = env.reset(seed=seed)
            policy.reset_stats()
            
            done = False
            while not done:
                action = policy.select_action(obs)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
            
            episode_results.append({
                'service_rate': info['service_rate'],
                'net_profit': info['net_profit'],
                'total_cost': info['total_cost']
            })
        
        # 计算平均结果
        avg_service_rate = np.mean([r['service_rate'] for r in episode_results])
        avg_net_profit = np.mean([r['net_profit'] for r in episode_results])
        
        print(f"  平均服务率: {avg_service_rate:.2%}")
        print(f"  平均净利润: ${avg_net_profit:.2f}")
        
        results[scenario] = {
            'avg_service_rate': avg_service_rate,
            'avg_net_profit': avg_net_profit,
            'episodes': episode_results
        }
    
    # 保存结果
    results_dir = project_root / "results"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    debug_file = results_dir / f"mincost_debug_{timestamp}.yaml"
    
    with open(debug_file, 'w') as f:
        yaml.dump(results, f)
    
    print(f"\n✅ 调试结果已保存: {debug_file}")
    
    return results


def task3_multi_scenario_evaluation(config: dict, best_params: dict = None):
    """任务3: 多场景评估"""
    print("\n" + "="*70)
    print("任务3: 多场景策略评估")
    print("="*70)
    
    # 使用优化后的参数或默认参数
    if best_params:
        threshold = best_params['threshold']
        ratio = best_params['rebalance_ratio']
        print(f"使用优化参数: threshold={threshold:.2f}, ratio={ratio:.2f}\n")
    else:
        threshold = 0.1
        ratio = 0.5
        print(f"使用默认参数: threshold={threshold:.2f}, ratio={ratio:.2f}\n")
    
    # 评估场景
    scenarios = ['default', 'sunny_weekday', 'rainy_weekend', 'summer_peak', 'winter_low']
    
    # 策略配置
    policies_config = [
        {'name': 'Zero-Action', 'class': ZeroActionPolicy, 'params': {}},
        {'name': 'Proportional-Optimized', 'class': ProportionalRefillPolicy, 
         'params': {'threshold': threshold, 'rebalance_ratio': ratio}},
        {'name': 'Min-Cost-Flow', 'class': MinCostFlowPolicy, 'params': {'threshold': 0.05}}
    ]
    
    # 评估结果
    all_results = []
    
    for scenario in scenarios:
        print(f"\n{'─'*50}")
        print(f"场景: {scenario}")
        print(f"{'─'*50}")
        
        env = BikeRebalancingEnv(config_dict=config, scenario=scenario)
        
        for policy_cfg in policies_config:
            policy_name = policy_cfg['name']
            policy_class = policy_cfg['class']
            policy_params = policy_cfg['params']
            
            print(f"  策略: {policy_name}...", end=' ')
            
            # 创建策略
            policy = policy_class(config, **policy_params)
            
            # 运行3个episode
            episode_metrics = []
            for seed in [42, 43, 44]:
                obs, _ = env.reset(seed=seed)
                policy.reset_stats()
                
                done = False
                while not done:
                    action = policy.select_action(obs)
                    obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                
                episode_metrics.append({
                    'service_rate': info['service_rate'],
                    'net_profit': info['net_profit'],
                    'total_cost': info['total_cost']
                })
            
            # 计算平均指标
            avg_service_rate = np.mean([m['service_rate'] for m in episode_metrics])
            avg_net_profit = np.mean([m['net_profit'] for m in episode_metrics])
            avg_cost = np.mean([m['total_cost'] for m in episode_metrics])
            
            print(f"服务率={avg_service_rate:.1%}, 净利润=${avg_net_profit:.0f}")
            
            all_results.append({
                'scenario': scenario,
                'policy': policy_name,
                'service_rate': avg_service_rate,
                'net_profit': avg_net_profit,
                'total_cost': avg_cost
            })
    
    # 保存结果
    results_dir = project_root / "results"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    results_df = pd.DataFrame(all_results)
    comparison_file = results_dir / f"multi_scenario_comparison_{timestamp}.csv"
    results_df.to_csv(comparison_file, index=False)
    
    print(f"\n✅ 多场景评估结果已保存: {comparison_file}")
    
    # 生成简单报告
    report_file = results_dir / f"day5_evaluation_report_{timestamp}.md"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("# Day 5 策略评估报告\n\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## 策略对比\n\n")
        f.write(results_df.to_markdown(index=False))
    
    print(f"✅ 评估报告已生成: {report_file}")
    
    return all_results


def main():
    """主函数"""
    print("="*70)
    print("Day 5: 策略参数优化")
    print("="*70)
    
    # 加载配置
    config = load_config()
    
    # 任务1: 参数优化
    optimization_results = task1_parameter_optimization(config)
    best_params = optimization_results['best_params']
    
    # 任务2: Min-Cost调试
    mincost_results = task2_mincost_debugging(config)
    
    # 任务3: 多场景评估
    evaluation_results = task3_multi_scenario_evaluation(config, best_params)
    
    print("\n" + "="*70)
    print("✅ Day 5 所有任务完成！")
    print("="*70)
    print(f"最优Proportional参数: {best_params}")
    print(f"评估了 {len(set([r['scenario'] for r in evaluation_results]))} 个场景")
    print("所有结果已保存到 results/ 目录")
    print("="*70)


if __name__ == "__main__":
    main()