#!/usr/bin/env python3
"""
Day 8 - 综合对比评估脚本
对比Day 7原始PPO、Day 8成本感知PPO、Day 8调优PPO和基线策略
"""

import os
import sys
import argparse
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from stable_baselines3 import PPO
from simulator.bike_env import BikeRebalancingEnv
# 兼容：优先使用 simulator 中的 baseline 实现，否则回退到 policies 实现
try:
    from simulator.baseline_strategies import ProportionalOptimizedStrategy  # type: ignore
    _HAS_SIM_BASELINE = True
except Exception:
    _HAS_SIM_BASELINE = False
    ProportionalOptimizedStrategy = None

from simulator.utils import load_config


def parse_args():
    parser = argparse.ArgumentParser(description='综合对比评估')
    parser.add_argument('--episodes', type=int, default=10,
                       help='每个场景的评估轮数')
    parser.add_argument('--scenarios', nargs='+', 
                       default=['default', 'sunny_weekday', 'rainy_weekend',
                               'summer_peak', 'winter_low'],
                       help='评估场景列表')
    return parser.parse_args()


def evaluate_model(model_path, config, scenario, episodes=10, model_name="Model"):
    """评估单个模型"""
    print(f"\n{'='*70}")
    print(f"评估: {model_name} - {scenario}")
    print(f"{'='*70}\n")
    
    # 加载模型
    if not os.path.exists(model_path):
        print(f"⚠️  模型不存在: {model_path}")
        return None
    
    model = PPO.load(model_path)
    
    # 评估
    results = []
    env = BikeRebalancingEnv(config_dict=config, scenario=scenario)
    
    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        
        episode_data = {
            'revenue': 0,
            'cost': 0,
            'penalty': 0,
            'served': 0,
            'demand': 0
        }
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            
            episode_data['revenue'] += info.get('revenue', 0)
            episode_data['cost'] += info.get('rebalance_cost', 0)
            episode_data['penalty'] += info.get('penalty', 0)
            episode_data['served'] += info.get('total_served', 0)
            episode_data['demand'] += info.get('total_demand', 0)
            
            if done or truncated:
                break
        
        service_rate = episode_data['served'] / max(episode_data['demand'], 1)
        net_profit = episode_data['revenue'] - episode_data['cost']
        
        results.append({
            'model': model_name,
            'scenario': scenario,
            'episode': ep + 1,
            'service_rate': service_rate,
            'net_profit': net_profit,
            'total_cost': episode_data['cost'],
            'revenue': episode_data['revenue'],
            'total_served': episode_data['served'],
            'total_demand': episode_data['demand']
        })
        
        print(f"  Episode {ep+1}/{episodes}: "
              f"服务率={service_rate*100:.1f}%, "
              f"净利润=${net_profit:.0f}, "
              f"成本=${episode_data['cost']:.0f}")
    
    return results


def evaluate_baseline(config, scenario, episodes=10):
    """评估基线策略"""
    print(f"\n{'='*70}")
    print(f"评估: Proportional-Optimized - {scenario}")
    print(f"{'='*70}\n")
    
    env = BikeRebalancingEnv(config_dict=config, scenario=scenario)
    # 选择实现：simulator 提供的 baseline 或回退到 policies 中的实现
    if _HAS_SIM_BASELINE and ProportionalOptimizedStrategy is not None:
        strategy = ProportionalOptimizedStrategy(env.num_zones)
        _strategy_type = 'simulator'
    else:
        from policies.baseline_policies import ProportionalRefillPolicy
        cfg = getattr(env, "config", {}) or {}
        strategy = ProportionalRefillPolicy(cfg)
        _strategy_type = 'policy'
    
    results = []
    
    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        
        episode_data = {
            'revenue': 0,
            'cost': 0,
            'served': 0,
            'demand': 0
        }
        
        while not done:
            try:
                if _strategy_type == 'simulator':
                    action = strategy.select_action(obs, env)
                else:
                    action = strategy.select_action(obs)
            except TypeError:
                try:
                    action = strategy.select_action(obs)
                except Exception:
                    action = strategy.select_action(obs, env)
            obs, reward, done, truncated, info = env.step(action)
            
            episode_data['revenue'] += info.get('revenue', 0)
            episode_data['cost'] += info.get('rebalance_cost', 0)
            episode_data['served'] += info.get('total_served', 0)
            episode_data['demand'] += info.get('total_demand', 0)
            
            if done or truncated:
                break
        
        service_rate = episode_data['served'] / max(episode_data['demand'], 1)
        net_profit = episode_data['revenue'] - episode_data['cost']
        
        results.append({
            'model': 'Proportional-Optimized',
            'scenario': scenario,
            'episode': ep + 1,
            'service_rate': service_rate,
            'net_profit': net_profit,
            'total_cost': episode_data['cost'],
            'revenue': episode_data['revenue'],
            'total_served': episode_data['served'],
            'total_demand': episode_data['demand']
        })
        
        print(f"  Episode {ep+1}/{episodes}: "
              f"服务率={service_rate*100:.1f}%, "
              f"净利润=${net_profit:.0f}, "
              f"成本=${episode_data['cost']:.0f}")
    
    return results


def generate_comparison_report(df, output_dir):
    """生成对比报告"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 计算统计信息
    summary = df.groupby(['model', 'scenario']).agg({
        'service_rate': ['mean', 'std'],
        'net_profit': ['mean', 'std'],
        'total_cost': ['mean', 'std']
    }).round(4)
    
    # 计算总体平均
    overall = df.groupby('model').agg({
        'service_rate': ['mean', 'std'],
        'net_profit': ['mean', 'std'],
        'total_cost': ['mean', 'std']
    }).round(4)
    
    # 生成文本报告
    report_path = output_dir / f"comparison_summary_{timestamp}.txt"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("Day 8 - 策略综合对比报告\n")
        f.write("="*70 + "\n\n")
        
        f.write("评估模型:\n")
        for model in df['model'].unique():
            f.write(f"  - {model}\n")
        f.write(f"\n评估场景: {', '.join(df['scenario'].unique())}\n")
        f.write(f"每场景轮数: {df.groupby(['model', 'scenario']).size().iloc[0]}\n\n")
        
        f.write("="*70 + "\n")
        f.write("详细场景对比\n")
        f.write("="*70 + "\n\n")
        f.write(summary.to_string())
        f.write("\n\n")
        
        f.write("="*70 + "\n")
        f.write("总体性能对比（所有场景平均）\n")
        f.write("="*70 + "\n\n")
        f.write(overall.to_string())
        f.write("\n\n")
        
        # 分析改进效果
        f.write("="*70 + "\n")
        f.write("改进效果分析\n")
        f.write("="*70 + "\n\n")
        
        baseline_profit = overall.loc['Proportional-Optimized', ('net_profit', 'mean')]
        baseline_cost = overall.loc['Proportional-Optimized', ('total_cost', 'mean')]
        baseline_service = overall.loc['Proportional-Optimized', ('service_rate', 'mean')]
        
        for model in overall.index:
            if model == 'Proportional-Optimized':
                continue
            
            model_profit = overall.loc[model, ('net_profit', 'mean')]
            model_cost = overall.loc[model, ('total_cost', 'mean')]
            model_service = overall.loc[model, ('service_rate', 'mean')]
            
            f.write(f"📊 {model} vs Proportional-Optimized:\n")
            
            # 服务率对比
            service_diff = (model_service - baseline_service) * 100
            if abs(service_diff) < 0.5:
                f.write(f"   服务率: {model_service*100:.2f}% ≈ {baseline_service*100:.2f}% "
                       f"(差异: {service_diff:+.2f}%)\n")
            elif service_diff > 0:
                f.write(f"   ✅ 服务率: {model_service*100:.2f}% > {baseline_service*100:.2f}% "
                       f"(提升: {service_diff:+.2f}%)\n")
            else:
                f.write(f"   ⚠️  服务率: {model_service*100:.2f}% < {baseline_service*100:.2f}% "
                       f"(降低: {service_diff:+.2f}%)\n")
            
            # 净利润对比
            profit_diff = model_profit - baseline_profit
            profit_pct = (profit_diff / baseline_profit) * 100
            if abs(profit_pct) < 1:
                f.write(f"   净利润: ${model_profit:.2f} ≈ ${baseline_profit:.2f} "
                       f"({profit_pct:+.2f}%)\n")
            elif profit_diff > 0:
                f.write(f"   ✅ 净利润: ${model_profit:.2f} > ${baseline_profit:.2f} "
                       f"(提升: ${profit_diff:+.2f}, {profit_pct:+.2f}%)\n")
            else:
                f.write(f"   ⚠️  净利润: ${model_profit:.2f} < ${baseline_profit:.2f} "
                       f"(降低: ${profit_diff:+.2f}, {profit_pct:+.2f}%)\n")
            
            # 成本对比
            cost_diff = model_cost - baseline_cost
            cost_pct = (cost_diff / baseline_cost) * 100
            if abs(cost_pct) < 10:
                f.write(f"   成本: ${model_cost:.2f} ≈ ${baseline_cost:.2f} "
                       f"({cost_pct:+.2f}%)\n")
            elif cost_diff < 0:
                f.write(f"   ✅ 成本: ${model_cost:.2f} < ${baseline_cost:.2f} "
                       f"(降低: ${-cost_diff:.2f}, {cost_pct:+.2f}%)\n")
            else:
                f.write(f"   ⚠️  成本: ${model_cost:.2f} > ${baseline_cost:.2f} "
                       f"(增加: ${cost_diff:+.2f}, {cost_pct:+.2f}%)\n")
            
            f.write("\n")
        
        # 总结
        f.write("="*70 + "\n")
        f.write("结论与建议\n")
        f.write("="*70 + "\n\n")
        
        # 找出最佳模型
        best_profit_model = overall['net_profit']['mean'].idxmax()
        best_cost_model = overall['total_cost']['mean'].idxmin()
        best_service_model = overall['service_rate']['mean'].idxmax()
        
        f.write(f"🏆 最高净利润: {best_profit_model} "
               f"(${overall.loc[best_profit_model, ('net_profit', 'mean')]:.2f})\n")
        f.write(f"🏆 最低成本: {best_cost_model} "
               f"(${overall.loc[best_cost_model, ('total_cost', 'mean')]:.2f})\n")
        f.write(f"🏆 最高服务率: {best_service_model} "
               f"({overall.loc[best_service_model, ('service_rate', 'mean')]*100:.2f}%)\n\n")
        
        # 给出建议
        if 'PPO-Day8-CostAware' in overall.index:
            ca_model = overall.loc['PPO-Day8-CostAware']
            ca_cost = ca_model[('total_cost', 'mean')]
            ca_profit = ca_model[('net_profit', 'mean')]
            
            if ca_cost < baseline_cost * 1.2 and ca_profit >= baseline_profit * 0.98:
                f.write("✨ Day 8成本感知训练成功！\n")
                f.write("   成本显著降低，性能接近或超越基线\n\n")
            elif ca_cost < baseline_cost * 1.5:
                f.write("🎯 Day 8成本感知训练有改进\n")
                f.write("   成本有所降低，但仍有优化空间\n")
                f.write("   建议：进一步增加cost_weight或增加训练步数\n\n")
            else:
                f.write("🤔 Day 8成本感知训练效果有限\n")
                f.write("   建议：检查奖励函数设计或尝试其他方法\n\n")
    
    print(f"💾 对比报告已保存: {report_path}")
    return report_path


def main():
    args = parse_args()
    
    print("="*70)
    print("Day 8 - 策略综合对比评估")
    print("="*70)
    print()
    
    # 创建输出目录
    output_dir = Path("results/day8_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 输出目录: {output_dir.absolute()}")
    print()
    
    # 加载配置
    print("📄 加载配置...")
    config = load_config()
    print("✅ 配置加载成功")
    print()
    
    # 定义要对比的模型
    models_to_compare = [
        {
            'name': 'PPO-Day7-Original',
            'path': 'results/ppo_training/models/best_model/best_model.zip'
        },
        {
            'name': 'PPO-Day8-CostAware',
            'path': 'results/ppo_cost_aware/models/best_model/best_model.zip'
        },
        {
            'name': 'PPO-Day8-Tuned',
            'path': 'results/ppo_tuned/models/best_model/best_model.zip'
        }
    ]
    
    # 收集所有结果
    all_results = []
    
    # 评估每个模型
    for model_info in models_to_compare:
        for scenario in args.scenarios:
            results = evaluate_model(
                model_info['path'],
                config,
                scenario,
                episodes=args.episodes,
                model_name=model_info['name']
            )
            if results:
                all_results.extend(results)
    
    # 评估基线
    for scenario in args.scenarios:
        results = evaluate_baseline(config, scenario, episodes=args.episodes)
        all_results.extend(results)
    
    # 转换为DataFrame
    df = pd.DataFrame(all_results)
    
    # 保存详细结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    detail_path = output_dir / f"comparison_detail_{timestamp}.csv"
    df.to_csv(detail_path, index=False)
    print(f"\n💾 详细结果已保存: {detail_path}")
    
    # 生成对比报告
    print("\n" + "="*70)
    print("生成对比报告")
    print("="*70)
    report_path = generate_comparison_report(df, output_dir)
    
    print("\n" + "="*70)
    print("✅ Day 8 综合对比完成！")
    print("="*70)
    print()
    print("📂 输出文件:")
    print(f"  - 详细数据: {detail_path}")
    print(f"  - 对比报告: {report_path}")
    print()
    print("💡 查看报告:")
    print(f"  cat {report_path}")
    print()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())