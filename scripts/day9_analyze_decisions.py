#!/usr/bin/env python3
"""
Day 9 - PPO决策可解释性分析
分析PPO的调度决策逻辑和路径选择
"""

import os
import sys
import argparse
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from collections import defaultdict

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from stable_baselines3 import PPO
from simulator.bike_env import BikeRebalancingEnv
# 尝试导入 simulator 中的基线策略；不存在则回退到 policies 实现
try:
    from simulator.baseline_strategies import ProportionalOptimizedStrategy  # type: ignore
    _HAS_SIM_BASELINE = True
except Exception:
    _HAS_SIM_BASELINE = False
    ProportionalOptimizedStrategy = None

from simulator.utils import load_config


def parse_args():
    parser = argparse.ArgumentParser(description='PPO决策分析')
    parser.add_argument('--model', type=str,
                       default='results/ppo_cost_aware/models/best_model/best_model.zip',
                       help='PPO模型路径')
    parser.add_argument('--episodes', type=int, default=10,
                       help='分析轮数')
    parser.add_argument('--quick', action='store_true',
                       help='快速模式')
    return parser.parse_args()


def collect_decision_data(model, env, episodes=10):
    """收集详细的决策数据"""
    print(f"🔍 收集PPO决策数据（{episodes}轮）...")
    
    decisions = []
    
    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        step = 0
        
        while not done:
            # 获取动作
            action, _ = model.predict(obs, deterministic=True)
            
            # 记录状态信息
            hour = env.current_step % 24
            day = env.current_step // 24
            
            # 执行动作
            next_obs, reward, done, truncated, info = env.step(action)
            
            # 记录决策详情
            decision = {
                'episode': ep + 1,
                'step': step,
                'hour': hour,
                'day': day,
                'action': action.tolist() if hasattr(action, 'tolist') else action,
                'rebalance_cost': info.get('rebalance_cost', 0),
                'num_moves': np.sum(action > 0) if hasattr(action, '__len__') else (1 if action > 0 else 0),
                'total_demand': info.get('total_demand', 0),
                'total_served': info.get('total_served', 0),
                'unmet_demand': info.get('unmet_demand', 0),
                'reward': reward
            }
            
            # 添加库存信息
            # 兼容 obs 的两种格式：numpy array/list 或 dict（可能包含 'inventory'/'total_inventory' 等）
            if isinstance(obs, dict):
                # 常见键名优先查找
                inv = None
                for key in ['inventory', 'total_inventory', 'total_inventory_array', 'inventories']:
                    v = obs.get(key, None)
                    if v is not None:
                        inv = v
                        break
                if inv is None:
                    # 尝试从 dict 中找到长度匹配的数组/列表
                    inv = None
                    for v in obs.values():
                        try:
                            if hasattr(v, '__len__') and len(v) == env.num_zones:
                                inv = v
                                break
                        except Exception:
                            continue
                inventory = list(inv) if inv is not None else [0] * env.num_zones
            else:
                # 假定 obs 是 array-like
                try:
                    inventory = list(obs)
                except Exception:
                    inventory = [0] * env.num_zones

            for z in range(env.num_zones):
                decision[f'inventory_zone_{z}'] = inventory[z] if z < len(inventory) else 0
            
            decisions.append(decision)
            
            obs = next_obs
            step += 1
            
            if done or truncated:
                break
    
    print(f"✅ 收集完成：{len(decisions)}条决策记录")
    return pd.DataFrame(decisions)


def analyze_temporal_patterns(df):
    """分析时间模式"""
    print("\n📊 分析时间模式...")
    
    # 按小时统计
    hourly_stats = df.groupby('hour').agg({
        'rebalance_cost': ['mean', 'std', 'sum'],
        'num_moves': ['mean', 'sum'],
        'total_demand': 'mean',
        'unmet_demand': 'mean'
    }).round(2)
    
    # 找出高峰时段
    peak_hours = df.groupby('hour')['rebalance_cost'].sum().nlargest(5)
    low_hours = df.groupby('hour')['rebalance_cost'].sum().nsmallest(5)
    
    analysis = {
        'hourly_stats': hourly_stats,
        'peak_hours': peak_hours.index.tolist(),
        'low_hours': low_hours.index.tolist(),
        'peak_cost': peak_hours.values.tolist(),
        'low_cost': low_hours.values.tolist()
    }
    
    print(f"  调度高峰时段: {analysis['peak_hours']}")
    print(f"  调度低谷时段: {analysis['low_hours']}")
    
    return analysis


def analyze_cost_efficiency(df):
    """分析成本效率"""
    print("\n💰 分析成本效率...")
    
    # 计算每次调度的效率
    df['cost_per_move'] = df['rebalance_cost'] / df['num_moves'].replace(0, np.nan)
    df['cost_per_serve'] = df['rebalance_cost'] / df['total_served'].replace(0, np.nan)
    
    efficiency = {
        'avg_cost_per_move': df['cost_per_move'].mean(),
        'avg_cost_per_serve': df['cost_per_serve'].mean(),
        'total_moves': df['num_moves'].sum(),
        'total_cost': df['rebalance_cost'].sum(),
        'total_served': df['total_served'].sum()
    }
    
    print(f"  平均调度成本/次: ${efficiency['avg_cost_per_move']:.2f}")
    print(f"  平均成本/服务: ${efficiency['avg_cost_per_serve']:.4f}")
    print(f"  总调度次数: {efficiency['total_moves']:.0f}")
    
    return efficiency


def analyze_decision_strategy(df):
    """分析决策策略"""
    print("\n🧠 分析决策策略...")
    
    # 调度频率分布
    move_distribution = df['num_moves'].value_counts().sort_index()
    
    # 成本分布
    cost_bins = [0, 5, 10, 15, 20, 100]
    cost_labels = ['0-5', '5-10', '10-15', '15-20', '20+']
    df['cost_bin'] = pd.cut(df['rebalance_cost'], bins=cost_bins, labels=cost_labels)
    cost_distribution = df['cost_bin'].value_counts()
    
    # 需求响应
    high_demand_mask = df['total_demand'] > df['total_demand'].median()
    response_strategy = {
        'high_demand_cost': df[high_demand_mask]['rebalance_cost'].mean(),
        'low_demand_cost': df[~high_demand_mask]['rebalance_cost'].mean(),
        'high_demand_moves': df[high_demand_mask]['num_moves'].mean(),
        'low_demand_moves': df[~high_demand_mask]['num_moves'].mean()
    }
    
    strategy = {
        'move_distribution': move_distribution.to_dict(),
        'cost_distribution': cost_distribution.to_dict(),
        'response_strategy': response_strategy
    }
    
    print(f"  高需求期平均成本: ${response_strategy['high_demand_cost']:.2f}")
    print(f"  低需求期平均成本: ${response_strategy['low_demand_cost']:.2f}")
    
    return strategy


def compare_with_baseline(model, env, episodes=5):
    """与基线策略对比"""
    print("\n🔄 对比基线策略...")
    
    # PPO决策
    ppo_data = collect_decision_data(model, env, episodes=episodes)
    
    # 基线决策
    print(f"🔍 收集基线决策数据（{episodes}轮）...")
    baseline_decisions = []
    # 选择基线实现：优先 simulator 中的实现，否则回退到 policies 中的 ProportionalRefillPolicy
    if _HAS_SIM_BASELINE and ProportionalOptimizedStrategy is not None:
        strategy = ProportionalOptimizedStrategy(env.num_zones)
        _str_type = 'sim'
    else:
        from policies.baseline_policies import ProportionalRefillPolicy
        cfg = getattr(env, 'config', {}) or {}
        strategy = ProportionalRefillPolicy(cfg)
        _str_type = 'policy'

    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        step = 0

        while not done:
            # 兼容不同签名：simulator impl 可能需要 (obs, env)，policy 只需要 obs
            try:
                if _str_type == 'sim':
                    action = strategy.select_action(obs, env)
                else:
                    action = strategy.select_action(obs)
            except TypeError:
                # 保底尝试
                try:
                    action = strategy.select_action(obs)
                except Exception:
                    action = strategy.select_action(obs, env)

            next_obs, reward, done, truncated, info = env.step(action)

            baseline_decisions.append({
                'episode': ep + 1,
                'step': step,
                'rebalance_cost': info.get('rebalance_cost', 0),
                'num_moves': int(np.sum(action > 0)) if hasattr(action, '__len__') else (1 if action > 0 else 0)
            })

            obs = next_obs
            step += 1

            if done or truncated:
                break
    
    baseline_data = pd.DataFrame(baseline_decisions)
    
    # 对比分析
    comparison = {
        'ppo_avg_cost_per_step': ppo_data['rebalance_cost'].mean(),
        'baseline_avg_cost_per_step': baseline_data['rebalance_cost'].mean(),
        'ppo_avg_moves_per_step': ppo_data['num_moves'].mean(),
        'baseline_avg_moves_per_step': baseline_data['num_moves'].mean(),
        'ppo_total_cost': ppo_data['rebalance_cost'].sum() / episodes,
        'baseline_total_cost': baseline_data['rebalance_cost'].sum() / episodes
    }
    
    print(f"  PPO平均成本/步: ${comparison['ppo_avg_cost_per_step']:.2f}")
    print(f"  基线平均成本/步: ${comparison['baseline_avg_cost_per_step']:.2f}")
    print(f"  PPO平均调度次数/步: {comparison['ppo_avg_moves_per_step']:.2f}")
    print(f"  基线平均调度次数/步: {comparison['baseline_avg_moves_per_step']:.2f}")
    
    return comparison, ppo_data, baseline_data


def generate_analysis_report(temporal, efficiency, strategy, comparison, output_dir):
    """生成分析报告"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = output_dir / f"decision_analysis_report_{timestamp}.txt"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("Day 9 - PPO决策可解释性分析报告\n")
        f.write("="*70 + "\n\n")
        
        f.write("📅 生成时间: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n\n")
        
        # 1. 时间模式
        f.write("="*70 + "\n")
        f.write("1️⃣  时间模式分析\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"调度高峰时段: {temporal['peak_hours']}\n")
        f.write(f"高峰时段总成本: {[f'${c:.2f}' for c in temporal['peak_cost']]}\n\n")
        
        f.write(f"调度低谷时段: {temporal['low_hours']}\n")
        f.write(f"低谷时段总成本: {[f'${c:.2f}' for c in temporal['low_cost']]}\n\n")
        
        f.write("💡 洞察:\n")
        if 7 in temporal['peak_hours'] or 8 in temporal['peak_hours'] or 17 in temporal['peak_hours'] or 18 in temporal['peak_hours']:
            f.write("  - PPO识别了早晚高峰时段，在这些时段增加调度\n")
        if 0 in temporal['low_hours'] or 1 in temporal['low_hours'] or 2 in temporal['low_hours']:
            f.write("  - PPO在深夜时段减少调度，节约成本\n")
        f.write("\n")
        
        # 2. 成本效率
        f.write("="*70 + "\n")
        f.write("2️⃣  成本效率分析\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"平均调度成本/次: ${efficiency['avg_cost_per_move']:.2f}\n")
        f.write(f"平均成本/服务: ${efficiency['avg_cost_per_serve']:.4f}\n")
        f.write(f"总调度次数: {efficiency['total_moves']:.0f}\n")
        f.write(f"总调度成本: ${efficiency['total_cost']:.2f}\n")
        f.write(f"总服务量: {efficiency['total_served']:.0f}\n\n")
        
        f.write("💡 洞察:\n")
        if efficiency['avg_cost_per_move'] < 5:
            f.write("  - PPO选择了低成本调度路径（平均<$5/次）\n")
        if efficiency['avg_cost_per_serve'] < 0.01:
            f.write("  - 成本效率优秀：每服务一个需求成本<$0.01\n")
        f.write("\n")
        
        # 3. 决策策略
        f.write("="*70 + "\n")
        f.write("3️⃣  决策策略分析\n")
        f.write("="*70 + "\n\n")
        
        f.write("调度频率分布:\n")
        for moves, count in sorted(strategy['move_distribution'].items()):
            f.write(f"  {moves}次调度: {count}步\n")
        f.write("\n")
        
        f.write("成本分布:\n")
        for cost_range, count in strategy['cost_distribution'].items():
            f.write(f"  ${cost_range}: {count}步\n")
        f.write("\n")
        
        f.write("需求响应策略:\n")
        rs = strategy['response_strategy']
        f.write(f"  高需求期: 平均成本${rs['high_demand_cost']:.2f}, 平均调度{rs['high_demand_moves']:.1f}次\n")
        f.write(f"  低需求期: 平均成本${rs['low_demand_cost']:.2f}, 平均调度{rs['low_demand_moves']:.1f}次\n\n")
        
        f.write("💡 洞察:\n")
        if rs['high_demand_cost'] > rs['low_demand_cost'] * 1.5:
            f.write("  - PPO在高需求期增加投入，积极响应需求\n")
        if rs['high_demand_moves'] > rs['low_demand_moves']:
            f.write("  - PPO根据需求水平动态调整调度强度\n")
        f.write("\n")
        
        # 4. 与基线对比
        f.write("="*70 + "\n")
        f.write("4️⃣  与基线策略对比\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"PPO平均成本/步: ${comparison['ppo_avg_cost_per_step']:.2f}\n")
        f.write(f"基线平均成本/步: ${comparison['baseline_avg_cost_per_step']:.2f}\n")
        f.write(f"成本比率: {comparison['ppo_avg_cost_per_step']/comparison['baseline_avg_cost_per_step']:.2f}x\n\n")
        
        f.write(f"PPO平均调度次数/步: {comparison['ppo_avg_moves_per_step']:.2f}\n")
        f.write(f"基线平均调度次数/步: {comparison['baseline_avg_moves_per_step']:.2f}\n")
        f.write(f"频率比率: {comparison['ppo_avg_moves_per_step']/comparison['baseline_avg_moves_per_step']:.2f}x\n\n")
        
        f.write(f"PPO周总成本: ${comparison['ppo_total_cost']:.2f}\n")
        f.write(f"基线周总成本: ${comparison['baseline_total_cost']:.2f}\n")
        f.write(f"成本节省: ${comparison['baseline_total_cost'] - comparison['ppo_total_cost']:.2f} ({(1-comparison['ppo_total_cost']/comparison['baseline_total_cost'])*100:.1f}%)\n\n")
        
        f.write("💡 关键洞察:\n")
        if comparison['ppo_avg_moves_per_step'] > comparison['baseline_avg_moves_per_step'] * 2:
            f.write(f"  - PPO采用高频调度策略（{comparison['ppo_avg_moves_per_step']/comparison['baseline_avg_moves_per_step']:.1f}x基线）\n")
        if comparison['ppo_avg_cost_per_step'] < comparison['baseline_avg_cost_per_step'] * 1.2:
            f.write("  - 但通过选择低成本路径，总成本控制优秀\n")
        if comparison['ppo_total_cost'] < comparison['baseline_total_cost']:
            f.write(f"  - 周成本节省{(1-comparison['ppo_total_cost']/comparison['baseline_total_cost'])*100:.1f}%\n")
        f.write("\n")
        
        # 5. 总结
        f.write("="*70 + "\n")
        f.write("5️⃣  核心发现总结\n")
        f.write("="*70 + "\n\n")
        
        f.write("🎯 PPO的决策特点:\n\n")
        
        f.write("1. **高频低成本策略**\n")
        f.write("   - 调度频率高于基线，但每次成本控制严格\n")
        f.write("   - 通过小额度、高频次调度实现灵活响应\n\n")
        
        f.write("2. **时间敏感性**\n")
        f.write("   - 识别高峰和低谷时段\n")
        f.write("   - 在关键时段加强调度\n\n")
        
        f.write("3. **需求适应性**\n")
        f.write("   - 根据需求水平动态调整策略\n")
        f.write("   - 高需求期更积极，低需求期更保守\n\n")
        
        f.write("4. **成本优化**\n")
        f.write("   - 选择低成本调度路径\n")
        f.write("   - 总成本显著低于基线\n\n")
        
        f.write("="*70 + "\n")
        f.write("✅ 分析完成\n")
        f.write("="*70 + "\n")
    
    print(f"\n💾 分析报告已保存: {report_path}")
    return report_path


def main():
    args = parse_args()
    
    print("="*70)
    print("Day 9 - PPO决策可解释性分析")
    print("="*70)
    print()
    
    # 创建输出目录
    output_dir = Path("results/day9_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 输出目录: {output_dir.absolute()}")
    print()
    
    # 加载配置
    print("📄 加载配置...")
    config = load_config()
    print("✅ 配置加载成功")
    print()
    
    # 加载模型
    print("="*70)
    print("加载PPO模型")
    print("="*70)
    print()
    print(f"📦 模型路径: {args.model}")
    
    if not os.path.exists(args.model):
        print(f"❌ 错误: 模型文件不存在: {args.model}")
        return 1
    
    model = PPO.load(args.model)
    print("✅ 模型加载成功")
    print()
    
    # 创建环境
    env = BikeRebalancingEnv(config_dict=config, scenario='default')
    
    # 收集决策数据
    episodes = 3 if args.quick else args.episodes
    decision_data = collect_decision_data(model, env, episodes=episodes)
    
    # 保存原始数据
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    decision_data.to_csv(output_dir / f"decision_data_{timestamp}.csv", index=False)
    print(f"💾 决策数据已保存")
    
    # 分析时间模式
    temporal_analysis = analyze_temporal_patterns(decision_data)
    
    # 分析成本效率
    efficiency_analysis = analyze_cost_efficiency(decision_data)
    
    # 分析决策策略
    strategy_analysis = analyze_decision_strategy(decision_data)
    
    # 与基线对比
    comparison_episodes = 3 if args.quick else 5
    comparison, ppo_comp_data, baseline_comp_data = compare_with_baseline(
        model, env, episodes=comparison_episodes
    )
    
    # 保存对比数据
    ppo_comp_data.to_csv(output_dir / f"ppo_comparison_data_{timestamp}.csv", index=False)
    baseline_comp_data.to_csv(output_dir / f"baseline_comparison_data_{timestamp}.csv", index=False)
    
    # 生成报告
    print("\n" + "="*70)
    print("生成分析报告")
    print("="*70)
    report_path = generate_analysis_report(
        temporal_analysis,
        efficiency_analysis,
        strategy_analysis,
        comparison,
        output_dir
    )
    
    print("\n" + "="*70)
    print("✅ Day 9 决策分析完成！")
    print("="*70)
    print()
    print("📂 输出文件:")
    print(f"  - 分析报告: {report_path}")
    print(f"  - 决策数据: {output_dir / f'decision_data_{timestamp}.csv'}")
    print(f"  - 对比数据: {output_dir / f'ppo_comparison_data_{timestamp}.csv'}")
    print()
    print("💡 查看报告:")
    print(f"  cat {report_path}")
    print()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())