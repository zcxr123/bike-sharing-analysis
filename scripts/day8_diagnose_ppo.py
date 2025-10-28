#!/usr/bin/env python3
"""
Day 8 - PPO诊断分析脚本
分析PPO的调度决策模式，找出过度调度的原因
"""

import os
import sys
import argparse
from datetime import datetime
import numpy as np
import pandas as pd
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from stable_baselines3 import PPO
from simulator.bike_env import BikeRebalancingEnv
# 尝试导入 simulator 中的 baseline 实现；不存在则回退到 policies 中的实现
try:
    from simulator.baseline_strategies import ProportionalOptimizedStrategy  # type: ignore
    _HAS_SIM_BASELINE = True
except Exception:
    _HAS_SIM_BASELINE = False
    ProportionalOptimizedStrategy = None

from simulator.utils import load_config


def parse_args():
    parser = argparse.ArgumentParser(description='PPO诊断分析')
    parser.add_argument('--model', type=str, required=True,
                       help='PPO模型路径')
    parser.add_argument('--episodes', type=int, default=5,
                       help='评估轮数')
    parser.add_argument('--quick', action='store_true',
                       help='快速模式（只评估核心指标）')
    return parser.parse_args()


def collect_action_details(env, policy, episodes=5):
    """收集PPO的详细调度决策"""
    action_data = []
    
    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        step = 0
        
        while not done:
            action, _ = policy.predict(obs, deterministic=True)
            next_obs, reward, done, truncated, info = env.step(action)
            
            # 记录调度详情
            rebalance_cost = info.get('rebalance_cost', 0)
            num_rebalances = np.sum(action > 0) if hasattr(action, '__len__') else (1 if action > 0 else 0)
            
            action_data.append({
                'episode': ep + 1,
                'step': step,
                'hour': info.get('hour', 0),
                'rebalance_cost': rebalance_cost,
                'num_rebalances': num_rebalances,
                'total_demand': info.get('total_demand', 0),
                'total_served': info.get('total_served', 0),
                'service_rate': info.get('service_rate', 0),
                'reward': reward
            })
            
            obs = next_obs
            step += 1
            
            if done or truncated:
                break
    
    return pd.DataFrame(action_data)


def collect_baseline_details(env, episodes=5):
    """收集基线策略的详细调度决策（支持 simulator 或 policies 回退）"""
    action_data = []
    if _HAS_SIM_BASELINE and ProportionalOptimizedStrategy is not None:
        strategy = ProportionalOptimizedStrategy(env.num_zones)
        _strategy_type = 'simulator'
    else:
        from policies.baseline_policies import ProportionalRefillPolicy
        # 使用 env 内部的 config（已由 BikeRebalancingEnv 解析）
        cfg = getattr(env, "config", None)
        strategy = ProportionalRefillPolicy(cfg if isinstance(cfg, dict) else {})
        _strategy_type = 'policy'
    
    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        step = 0
        
        while not done:
            try:
                if _strategy_type == 'simulator':
                    action = strategy.select_action(obs, env)
                else:
                    action = strategy.select_action(obs)
            except TypeError:
                # 容错：尝试两种签名
                try:
                    action = strategy.select_action(obs)
                except Exception:
                    action = strategy.select_action(obs, env)
            next_obs, reward, done, truncated, info = env.step(action)
            
            rebalance_cost = info.get('rebalance_cost', 0)
            num_rebalances = np.sum(action > 0) if hasattr(action, '__len__') else (1 if action > 0 else 0)
            
            action_data.append({
                'episode': ep + 1,
                'step': step,
                'hour': info.get('hour', 0),
                'rebalance_cost': rebalance_cost,
                'num_rebalances': num_rebalances,
                'total_demand': info.get('total_demand', 0),
                'total_served': info.get('total_served', 0),
                'service_rate': info.get('service_rate', 0),
                'reward': reward
            })
            
            obs = next_obs
            step += 1
            
            if done or truncated:
                break
    
    return pd.DataFrame(action_data)


def analyze_action_patterns(ppo_data, baseline_data):
    """分析调度模式的差异"""
    analysis = {}
    
    # 1. 调度频率对比
    analysis['ppo_avg_rebalances_per_step'] = ppo_data['num_rebalances'].mean()
    analysis['baseline_avg_rebalances_per_step'] = baseline_data['num_rebalances'].mean()
    analysis['rebalance_frequency_ratio'] = (
        analysis['ppo_avg_rebalances_per_step'] / 
        max(analysis['baseline_avg_rebalances_per_step'], 0.01)
    )
    
    # 2. 成本对比
    analysis['ppo_avg_cost_per_step'] = ppo_data['rebalance_cost'].mean()
    analysis['baseline_avg_cost_per_step'] = baseline_data['rebalance_cost'].mean()
    analysis['cost_ratio'] = (
        analysis['ppo_avg_cost_per_step'] / 
        max(analysis['baseline_avg_cost_per_step'], 0.01)
    )
    
    # 3. 总成本
    analysis['ppo_total_cost'] = ppo_data.groupby('episode')['rebalance_cost'].sum().mean()
    analysis['baseline_total_cost'] = baseline_data.groupby('episode')['rebalance_cost'].sum().mean()
    
    # 4. 服务率对比
    analysis['ppo_avg_service_rate'] = ppo_data['service_rate'].mean()
    analysis['baseline_avg_service_rate'] = baseline_data['service_rate'].mean()
    
    # 5. 成本效率（cost per service rate）
    ppo_total_served = ppo_data.groupby('episode')['total_served'].sum().mean()
    baseline_total_served = baseline_data.groupby('episode')['total_served'].sum().mean()
    
    analysis['ppo_cost_per_serve'] = analysis['ppo_total_cost'] / max(ppo_total_served, 1)
    analysis['baseline_cost_per_serve'] = analysis['baseline_total_cost'] / max(baseline_total_served, 1)
    
    # 6. 时间分布分析
    ppo_hourly = ppo_data.groupby('hour')['rebalance_cost'].mean()
    baseline_hourly = baseline_data.groupby('hour')['rebalance_cost'].mean()
    analysis['ppo_peak_hour'] = ppo_hourly.idxmax()
    analysis['baseline_peak_hour'] = baseline_hourly.idxmax()
    
    return analysis


def generate_diagnosis_report(analysis, output_dir):
    """生成诊断报告"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = output_dir / f"diagnosis_summary_{timestamp}.txt"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("PPO诊断分析报告\n")
        f.write("="*70 + "\n\n")
        
        f.write("1️⃣  调度频率对比\n")
        f.write(f"   PPO平均调度次数/步: {analysis['ppo_avg_rebalances_per_step']:.2f}\n")
        f.write(f"   基线平均调度次数/步: {analysis['baseline_avg_rebalances_per_step']:.2f}\n")
        f.write(f"   频率比率: {analysis['rebalance_frequency_ratio']:.2f}x\n\n")
        
        f.write("2️⃣  成本对比\n")
        f.write(f"   PPO平均成本/步: ${analysis['ppo_avg_cost_per_step']:.2f}\n")
        f.write(f"   基线平均成本/步: ${analysis['baseline_avg_cost_per_step']:.2f}\n")
        f.write(f"   成本比率: {analysis['cost_ratio']:.2f}x\n\n")
        
        f.write("3️⃣  总成本\n")
        f.write(f"   PPO总成本: ${analysis['ppo_total_cost']:.2f}\n")
        f.write(f"   基线总成本: ${analysis['baseline_total_cost']:.2f}\n")
        f.write(f"   成本差异: ${analysis['ppo_total_cost'] - analysis['baseline_total_cost']:.2f}\n\n")
        
        f.write("4️⃣  服务率对比\n")
        f.write(f"   PPO平均服务率: {analysis['ppo_avg_service_rate']*100:.2f}%\n")
        f.write(f"   基线平均服务率: {analysis['baseline_avg_service_rate']*100:.2f}%\n\n")
        
        f.write("5️⃣  成本效率\n")
        f.write(f"   PPO成本/服务: ${analysis['ppo_cost_per_serve']:.4f}\n")
        f.write(f"   基线成本/服务: ${analysis['baseline_cost_per_serve']:.4f}\n")
        f.write(f"   效率比: {analysis['ppo_cost_per_serve']/max(analysis['baseline_cost_per_serve'], 0.0001):.2f}x\n\n")
        
        f.write("="*70 + "\n")
        f.write("🔍 核心发现\n")
        f.write("="*70 + "\n\n")
        
        if analysis['rebalance_frequency_ratio'] > 1.5:
            f.write("⚠️  **过度调度问题**\n")
            f.write(f"   PPO的调度频率是基线的{analysis['rebalance_frequency_ratio']:.1f}倍\n")
            f.write("   建议：增加奖励函数中的成本权重\n\n")
        
        if analysis['cost_ratio'] > 1.5:
            f.write("⚠️  **成本过高问题**\n")
            f.write(f"   PPO的调度成本是基线的{analysis['cost_ratio']:.1f}倍\n")
            f.write("   建议：重新设计奖励函数，增强成本敏感性\n\n")
        
        if abs(analysis['ppo_avg_service_rate'] - analysis['baseline_avg_service_rate']) < 0.005:
            f.write("ℹ️  **服务率相近**\n")
            f.write("   PPO和基线的服务率接近，但成本更高\n")
            f.write("   说明：PPO过度追求服务率，忽视了成本效益\n\n")
        
        f.write("="*70 + "\n")
        f.write("💡 改进建议\n")
        f.write("="*70 + "\n\n")
        
        f.write("1. **奖励函数优化**\n")
        current_cost_weight = 1.0  # 当前权重
        suggested_cost_weight = current_cost_weight * analysis['cost_ratio']
        f.write(f"   当前成本权重: {current_cost_weight}\n")
        f.write(f"   建议成本权重: {suggested_cost_weight:.1f}\n")
        f.write(f"   新奖励函数: reward = revenue - 5.0*penalty - {suggested_cost_weight:.1f}*cost\n\n")
        
        f.write("2. **超参数调整**\n")
        f.write("   - 降低学习率: 3e-4 → 1e-4\n")
        f.write("   - 增加采样步数: 2048 → 4096\n")
        f.write("   - 增加批大小: 64 → 128\n\n")
        
        f.write("3. **训练策略**\n")
        f.write("   - 增加训练步数: 100k → 150k\n")
        f.write("   - 使用多个随机种子\n")
        f.write("   - 监控训练曲线稳定性\n\n")
    
    print(f"💾 诊断报告已保存: {report_path}")
    return report_path


def main():
    args = parse_args()
    
    print("="*70)
    print("Day 8 - PPO诊断分析")
    print("="*70)
    print()
    
    # 创建输出目录
    output_dir = Path("results/ppo_diagnosis")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 输出目录: {output_dir.absolute()}")
    print()
    
    # 加载配置
    print("📄 加载配置...")
    config = load_config()
    print("✅ 配置加载成功")
    print()
    
    # 加载PPO模型
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
    print("="*70)
    print("收集PPO调度数据")
    print("="*70)
    print()
    # 兼容 config_dict 参数：load_config() 返回 dict 时使用 config_dict
    if isinstance(config, dict):
        env = BikeRebalancingEnv(config_dict=config, scenario='default')
    else:
        env = BikeRebalancingEnv(config=config, scenario='default')
    print(f"🔄 运行{args.episodes}轮模拟...")
    ppo_data = collect_action_details(env, model, episodes=args.episodes)
    print(f"✅ PPO数据收集完成: {len(ppo_data)}条记录")
    print()
    
    # 收集基线数据
    print("="*70)
    print("收集基线调度数据")
    print("="*70)
    print()
    if isinstance(config, dict):
        env_baseline = BikeRebalancingEnv(config_dict=config, scenario='default')
    else:
        env_baseline = BikeRebalancingEnv(config=config, scenario='default')
    print(f"🔄 运行{args.episodes}轮模拟...")
    baseline_data = collect_baseline_details(env_baseline, episodes=args.episodes)
    print(f"✅ 基线数据收集完成: {len(baseline_data)}条记录")
    print()
    
    # 保存详细数据
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ppo_data.to_csv(output_dir / f"ppo_actions_{timestamp}.csv", index=False)
    baseline_data.to_csv(output_dir / f"baseline_actions_{timestamp}.csv", index=False)
    print(f"💾 详细数据已保存")
    print()
    
    # 分析模式
    print("="*70)
    print("分析调度模式")
    print("="*70)
    print()
    analysis = analyze_action_patterns(ppo_data, baseline_data)
    
    # 打印关键指标
    print(f"PPO调度频率: {analysis['ppo_avg_rebalances_per_step']:.2f} 次/步")
    print(f"基线调度频率: {analysis['baseline_avg_rebalances_per_step']:.2f} 次/步")
    print(f"频率比率: {analysis['rebalance_frequency_ratio']:.2f}x")
    print()
    print(f"PPO平均成本: ${analysis['ppo_avg_cost_per_step']:.2f}/步")
    print(f"基线平均成本: ${analysis['baseline_avg_cost_per_step']:.2f}/步")
    print(f"成本比率: {analysis['cost_ratio']:.2f}x")
    print()
    print(f"PPO总成本: ${analysis['ppo_total_cost']:.2f}")
    print(f"基线总成本: ${analysis['baseline_total_cost']:.2f}")
    print()
    
    # 生成报告
    print("="*70)
    print("生成诊断报告")
    print("="*70)
    print()
    report_path = generate_diagnosis_report(analysis, output_dir)
    print()
    print("💡 请查看诊断报告了解详细分析和改进建议")
    print()
    
    print("="*70)
    print("✅ Day 8 诊断任务完成！")
    print("="*70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())