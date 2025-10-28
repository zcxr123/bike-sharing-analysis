#!/usr/bin/env python3
"""
Day 7 - 任务3: PPO评估与对比
评估PPO性能，与Proportional-Optimized基线对比
"""

import sys
import os
from pathlib import Path
from datetime import datetime
import argparse

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import yaml
import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from simulator.bike_env import BikeRebalancingEnv
from policies.baseline_policies import ProportionalRefillPolicy


class PPOEvaluator:
    """PPO评估器"""
    
    def __init__(self, config_path='config/env_config.yaml'):
        """初始化评估器"""
        
        self.project_root = project_root
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 创建输出目录
        self.output_dir = project_root / 'results' / 'ppo_evaluation'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print("="*70)
        print("Day 7 - PPO评估与对比系统")
        print("="*70)
        print(f"\n📁 输出目录: {self.output_dir}")
        
        # 加载配置
        print("\n📄 加载配置...")
        config_full_path = project_root / config_path
        with open(config_full_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        print(f"✅ 配置加载成功")
        
    def evaluate_ppo(self, model_path, scenarios=None, n_episodes=10):
        """评估PPO模型"""
        
        if scenarios is None:
            scenarios = ['default', 'sunny_weekday', 'rainy_weekend', 
                        'summer_peak', 'winter_low']
        
        print("\n" + "="*70)
        print("评估PPO模型")
        print("="*70)
        
        # 加载模型
        print(f"\n📦 加载PPO模型: {model_path}")
        model = PPO.load(model_path)
        print("✅ 模型加载成功")
        
        all_results = []
        
        for scenario in scenarios:
            print(f"\n" + "-"*70)
            print(f"场景: {scenario}")
            print("-"*70)
            
            env = BikeRebalancingEnv(config_dict=self.config, scenario=scenario)
            
            for episode in range(n_episodes):
                obs, info = env.reset(seed=42 + episode)
                
                episode_stats = {
                    'policy': 'PPO',
                    'scenario': scenario,
                    'episode': episode,
                    'total_reward': 0,
                    'total_served': 0,
                    'total_demand': 0,
                    'total_cost': 0
                }
                
                last_info = info
                done = False
                while not done:
                    action, _states = model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = env.step(action)
                    last_info = info
                    done = terminated or truncated
                    episode_stats['total_reward'] += reward

                # 使用 episode 结束时的累计信息（避免重复累加累计值）
                episode_stats['total_served'] = last_info.get('total_served', 0)
                episode_stats['total_demand'] = last_info.get('total_demand', 0)
                episode_stats['total_cost'] = last_info.get('total_cost', 0)
                
                # 计算服务率和净利润
                episode_stats['service_rate'] = (
                    last_info.get('service_rate',
                                  (episode_stats['total_served'] / episode_stats['total_demand'])
                                  if episode_stats['total_demand'] > 0 else 0)
                )
                
                revenue = episode_stats['total_served'] * self.config['economics']['revenue_per_trip']
                episode_stats['net_profit'] = revenue - episode_stats['total_cost']
                
                all_results.append(episode_stats)
                
                print(f"  Episode {episode+1}/{n_episodes}: "
                      f"服务率={episode_stats['service_rate']*100:.1f}%, "
                      f"净利润=${episode_stats['net_profit']:.0f}, "
                      f"成本=${episode_stats['total_cost']:.0f}")
        
        # 转换为DataFrame
        df = pd.DataFrame(all_results)
        
        # 保存详细结果
        detail_path = self.output_dir / f'ppo_detail_{self.timestamp}.csv'
        df.to_csv(detail_path, index=False)
        print(f"\n💾 详细结果已保存: {detail_path}")
        
        return df
    
    def evaluate_baseline(self, scenarios=None, n_episodes=10):
        """评估Proportional-Optimized基线"""
        
        if scenarios is None:
            scenarios = ['default', 'sunny_weekday', 'rainy_weekend', 
                        'summer_peak', 'winter_low']
        
        print("\n" + "="*70)
        print("评估Proportional-Optimized基线")
        print("="*70)
        
        all_results = []
        
        # 最优参数
        threshold = 0.25
        rebalance_ratio = 0.2
        
        for scenario in scenarios:
            print(f"\n" + "-"*70)
            print(f"场景: {scenario}")
            print("-"*70)
            
            env = BikeRebalancingEnv(config_dict=self.config, scenario=scenario)
            policy = ProportionalRefillPolicy(
                self.config, 
                threshold=threshold,
                rebalance_ratio=rebalance_ratio
            )
            
            for episode in range(n_episodes):
                obs, info = env.reset(seed=42 + episode)
                policy.reset()
                
                episode_stats = {
                    'policy': 'Proportional-Optimized',
                    'scenario': scenario,
                    'episode': episode,
                    'total_reward': 0,
                    'total_served': 0,
                    'total_demand': 0,
                    'total_cost': 0
                }
                
                last_info = info
                done = False
                while not done:
                    action = policy.select_action(obs)
                    obs, reward, terminated, truncated, info = env.step(action)
                    last_info = info
                    done = terminated or truncated
                    episode_stats['total_reward'] += reward

                # 使用 episode 结束时的累计信息
                episode_stats['total_served'] = last_info.get('total_served', 0)
                episode_stats['total_demand'] = last_info.get('total_demand', 0)
                episode_stats['total_cost'] = last_info.get('total_cost', 0)
                
                # 计算服务率和净利润
                episode_stats['service_rate'] = (
                    last_info.get('service_rate',
                                  (episode_stats['total_served'] / episode_stats['total_demand'])
                                  if episode_stats['total_demand'] > 0 else 0)
                )
                
                revenue = episode_stats['total_served'] * self.config['economics']['revenue_per_trip']
                episode_stats['net_profit'] = revenue - episode_stats['total_cost']
                
                all_results.append(episode_stats)
                
                print(f"  Episode {episode+1}/{n_episodes}: "
                      f"服务率={episode_stats['service_rate']*100:.1f}%, "
                      f"净利润=${episode_stats['net_profit']:.0f}, "
                      f"成本=${episode_stats['total_cost']:.0f}")
        
        # 转换为DataFrame
        df = pd.DataFrame(all_results)
        
        # 保存详细结果
        detail_path = self.output_dir / f'baseline_detail_{self.timestamp}.csv'
        df.to_csv(detail_path, index=False)
        print(f"\n💾 详细结果已保存: {detail_path}")
        
        return df
    
    def compare_policies(self, ppo_df, baseline_df):
        """对比PPO和基线策略"""
        
        print("\n" + "="*70)
        print("策略对比分析")
        print("="*70)
        
        # 合并数据
        all_df = pd.concat([ppo_df, baseline_df], ignore_index=True)
        
        # 按策略和场景聚合
        comparison = all_df.groupby(['policy', 'scenario']).agg({
            'service_rate': ['mean', 'std'],
            'net_profit': ['mean', 'std'],
            'total_cost': ['mean', 'std']
        }).round(4)
        
        print("\n" + "="*70)
        print("详细对比结果")
        print("="*70)
        print(comparison.to_string())
        
        # 总体对比
        print("\n" + "="*70)
        print("总体性能对比（所有场景平均）")
        print("="*70)
        
        overall = all_df.groupby('policy').agg({
            'service_rate': ['mean', 'std'],
            'net_profit': ['mean', 'std'],
            'total_cost': ['mean', 'std']
        }).round(4)
        
        print(overall.to_string())
        
        # 保存对比结果
        comparison_path = self.output_dir / f'ppo_vs_baseline_{self.timestamp}.csv'
        comparison.to_csv(comparison_path)
        print(f"\n💾 对比结果已保存: {comparison_path}")
        
        # 生成总结
        self.generate_summary(overall, all_df)
        
        return comparison
    
    def generate_summary(self, overall_stats, all_df):
        """生成评估总结报告"""
        
        print("\n" + "="*70)
        print("评估总结")
        print("="*70)
        
        # PPO性能
        ppo_stats = overall_stats.loc['PPO']
        baseline_stats = overall_stats.loc['Proportional-Optimized']
        
        ppo_service = ppo_stats[('service_rate', 'mean')]
        baseline_service = baseline_stats[('service_rate', 'mean')]
        
        ppo_profit = ppo_stats[('net_profit', 'mean')]
        baseline_profit = baseline_stats[('net_profit', 'mean')]
        
        ppo_cost = ppo_stats[('total_cost', 'mean')]
        baseline_cost = baseline_stats[('total_cost', 'mean')]
        
        print(f"\n1️⃣  PPO策略:")
        print(f"   服务率: {ppo_service*100:.2f}%")
        print(f"   净利润: ${ppo_profit:.2f}")
        print(f"   调度成本: ${ppo_cost:.2f}")
        
        print(f"\n2️⃣  Proportional-Optimized基线:")
        print(f"   服务率: {baseline_service*100:.2f}%")
        print(f"   净利润: ${baseline_profit:.2f}")
        print(f"   调度成本: ${baseline_cost:.2f}")
        
        print(f"\n3️⃣  对比结果:")
        
        service_diff = (ppo_service - baseline_service) * 100
        profit_diff = ppo_profit - baseline_profit
        profit_pct = (profit_diff / baseline_profit) * 100 if baseline_profit != 0 else 0
        
        if ppo_service > baseline_service:
            print(f"   ✅ PPO服务率高出 {service_diff:.2f}%")
        elif ppo_service < baseline_service:
            print(f"   ⚠️  PPO服务率低于基线 {-service_diff:.2f}%")
        else:
            print(f"   🤝 PPO与基线服务率相当")
        
        if ppo_profit > baseline_profit:
            print(f"   ✅ PPO净利润高出 ${profit_diff:.2f} ({profit_pct:+.2f}%)")
        elif ppo_profit < baseline_profit:
            print(f"   ⚠️  PPO净利润低于基线 ${-profit_diff:.2f} ({profit_pct:+.2f}%)")
        else:
            print(f"   🤝 PPO与基线净利润相当")
        
        print("\n" + "="*70)
        
        # 保存总结报告
        report_path = self.output_dir / f'evaluation_summary_{self.timestamp}.txt'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("PPO vs Proportional-Optimized 评估总结\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"PPO策略:\n")
            f.write(f"  服务率: {ppo_service*100:.2f}%\n")
            f.write(f"  净利润: ${ppo_profit:.2f}\n")
            f.write(f"  调度成本: ${ppo_cost:.2f}\n\n")
            
            f.write(f"Proportional-Optimized基线:\n")
            f.write(f"  服务率: {baseline_service*100:.2f}%\n")
            f.write(f"  净利润: ${baseline_profit:.2f}\n")
            f.write(f"  调度成本: ${baseline_cost:.2f}\n\n")
            
            f.write(f"对比结果:\n")
            f.write(f"  服务率差异: {service_diff:+.2f}%\n")
            f.write(f"  净利润差异: ${profit_diff:+.2f} ({profit_pct:+.2f}%)\n")
        
        print(f"💾 总结报告已保存: {report_path}")


def main():
    parser = argparse.ArgumentParser(description='PPO评估与对比')
    parser.add_argument('--model', type=str, required=True,
                       help='PPO模型路径（.zip文件）')
    parser.add_argument('--episodes', type=int, default=10,
                       help='每个场景的评估轮数 (默认: 10)')
    parser.add_argument('--scenarios', nargs='+', default=None,
                       help='评估场景列表')
    parser.add_argument('--ppo-only', action='store_true',
                       help='只评估PPO，不对比基线')
    
    args = parser.parse_args()
    
    # 创建评估器
    evaluator = PPOEvaluator()
    
    # 评估PPO
    ppo_df = evaluator.evaluate_ppo(
        model_path=args.model,
        scenarios=args.scenarios,
        n_episodes=args.episodes
    )
    
    if not args.ppo_only:
        # 评估基线
        baseline_df = evaluator.evaluate_baseline(
            scenarios=args.scenarios,
            n_episodes=args.episodes
        )
        
        # 对比
        evaluator.compare_policies(ppo_df, baseline_df)
    
    print("\n" + "="*70)
    print("✅ Day 7 评估任务完成！")
    print("="*70)


if __name__ == '__main__':
    main()