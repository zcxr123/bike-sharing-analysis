#!/usr/bin/env python3
"""
Day 8 - 成本感知PPO训练脚本
使用改进的奖励函数，增加调度成本的惩罚权重
"""

import os
import sys
import argparse
from datetime import datetime
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from simulator.bike_env import BikeRebalancingEnv
from simulator.utils import load_config


def parse_args():
    parser = argparse.ArgumentParser(description='成本感知PPO训练')
    parser.add_argument('--timesteps', type=int, default=100000,
                       help='总训练步数')
    parser.add_argument('--lr', type=float, default=3e-4,
                       help='学习率')
    parser.add_argument('--n-steps', type=int, default=2048,
                       help='每次更新的步数')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='批大小')
    parser.add_argument('--cost-weight', type=float, default=2.0,
                       help='调度成本权重（原始1.0，建议2.0-3.0）')
    parser.add_argument('--penalty-weight', type=float, default=5.0,
                       help='未满足需求惩罚权重')
    parser.add_argument('--revenue-weight', type=float, default=1.0,
                       help='收益权重')
    parser.add_argument('--ent-coef', type=float, default=0.0,
                       help='熵系数（探索性）')
    parser.add_argument('--quick-test', action='store_true',
                       help='训练后进行快速测试')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    return parser.parse_args()


class CostAwareEnv(BikeRebalancingEnv):
    """成本感知的环境包装器"""
    
    def __init__(self, config, scenario='default', 
                 cost_weight=2.0, penalty_weight=5.0, revenue_weight=1.0):
        super().__init__(config_dict=config, scenario=scenario)
        self.cost_weight = cost_weight
        self.penalty_weight = penalty_weight
        self.revenue_weight = revenue_weight
        
        print(f"[CostAwareEnv] 奖励权重配置:")
        print(f"  - 收益权重: {revenue_weight}")
        print(f"  - 成本权重: {cost_weight}")
        print(f"  - 惩罚权重: {penalty_weight}")
        print(f"  - 新奖励函数: {revenue_weight}*revenue - {penalty_weight}*penalty - {cost_weight}*cost")
    
    def step(self, action):
        """重写step方法，使用新的奖励函数"""
        obs, _, done, truncated, info = super().step(action)
        
        # 提取原始奖励组件
        revenue = info.get('revenue', 0)
        penalty = info.get('penalty', 0)
        rebalance_cost = info.get('rebalance_cost', 0)
        
        # 使用新的奖励函数
        new_reward = (
            self.revenue_weight * revenue -
            self.penalty_weight * penalty -
            self.cost_weight * rebalance_cost
        )
        
        # 更新info
        info['reward_components'] = {
            'revenue': revenue,
            'penalty': penalty,
            'cost': rebalance_cost,
            'weighted_reward': new_reward
        }
        
        return obs, new_reward, done, truncated, info


def create_cost_aware_env(config, cost_weight, penalty_weight, revenue_weight):
    """创建成本感知环境"""
    return CostAwareEnv(
        config=config,
        scenario='default',
        cost_weight=cost_weight,
        penalty_weight=penalty_weight,
        revenue_weight=revenue_weight
    )


def quick_test(model, config, episodes=3):
    """快速测试训练的模型"""
    print("\n" + "="*70)
    print("快速测试")
    print("="*70 + "\n")
    
    # 创建测试环境（使用标准环境）
    test_env = BikeRebalancingEnv(config_dict=config, scenario='default')
    
    results = []
    for ep in range(episodes):
        obs, _ = test_env.reset()
        done = False
        episode_reward = 0
        episode_revenue = 0
        episode_cost = 0
        episode_served = 0
        episode_demand = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = test_env.step(action)
            
            episode_reward += reward
            episode_revenue += info.get('revenue', 0)
            episode_cost += info.get('rebalance_cost', 0)
            episode_served += info.get('total_served', 0)
            episode_demand += info.get('total_demand', 0)
            
            if done or truncated:
                break
        
        service_rate = episode_served / max(episode_demand, 1)
        net_profit = episode_revenue - episode_cost
        
        results.append({
            'episode': ep + 1,
            'reward': episode_reward,
            'service_rate': service_rate,
            'net_profit': net_profit,
            'cost': episode_cost
        })
        
        print(f"  Episode {ep+1}/{episodes}: "
              f"服务率={service_rate*100:.1f}%, "
              f"净利润=${net_profit:.0f}, "
              f"成本=${episode_cost:.0f}")
    
    # 计算平均值
    import numpy as np
    avg_service_rate = np.mean([r['service_rate'] for r in results])
    avg_net_profit = np.mean([r['net_profit'] for r in results])
    avg_cost = np.mean([r['cost'] for r in results])
    
    print(f"\n平均表现:")
    print(f"  服务率: {avg_service_rate*100:.2f}%")
    print(f"  净利润: ${avg_net_profit:.2f}")
    print(f"  调度成本: ${avg_cost:.2f}")
    print()
    
    return results


def main():
    args = parse_args()
    
    print("="*70)
    print("Day 8 - 成本感知PPO训练")
    print("="*70)
    print()
    
    # 创建输出目录
    output_dir = Path("results/ppo_cost_aware")
    models_dir = output_dir / "models"
    logs_dir = output_dir / "logs"
    
    for d in [output_dir, models_dir, logs_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 输出目录: {output_dir.absolute()}")
    print()
    
    # 加载配置
    print("📄 加载配置...")
    config = load_config()
    print("✅ 配置加载成功")
    print()
    
    # 显示训练配置
    print("="*70)
    print("训练配置")
    print("="*70)
    print()
    print(f"总步数: {args.timesteps:,}")
    print(f"学习率: {args.lr}")
    print(f"n_steps: {args.n_steps}")
    print(f"batch_size: {args.batch_size}")
    print(f"熵系数: {args.ent_coef}")
    print(f"随机种子: {args.seed}")
    print()
    print("奖励函数权重:")
    print(f"  - 收益: {args.revenue_weight}")
    print(f"  - 成本: {args.cost_weight} (原始1.0，提高{args.cost_weight}x)")
    print(f"  - 惩罚: {args.penalty_weight}")
    print()
    print(f"新奖励函数: reward = {args.revenue_weight}*revenue - {args.penalty_weight}*penalty - {args.cost_weight}*cost")
    print()
    
    # 创建环境
    print("="*70)
    print("创建训练环境")
    print("="*70)
    print()
    
    def make_env():
        return create_cost_aware_env(
            config,
            cost_weight=args.cost_weight,
            penalty_weight=args.penalty_weight,
            revenue_weight=args.revenue_weight
        )
    
    env = DummyVecEnv([make_env])
    print("✅ 训练环境创建成功")
    print()
    
    # 创建评估环境
    eval_env = DummyVecEnv([make_env])
    
    # 创建PPO模型
    print("="*70)
    print("初始化PPO模型")
    print("="*70)
    print()
    
    model = PPO(
        "MultiInputPolicy",
        env,
        learning_rate=args.lr,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        ent_coef=args.ent_coef,
        verbose=1,
        tensorboard_log=str(logs_dir),
        seed=args.seed
    )
    
    print("✅ 模型初始化成功")
    print()
    
    # 创建回调
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(models_dir / "best_model"),
        log_path=str(logs_dir / "eval"),
        eval_freq=max(args.timesteps // 10, 1000),
        n_eval_episodes=5,
        deterministic=True,
        render=False,
        verbose=1
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=max(args.timesteps // 5, 5000),
        save_path=str(models_dir / "checkpoints"),
        name_prefix=f"ppo_cost_aware",
        verbose=1
    )
    
    # 开始训练
    print("="*70)
    print("开始训练")
    print("="*70)
    print()
    print(f"🚀 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📊 可以使用TensorBoard查看训练进度:")
    print(f"   tensorboard --logdir {logs_dir.absolute()}")
    print()
    
    model.learn(
        total_timesteps=args.timesteps,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=True
    )
    
    print()
    print(f"✅ 训练完成: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # 保存最终模型
    final_model_path = models_dir / f"ppo_cost_aware_final_{timestamp}.zip"
    model.save(str(final_model_path))
    print(f"💾 最终模型已保存: {final_model_path}")
    print()
    
    # 快速测试
    if args.quick_test:
        quick_test(model, config, episodes=3)
    
    print("="*70)
    print("✅ Day 8 成本感知训练完成！")
    print("="*70)
    print()
    print("📂 输出文件:")
    print(f"  - 最佳模型: {models_dir / 'best_model' / 'best_model.zip'}")
    print(f"  - 最终模型: {final_model_path}")
    print(f"  - 训练日志: {logs_dir}")
    print()
    print("🎯 下一步:")
    print("  1. 运行评估脚本对比性能")
    print("  2. 如果成本仍高，尝试增加 --cost-weight")
    print("  3. 如果服务率下降，尝试增加 --penalty-weight")
    print()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())