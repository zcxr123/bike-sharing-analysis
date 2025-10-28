#!/usr/bin/env python3
"""
Day 8 - 超参数调优PPO训练脚本
使用优化的超参数配置进行训练
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
import torch.nn as nn


def parse_args():
    parser = argparse.ArgumentParser(description='超参数调优PPO训练')
    parser.add_argument('--timesteps', type=int, default=150000,
                       help='总训练步数（默认150k，比Day 7的100k更多）')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='学习率（降低以提高稳定性）')
    parser.add_argument('--n-steps', type=int, default=4096,
                       help='每次更新的步数（增加以获得更多经验）')
    parser.add_argument('--batch-size', type=int, default=128,
                       help='批大小（增加以提高梯度估计质量）')
    parser.add_argument('--n-epochs', type=int, default=10,
                       help='每次更新的epoch数')
    parser.add_argument('--gamma', type=float, default=0.99,
                       help='折扣因子')
    parser.add_argument('--gae-lambda', type=float, default=0.95,
                       help='GAE lambda')
    parser.add_argument('--clip-range', type=float, default=0.2,
                       help='PPO clip range')
    parser.add_argument('--ent-coef', type=float, default=0.01,
                       help='熵系数（增加探索性）')
    parser.add_argument('--vf-coef', type=float, default=0.5,
                       help='价值函数系数')
    parser.add_argument('--max-grad-norm', type=float, default=0.5,
                       help='梯度裁剪')
    parser.add_argument('--net-arch', type=str, default='256,256',
                       help='网络结构（逗号分隔，如: 256,256,128）')
    parser.add_argument('--quick-test', action='store_true',
                       help='训练后进行快速测试')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    return parser.parse_args()


def parse_net_arch(arch_str):
    """解析网络结构字符串"""
    return [int(x) for x in arch_str.split(',')]


def quick_test(model, config, episodes=3):
    """快速测试训练的模型"""
    print("\n" + "="*70)
    print("快速测试")
    print("="*70 + "\n")
    
    test_env = BikeRebalancingEnv(config_dict=config, scenario='default')
    
    results = []
    for ep in range(episodes):
        obs, _ = test_env.reset()
        done = False
        episode_revenue = 0
        episode_cost = 0
        episode_served = 0
        episode_demand = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = test_env.step(action)
            
            episode_revenue += info.get('revenue', 0)
            episode_cost += info.get('rebalance_cost', 0)
            episode_served += info.get('total_served', 0)
            episode_demand += info.get('total_demand', 0)
            
            if done or truncated:
                break
        
        service_rate = episode_served / max(episode_demand, 1)
        net_profit = episode_revenue - episode_cost
        
        results.append({
            'service_rate': service_rate,
            'net_profit': net_profit,
            'cost': episode_cost
        })
        
        print(f"  Episode {ep+1}/{episodes}: "
              f"服务率={service_rate*100:.1f}%, "
              f"净利润=${net_profit:.0f}, "
              f"成本=${episode_cost:.0f}")
    
    import numpy as np
    print(f"\n平均表现:")
    print(f"  服务率: {np.mean([r['service_rate'] for r in results])*100:.2f}%")
    print(f"  净利润: ${np.mean([r['net_profit'] for r in results]):.2f}")
    print(f"  调度成本: ${np.mean([r['cost'] for r in results]):.2f}")
    print()


def main():
    args = parse_args()
    
    print("="*70)
    print("Day 8 - 超参数调优PPO训练")
    print("="*70)
    print()
    
    # 创建输出目录
    output_dir = Path("results/ppo_tuned")
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
    net_arch = parse_net_arch(args.net_arch)
    
    print("="*70)
    print("训练配置")
    print("="*70)
    print()
    print("🎯 超参数优化重点:")
    print("  - 更低学习率 → 更稳定训练")
    print("  - 更大batch → 更准确梯度")
    print("  - 更多采样步 → 更多经验")
    print("  - 更长训练 → 更充分学习")
    print()
    print(f"总步数: {args.timesteps:,} (vs Day 7: 100k)")
    print(f"学习率: {args.lr} (vs Day 7: 3e-4)")
    print(f"n_steps: {args.n_steps} (vs Day 7: 2048)")
    print(f"batch_size: {args.batch_size} (vs Day 7: 64)")
    print(f"n_epochs: {args.n_epochs}")
    print(f"gamma: {args.gamma}")
    print(f"gae_lambda: {args.gae_lambda}")
    print(f"clip_range: {args.clip_range}")
    print(f"ent_coef: {args.ent_coef}")
    print(f"vf_coef: {args.vf_coef}")
    print(f"max_grad_norm: {args.max_grad_norm}")
    print(f"网络结构: {net_arch}")
    print(f"随机种子: {args.seed}")
    print()
    
    # 创建环境
    print("="*70)
    print("创建训练环境")
    print("="*70)
    print()
    
    def make_env():
        return BikeRebalancingEnv(config_dict=config, scenario='default')
    
    env = DummyVecEnv([make_env])
    eval_env = DummyVecEnv([make_env])
    
    print("✅ 训练环境创建成功")
    print()
    
    # 创建PPO模型（使用调优的超参数）
    print("="*70)
    print("初始化PPO模型")
    print("="*70)
    print()
    
    policy_kwargs = dict(
        net_arch=dict(pi=net_arch, vf=net_arch),
        activation_fn=nn.ReLU
    )
    
    model = PPO(
        "MultiInputPolicy",
        env,
        learning_rate=args.lr,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=str(logs_dir),
        seed=args.seed
    )
    
    print("✅ 模型初始化成功")
    print(f"📊 网络结构: {net_arch}")
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
        name_prefix=f"ppo_tuned",
        verbose=1
    )
    
    # 开始训练
    print("="*70)
    print("开始训练")
    print("="*70)
    print()
    print(f"🚀 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📊 TensorBoard: tensorboard --logdir {logs_dir.absolute()}")
    print(f"⏱️  预计时间: ~{args.timesteps/1000*0.5:.0f}-{args.timesteps/1000:.0f}分钟")
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
    final_model_path = models_dir / f"ppo_tuned_final_{timestamp}.zip"
    model.save(str(final_model_path))
    print(f"💾 最终模型已保存: {final_model_path}")
    print()
    
    # 快速测试
    if args.quick_test:
        quick_test(model, config, episodes=3)
    
    print("="*70)
    print("✅ Day 8 超参数调优训练完成！")
    print("="*70)
    print()
    print("📂 输出文件:")
    print(f"  - 最佳模型: {models_dir / 'best_model' / 'best_model.zip'}")
    print(f"  - 最终模型: {final_model_path}")
    print(f"  - 训练日志: {logs_dir}")
    print()
    print("🎯 下一步:")
    print("  运行 day8_compare_all.py 对比所有模型的性能")
    print()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())