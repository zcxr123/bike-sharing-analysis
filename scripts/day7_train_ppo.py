#!/usr/bin/env python3
"""
Day 7 - 任务2: PPO训练脚本
训练PPO策略用于共享单车调度
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
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    EvalCallback, 
    CheckpointCallback,
    CallbackList
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from simulator.bike_env import BikeRebalancingEnv


class PPOTrainer:
    """PPO训练器"""
    
    def __init__(self, config_path='config/env_config.yaml'):
        """初始化训练器"""
        
        self.project_root = project_root
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 创建输出目录
        self.output_dir = project_root / 'results' / 'ppo_training'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.models_dir = self.output_dir / 'models'
        self.models_dir.mkdir(exist_ok=True)
        
        self.logs_dir = self.output_dir / 'logs'
        self.logs_dir.mkdir(exist_ok=True)
        
        # 加载配置
        print("="*70)
        print("Day 7 - PPO训练系统")
        print("="*70)
        print(f"\n📁 输出目录: {self.output_dir}")
        print(f"📁 模型目录: {self.models_dir}")
        print(f"📁 日志目录: {self.logs_dir}")
        
        print("\n📄 加载配置...")
        config_full_path = project_root / config_path
        with open(config_full_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        print(f"✅ 配置加载成功: {config_full_path}")
        
    def create_env(self, scenario='default', monitor=True):
        """创建训练环境"""
        print(f"\n🏗️  创建环境 (scenario={scenario})...")
        
        # 创建环境
        env = BikeRebalancingEnv(config_dict=self.config, scenario=scenario)
        
        # 添加Monitor包装（用于记录奖励）
        if monitor:
            log_path = self.logs_dir / f'env_{scenario}'
            log_path.mkdir(exist_ok=True)
            env = Monitor(env, str(log_path))
            print(f"   ✅ Monitor已启用: {log_path}")
        
        print("✅ 环境创建成功")
        return env
    
    def create_model(self, env, hyperparams=None):
        """创建PPO模型"""
        print("\n🤖 创建PPO模型...")
        
        # 默认超参数
        default_hyperparams = {
            'learning_rate': 3e-4,
            'n_steps': 2048,
            'batch_size': 64,
            'n_epochs': 10,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_range': 0.2,
            'ent_coef': 0.01,
            'vf_coef': 0.5,
            'max_grad_norm': 0.5,
            'verbose': 1,
            'tensorboard_log': str(self.logs_dir)
        }
        
        # 合并用户提供的超参数
        if hyperparams:
            default_hyperparams.update(hyperparams)
        
        print("\n超参数配置:")
        for key, value in default_hyperparams.items():
            print(f"   {key}: {value}")
        
        # 创建模型
        model = PPO(
            policy='MultiInputPolicy',
            env=env,
            **default_hyperparams
        )
        
        print("✅ PPO模型创建成功")
        return model
    
    def create_callbacks(self, eval_env):
        """创建训练回调"""
        print("\n📋 配置训练回调...")
        
        callbacks = []
        
        # 1. 评估回调
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=str(self.models_dir / 'best_model'),
            log_path=str(self.logs_dir / 'eval'),
            eval_freq=10000,  # 每10000步评估一次
            n_eval_episodes=5,
            deterministic=True,
            render=False,
            verbose=1
        )
        callbacks.append(eval_callback)
        print("   ✅ 评估回调已配置")
        
        # 2. 检查点回调
        checkpoint_callback = CheckpointCallback(
            save_freq=20000,  # 每20000步保存一次
            save_path=str(self.models_dir / 'checkpoints'),
            name_prefix='ppo_bike'
        )
        callbacks.append(checkpoint_callback)
        print("   ✅ 检查点回调已配置")
        
        return CallbackList(callbacks)
    
    def train(self, total_timesteps=100000, hyperparams=None):
        """训练PPO模型"""
        
        print("\n" + "="*70)
        print("开始训练")
        print("="*70)
        
        # 创建训练和评估环境
        print("\n📦 准备环境...")
        train_env = self.create_env(scenario='default', monitor=True)
        eval_env = self.create_env(scenario='default', monitor=False)
        
        # 创建模型
        model = self.create_model(train_env, hyperparams)
        
        # 创建回调
        callbacks = self.create_callbacks(eval_env)
        
        # 开始训练
        print(f"\n🚀 开始训练 (total_timesteps={total_timesteps})...")
        print("="*70)
        
        try:
            model.learn(
                total_timesteps=total_timesteps,
                callback=callbacks,
                log_interval=10,
                progress_bar=True
            )
            
            print("\n" + "="*70)
            print("✅ 训练完成！")
            print("="*70)
            
            # 保存最终模型
            final_model_path = self.models_dir / f'ppo_final_{self.timestamp}'
            model.save(final_model_path)
            print(f"\n💾 最终模型已保存: {final_model_path}.zip")
            
            return model
            
        except KeyboardInterrupt:
            print("\n\n⚠️  训练被中断")
            
            # 保存中断时的模型
            interrupt_model_path = self.models_dir / f'ppo_interrupted_{self.timestamp}'
            model.save(interrupt_model_path)
            print(f"💾 中断模型已保存: {interrupt_model_path}.zip")
            
            return model
        
        except Exception as e:
            print(f"\n❌ 训练失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def quick_test(self, model):
        """快速测试训练好的模型"""
        print("\n" + "="*70)
        print("快速测试")
        print("="*70)
        
        env = self.create_env(scenario='default', monitor=False)
        
        print("\n运行1个episode...")
        obs, info = env.reset(seed=42)
        
        total_reward = 0
        total_served = 0
        total_cost = 0
        step = 0
        done = False
        
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            total_reward += reward
            total_served += info.get('served', 0)
            total_cost += info.get('rebalance_cost', 0)
            step += 1
        
        service_rate = info.get('service_rate', 0)
        
        print(f"\n测试结果:")
        print(f"   总步数: {step}")
        print(f"   总奖励: {total_reward:.2f}")
        print(f"   服务率: {service_rate*100:.2f}%")
        print(f"   满足需求: {total_served:.0f}")
        print(f"   调度成本: ${total_cost:.2f}")
        
        print("\n✅ 快速测试完成")


def main():
    parser = argparse.ArgumentParser(description='PPO训练脚本')
    parser.add_argument('--timesteps', type=int, default=100000,
                       help='总训练步数 (默认: 100000)')
    parser.add_argument('--lr', type=float, default=3e-4,
                       help='学习率 (默认: 3e-4)')
    parser.add_argument('--n-steps', type=int, default=2048,
                       help='每次更新的步数 (默认: 2048)')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='批大小 (默认: 64)')
    parser.add_argument('--quick-test', action='store_true',
                       help='训练后进行快速测试')
    parser.add_argument('--test-only', type=str, default=None,
                       help='只测试已有模型 (提供模型路径)')
    
    args = parser.parse_args()
    
    # 创建训练器
    trainer = PPOTrainer()
    
    # 只测试模式
    if args.test_only:
        print(f"\n🧪 测试模式: 加载模型 {args.test_only}")
        model = PPO.load(args.test_only)
        trainer.quick_test(model)
        return
    
    # 训练模式
    hyperparams = {
        'learning_rate': args.lr,
        'n_steps': args.n_steps,
        'batch_size': args.batch_size
    }
    
    model = trainer.train(
        total_timesteps=args.timesteps,
        hyperparams=hyperparams
    )
    
    # 快速测试
    if model and args.quick_test:
        trainer.quick_test(model)
    
    print("\n" + "="*70)
    print("Day 7 训练任务完成！")
    print("="*70)
    print("\n📊 查看训练日志:")
    print(f"   tensorboard --logdir {trainer.logs_dir}")
    print("\n📁 输出文件:")
    print(f"   模型: {trainer.models_dir}")
    print(f"   日志: {trainer.logs_dir}")


if __name__ == '__main__':
    main()