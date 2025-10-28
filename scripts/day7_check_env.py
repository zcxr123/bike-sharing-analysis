#!/usr/bin/env python3
"""
Day 7 - 任务1: PPO环境兼容性检查
检查BikeRebalancingEnv是否与stable-baselines3兼容
"""

import sys
import os
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import gymnasium as gym
from stable_baselines3.common.env_checker import check_env
import yaml

def check_sb3_compatibility():
    """检查环境与SB3的兼容性"""
    
    print("="*70)
    print("Day 7 - 任务1: PPO环境兼容性检查")
    print("="*70)
    
    # 1. 导入环境
    print("\n📦 步骤1: 导入环境...")
    try:
        from simulator.bike_env import BikeRebalancingEnv
        print("✅ BikeRebalancingEnv导入成功")
    except Exception as e:
        print(f"❌ 环境导入失败: {e}")
        return False
    
    # 2. 加载配置
    print("\n📄 步骤2: 加载配置...")
    try:
        config_path = project_root / 'config' / 'env_config.yaml'
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print(f"✅ 配置加载成功: {config_path}")
    except Exception as e:
        print(f"❌ 配置加载失败: {e}")
        return False
    
    # 3. 创建环境
    print("\n🏗️  步骤3: 创建环境...")
    try:
        env = BikeRebalancingEnv(config_dict=config)
        print("✅ 环境创建成功")
    except Exception as e:
        print(f"❌ 环境创建失败: {e}")
        return False
    
    # 4. 检查空间
    print("\n🔍 步骤4: 检查观察和动作空间...")
    print(f"   观察空间: {env.observation_space}")
    print(f"   动作空间: {env.action_space}")
    
    # 5. SB3兼容性检查
    print("\n✅ 步骤5: SB3兼容性检查...")
    try:
        check_env(env, warn=True)
        print("✅ 环境通过SB3兼容性检查！")
    except Exception as e:
        print(f"❌ SB3兼容性检查失败: {e}")
        return False
    
    # 6. 测试reset和step
    print("\n🧪 步骤6: 测试基本功能...")
    try:
        obs, info = env.reset(seed=42)
        print(f"✅ reset()成功，观察形状: {obs}")
        
        # 测试随机动作
        action = env.action_space.sample()
        print(f"   随机动作形状: {action.shape}")
        
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"✅ step()成功")
        print(f"   reward: {reward:.2f}")
        print(f"   terminated: {terminated}")
        print(f"   truncated: {truncated}")
        
    except Exception as e:
        print(f"❌ 基本功能测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 7. 推荐的Policy类型
    print("\n📋 步骤7: 推荐配置...")
    print("\n推荐的PPO配置:")
    print("  policy='MultiInputPolicy'  # 因为观察空间是Dict")
    print("  learning_rate=3e-4")
    print("  n_steps=2048")
    print("  batch_size=64")
    print("  n_epochs=10")
    print("  gamma=0.99")
    
    print("\n" + "="*70)
    print("✅ 所有检查通过！环境已准备好用于PPO训练")
    print("="*70)
    
    return True


if __name__ == '__main__':
    success = check_sb3_compatibility()
    sys.exit(0 if success else 1)