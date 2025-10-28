"""
检查真实环境是否可用
"""
import sys
sys.path.insert(0, '..')

print("=" * 80)
print("真实环境可用性检查")
print("=" * 80)

# 1. 检查模块导入
print("\n[1/4] 检查模块导入...")
try:
    from simulator.bike_env import BikeRebalancingEnv
    from simulator.demand_sampler import DemandSampler
    print("✅ simulator模块导入成功")
except ImportError as e:
    print(f"❌ 模块导入失败: {e}")
    sys.exit(1)

# 2. 检查lambda_params.pkl
print("\n[2/4] 检查lambda参数文件...")
import pickle
from pathlib import Path

lambda_file = Path('../results/lambda_params.pkl')
if lambda_file.exists():
    try:
        with open(lambda_file, 'rb') as f:
            params = pickle.load(f)
        print(f"✅ lambda_params.pkl 加载成功")
        print(f"   - 参数数量: {len(params)}")
        print(f"   - 参数类型: {type(params)}")
        if isinstance(params, dict):
            print(f"   - 参数键: {list(params.keys())[:5]}...")
    except Exception as e:
        print(f"❌ 文件加载失败: {e}")
        sys.exit(1)
else:
    print(f"❌ 文件不存在: {lambda_file}")
    sys.exit(1)

# 3. 检查配置文件
print("\n[3/4] 检查配置文件...")
import yaml

config_file = Path('../config/env_config.yaml')
if config_file.exists():
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        print(f"✅ env_config.yaml 加载成功")
        print(f"   - 区域数量: {config['zones']['num_zones']}")
        print(f"   - 总单车数: {config['inventory']['total_bikes']}")
        print(f"   - 模拟时长: {config['time']['time_horizon']} 小时")
    except Exception as e:
        print(f"❌ 配置加载失败: {e}")
        sys.exit(1)
else:
    print(f"❌ 配置文件不存在: {config_file}")
    sys.exit(1)

# 4. 测试环境创建
print("\n[4/4] 测试环境创建...")
try:
    env = BikeRebalancingEnv(config)
    print("✅ 环境创建成功")
    
    # 测试reset
    obs, info = env.reset(seed=42)
    print(f"✅ 环境reset成功")
    print(f"   - 观测空间: {obs['inventory'].shape}")
    print(f"   - 当前库存: {obs['inventory']}")
    
    # 测试step
    import numpy as np
    action = np.zeros((config['zones']['num_zones'], config['zones']['num_zones']))
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"✅ 环境step成功")
    print(f"   - 奖励: {reward:.2f}")
    print(f"   - 服务率: {info.get('service_rate', 0)*100:.1f}%")
    
except Exception as e:
    print(f"❌ 环境测试失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 80)
print("✅ 所有检查通过！真实环境完全可用！")
print("=" * 80)
print("\n可以开始运行真实环境评估:")
print("  cd ~/bike-sharing-analysis/policies")
print("  python3 evaluate_baselines.py")