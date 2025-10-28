"""
快速测试：验证还车机制
"""

import sys
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / 'simulator'))

from bike_env import BikeRebalancingEnv

# 创建环境（24小时）
import yaml
config_path = Path(__file__).parent.parent / "config" / "env_config.yaml"
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

config['time']['time_horizon'] = 24

env = BikeRebalancingEnv(config_dict=config, scenario="sunny_weekday")

print("="*60)
print("测试还车机制")
print("="*60)

obs, info = env.reset(seed=42)
print(f"\n初始库存: {obs['inventory'].sum():.1f} (归一化)")
print(f"初始实际库存: {info['total_inventory']:.0f} 辆")

# 运行24小时，Zero-Action
inventory_history = [info['total_inventory']]
service_rate_history = []

for hour in range(24):
    action = np.zeros(env.action_space.shape)
    obs, reward, terminated, truncated, info = env.step(action)
    
    inventory_history.append(info['total_inventory'])
    service_rate_history.append(info['service_rate'])
    
    if hour % 6 == 5:  # 每6小时打印
        print(f"\nHour {hour+1:02d}:")
        print(f"  库存: {info['total_inventory']:.0f} 辆")
        print(f"  服务率: {info['service_rate']*100:.1f}%")
        print(f"  累计需求: {info['total_demand']:.0f}")
        print(f"  已服务: {info['total_served']:.0f}")

print(f"\n{'='*60}")
print("最终统计")
print(f"{'='*60}")
print(f"最终库存: {info['total_inventory']:.0f} 辆 (初始: 800)")
print(f"平均服务率: {np.mean(service_rate_history)*100:.1f}%")
print(f"总需求: {info['total_demand']:.0f}")
print(f"总服务: {info['total_served']:.0f}")
print(f"净利润: ${info['net_profit']:.2f}")

# 判断是否修复成功
if info['total_inventory'] > 600:  # 至少保持75%库存
    print(f"\n✅ 还车机制工作正常！库存维持在较高水平。")
else:
    print(f"\n⚠️  库存仍在下降，可能需要进一步调整参数。")

if np.mean(service_rate_history) > 0.8:
    print(f"✅ 服务率良好（{np.mean(service_rate_history)*100:.1f}%）")
else:
    print(f"⚠️  服务率较低（{np.mean(service_rate_history)*100:.1f}%）")