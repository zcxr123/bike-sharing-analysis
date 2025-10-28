"""
诊断simulator模块问题
"""
import os
from pathlib import Path

print("=" * 80)
print("Simulator模块诊断")
print("=" * 80)

simulator_dir = Path('simulator')

print(f"\n1. 检查simulator目录:")
print(f"   路径: {simulator_dir.resolve()}")
print(f"   存在: {simulator_dir.exists()}")
print(f"   是目录: {simulator_dir.is_dir()}")

if simulator_dir.exists():
    print(f"\n2. 目录内容:")
    for item in sorted(simulator_dir.iterdir()):
        size = item.stat().st_size if item.is_file() else 0
        print(f"   - {item.name:30s} ({size:>8d} bytes) {'[DIR]' if item.is_dir() else ''}")
    
    print(f"\n3. 检查关键文件:")
    critical_files = ['__init__.py', 'bike_env.py', 'demand_sampler.py']
    for filename in critical_files:
        filepath = simulator_dir / filename
        exists = filepath.exists()
        size = filepath.stat().st_size if exists else 0
        print(f"   - {filename:30s} {'✅' if exists else '❌'} ({size} bytes)")
        
        # 如果文件存在，检查是否为空
        if exists and size == 0:
            print(f"     ⚠️  文件为空！")
        
        # 如果是Python文件，检查语法
        if exists and filename.endswith('.py'):
            try:
                with open(filepath, 'r') as f:
                    compile(f.read(), filename, 'exec')
                print(f"     ✅ Python语法正确")
            except SyntaxError as e:
                print(f"     ❌ 语法错误: {e}")

print(f"\n4. 测试导入:")
try:
    import sys
    sys.path.insert(0, '.')
    from simulator.bike_env import BikeRebalancingEnv
    print(f"   ✅ BikeRebalancingEnv 导入成功")
except Exception as e:
    print(f"   ❌ BikeRebalancingEnv 导入失败: {e}")

try:
    from simulator.demand_sampler import DemandSampler
    print(f"   ✅ DemandSampler 导入成功")
except Exception as e:
    print(f"   ❌ DemandSampler 导入失败: {e}")

print("\n" + "=" * 80)