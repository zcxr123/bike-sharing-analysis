"""
环境测试脚本
验证需求采样器和Gym环境是否正常工作

Author: renr
Date: 2025-10-28
"""

import sys
import os
import numpy as np
from pathlib import Path

# ---------------------------------------------------------
# 路径设置（稳）：自动探测项目根目录
# 规则：从当前文件开始向上找，直到找到同时包含
#       simulator/、results/、config/ 的目录
# ---------------------------------------------------------
HERE = Path(__file__).resolve().parent          # …/bike-sharing-analysis/tests

def find_project_root(start: Path) -> Path:
    NEED = {"simulator", "results", "config"}
    p = start
    seen = set()
    for _ in range(10):
        if p in seen:
            break
        seen.add(p)
        if p.is_dir():
            names = {child.name for child in p.iterdir() if child.is_dir()}
            if NEED.issubset(names):
                return p
        if p.parent == p:
            break
        p = p.parent
    return HERE.parent  # 兜底

ROOT = find_project_root(HERE)

# 确保把“项目根目录”加到 sys.path 便于以包名导入
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 规范化关键文件的绝对路径
LAMBDA_PATH = ROOT / "results" / "lambda_params.pkl"
CONFIG_PATH = ROOT / "config" / "env_config.yaml"

# 统一使用包前缀导入
from simulator.demand_sampler import DemandSampler
from simulator.bike_env import BikeRebalancingEnv


def _ensure_lambda_path() -> Path:
    """
    确认 lambda_params.pkl 存在；若默认路径缺失，则在项目根内做一次搜索。
    返回可用的绝对路径；若仍不存在则抛出 FileNotFoundError。
    """
    if LAMBDA_PATH.is_file():
        return LAMBDA_PATH
    candidates = list(ROOT.rglob("lambda_params.pkl"))
    if candidates:
        print(f"ℹ️ 未在默认位置找到，改用检测到的文件: {candidates[0]}")
        return candidates[0]
    raise FileNotFoundError(f"Lambda参数文件不存在: {LAMBDA_PATH}\n"
                            f"已搜索项目根目录: {ROOT}\n"
                            f"建议：先生成该文件，或将其放到 results/ 下。")

def _patch_lambda_path_in_config(cfg: dict, lambda_abs: Path) -> dict:
    """
    将配置中所有指向 lambda_params.pkl 的相对路径改成绝对路径。
    - 若键名包含 'lambda' 且值是字符串并且包含 'lambda_params.pkl'，则替换。
    - 否则若值正好等于该文件名或以该文件名结尾，也替换。
    递归处理子字典与列表。
    """
    def _patch(obj):
        if isinstance(obj, dict):
            newd = {}
            for k, v in obj.items():
                if isinstance(v, (dict, list)):
                    newd[k] = _patch(v)
                elif isinstance(v, str):
                    lowk = str(k).lower()
                    lowv = v.lower()
                    if ("lambda" in lowk and "lambda_params.pkl" in lowv) or \
                       lowv.endswith("lambda_params.pkl") or \
                       lowv == "lambda_params.pkl":
                        newd[k] = str(lambda_abs)
                    else:
                        newd[k] = v
                else:
                    newd[k] = v
            return newd
        elif isinstance(obj, list):
            return [_patch(x) for x in obj]
        else:
            return obj
    return _patch(cfg)


def test_demand_sampler():
    """测试需求采样器"""
    print("\n" + "="*60)
    print("测试1: 需求采样器 (DemandSampler)")
    print("="*60)

    try:
        lambda_path = _ensure_lambda_path()
        print(f"\n🔎 lambda_params 路径: {lambda_path}")

        sampler = DemandSampler(
            lambda_params_path=str(lambda_path),
            zone_weights=[0.25, 0.25, 0.15, 0.15, 0.10, 0.10],
            demand_scale=1.0,
            random_seed=42
        )

        print("\n📊 单次需求采样:")
        demands = sampler.sample_demand(hour=17, season=3, workingday=1, weather=1)
        print(f"  场景: 夏季晴天工作日 17:00")
        print(f"  各区域需求: {demands}")
        print(f"  总需求: {demands.sum():.0f} 单")

        print("\n📈 期望需求:")
        expected = sampler.get_expected_demand(17, 3, 1, 1)
        print(f"  各区域期望: {expected}")
        print(f"  期望总需求: {expected.sum():.2f} 单")

        print("\n📉 需求统计 (1000次采样):")
        stats = sampler.get_demand_statistics(num_samples=1000)
        print(f"  平均总需求: {stats['total_mean']:.2f} 单/小时")
        print(f"  标准差: {stats['std']:.2f}")
        print(f"  需求范围: [{stats['min']:.0f}, {stats['max']:.0f}]")

        print("\n✅ 需求采样器测试通过!")
        return True

    except FileNotFoundError as e:
        print(f"\n❌ 错误: {e}")
        print(f"📁 当前检测到的项目根: {ROOT}")
        print(f"📄 默认lambda路径: {LAMBDA_PATH}")
        print("💡 提示: 需要先生成 lambda_params.pkl 文件，或将其放到 results/ 下")
        return False

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_gym_environment():
    """测试 Gym 环境"""
    print("\n" + "="*60)
    print("测试2: Gym 调度环境 (BikeRebalancingEnv)")
    print("="*60)

    try:
        import yaml
        # 读取 YAML，然后把 lambda 路径改成绝对路径
        with open(CONFIG_PATH, 'r') as f:
            cfg = yaml.safe_load(f) or {}
        lambda_abs = _ensure_lambda_path()
        cfg = _patch_lambda_path_in_config(cfg, lambda_abs)

        # 用修改后的 config_dict 构建环境（避免相对路径问题）
        env = BikeRebalancingEnv(config_path=None, config_dict=cfg, scenario="sunny_weekday")

        # 重置
        print("\n🔄 环境重置:")
        obs, info = env.reset(seed=42)
        print(f"  观测空间类型: {type(obs)}")
        print(f"  库存: {obs['inventory']}")
        print(f"  小时: {obs['hour']}")
        print(f"  初始总库存: {info.get('total_inventory', float(np.sum(obs['inventory']))):.0f}")

        # 动作/观测空间
        print(f"\n🎮 动作空间: {env.action_space}")
        print(f"👀 观测空间: {env.observation_space}")

        # 多步模拟
        print("\n⏭️  执行 5 步模拟:")
        for step in range(5):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)

            print(f"\n  Step {step+1}:")
            print(f"    Reward: {reward:.4f}")
            if 'service_rate' in info:
                print(f"    Service Rate: {info['service_rate']*100:.1f}%")
            if 'served' in info and 'demands' in info:
                print(f"    Served/Demand: {info['served']:.0f}/{info['demands'].sum():.0f}")
            if 'rebalance_cost' in info:
                print(f"    Rebalance Cost: ${info['rebalance_cost']:.2f}")
            print(f"    库存总和: {obs['inventory'].sum():.0f}")

            if terminated or truncated:
                print("\n    ✅ Episode 结束!")
                break

        # 渲染
        print("\n🖼️  环境渲染:")
        env.render()

        # 不同场景
        print("\n🌦️  测试不同场景:")
        scenarios = ["sunny_weekday", "rainy_weekend", "summer_peak"]
        for scenario in scenarios:
            obs, info = env.reset(seed=42, options={'scenario': scenario})
            ti = info.get('total_inventory', float(np.sum(obs['inventory'])))
            print(f"  {scenario}: 初始库存总和 = {ti:.0f}")

        env.close()

        print("\n✅ Gym 环境测试通过!")
        return True

    except FileNotFoundError as e:
        print(f"\n❌ 配置/依赖文件不存在: {e}")
        print(f"📄 期望配置路径: {CONFIG_PATH}")
        print(f"📄 期望lambda路径: {LAMBDA_PATH}")
        return False

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_integration():
    """集成测试：完整 episode"""
    print("\n" + "="*60)
    print("测试3: 集成测试 (完整 Episode)")
    print("="*60)

    try:
        import yaml

        with open(CONFIG_PATH, 'r') as f:
            config = yaml.safe_load(f) or {}

        # 统一修补 lambda 绝对路径
        lambda_abs = _ensure_lambda_path()
        config = _patch_lambda_path_in_config(config, lambda_abs)

        # 修改为 24 小时
        config.setdefault('time', {})
        config['time']['time_horizon'] = 24

        env = BikeRebalancingEnv(config_dict=config, scenario="sunny_weekday")

        print(f"\n🏃 运行完整 Episode (24 小时):")
        obs, info = env.reset(seed=42)

        episode_reward = 0.0
        step_count = 0
        last_info = {}

        while True:
            # Zero-action 基线（不调度）
            action = np.zeros(env.action_space.shape, dtype=np.float32)
            obs, reward, terminated, truncated, last_info = env.step(action)

            episode_reward += reward
            step_count += 1

            if step_count % 6 == 0 and 'hour' in last_info:
                inv = float(np.sum(obs['inventory']))
                sr  = f"{last_info.get('service_rate', 0)*100:.1f}%"
                print(f"  Hour {last_info['hour']:02d}: Service Rate={sr}, Inventory={inv:.0f}")

            if terminated or truncated:
                break

        print(f"\n📊 Episode 统计:")
        print(f"  总步数: {step_count}")
        print(f"  累计奖励: {episode_reward:.2f}")
        if last_info:
            if 'service_rate' in last_info:
                print(f"  服务率: {last_info['service_rate']*100:.1f}%")
            if 'total_demand' in last_info:
                print(f"  总需求: {last_info['total_demand']:.0f}")
            if 'total_served' in last_info:
                print(f"  已服务: {last_info['total_served']:.0f}")
            if 'total_unmet' in last_info:
                print(f"  未满足: {last_info['total_unmet']:.0f}")
            if 'net_profit' in last_info:
                print(f"  净利润: ${last_info['net_profit']:.2f}")

        env.close()

        print("\n✅ 集成测试通过!")
        return True

    except FileNotFoundError as e:
        print(f"\n❌ 配置/依赖文件不存在: {e}")
        print(f"📄 期望配置路径: {CONFIG_PATH}")
        print(f"📄 期望lambda路径: {LAMBDA_PATH}")
        return False

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print("\n" + "🚴"*20)
    print("共享单车调度环境 - 完整测试套件")
    print("🚴"*20)

    print(f"\n🧭 解析到的项目根: {ROOT}")
    print(f"📄 lambda 默认路径: {LAMBDA_PATH}")
    print(f"📄 配置文件路径: {CONFIG_PATH}")

    results = []

    # 测试1: 需求采样器
    results.append(test_demand_sampler())

    # 测试2: Gym 环境（仅在采样器通过后继续）
    if results[-1]:
        results.append(test_gym_environment())

    # 测试3: 集成测试（前两项都通过才继续）
    if all(results):
        results.append(test_integration())

    # 总结
    print("\n" + "="*60)
    print("测试总结")
    print("="*60)

    test_names = [
        "需求采样器",
        "Gym环境",
        "集成测试"
    ]

    for i, (name, result) in enumerate(zip(test_names[:len(results)], results)):
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{i+1}. {name}: {status}")

    if all(results):
        print("\n🎉 所有测试通过！环境已准备就绪。")
        print("\n下一步可以:")
        print("  1. 实现基线策略 (Zero/Proportional/MinCost)")
        print("  2. 开始 RL 训练 (PPO/DQN)")
        print("  3. 进行策略对比评估")
    else:
        print("\n⚠️  部分测试失败，请先解决问题。")
        print("\n常见问题:")
        print("  1. 确保 lambda_params.pkl 文件存在（或把它放到 results/ 下）")
        print("  2. 检查文件路径是否正确（上面已打印解析到的 ROOT/路径）")
        print("  3. 安装必要的依赖: pyyaml, numpy, gymnasium")


if __name__ == "__main__":
    main()
