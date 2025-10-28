# 运行： python3 scripts/debug_service_rate.py
import yaml
from pathlib import Path
from simulator.bike_env import BikeRebalancingEnv
from stable_baselines3 import PPO

cfg = yaml.safe_load(open(Path('config/env_config.yaml'),'r',encoding='utf-8'))
env = BikeRebalancingEnv(config_dict=cfg, scenario='default')
model = PPO.load('results/ppo_training/models/best_model/best_model.zip')

obs, info = env.reset(seed=0)
print("reset info keys:", list(info.keys()))
total_served = total_demand = 0
for step in range(200):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"step {step} info keys: {list(info.keys())}")
    total_served += info.get('served', 0) + info.get('total_served', 0) + info.get('served_trips', 0)
    total_demand += info.get('demand', 0) + info.get('total_demand', 0) + info.get('requests', 0)
    if terminated or truncated:
        break

print("累计 served =", total_served)
print("累计 demand =", total_demand)
print("raw service rate =", (total_served / total_demand) if total_demand > 0 else None)
env.close()