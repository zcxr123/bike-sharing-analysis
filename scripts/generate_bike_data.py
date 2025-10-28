#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
共享单车数据生成脚本 - 生成10万条订单数据
基于Kaggle hour.csv校准需求模型λ(t)
华盛顿特区Capital Bikeshare场景
"""

import pandas as pd
import numpy as np
from faker import Faker
from datetime import datetime, timedelta
import random
import warnings
import os
warnings.filterwarnings('ignore')

# 设置随机种子保证可复现
np.random.seed(42)
random.seed(42)
fake = Faker('zh_CN')
Faker.seed(42)

# ==================== 配置参数 ====================
TARGET_ORDERS = 100000  # 目标订单数
NUM_USERS = 10000       # 用户数量
NUM_BIKES = 5000        # 单车数量

# 城市区域定义（华盛顿特区真实区域）
# 基于Capital Bikeshare的实际服务区域
ZONES = {
    'A_Capitol_Hill': {'lat': 38.8899, 'lng': -77.0091, 'weight': 0.25},      # 国会山（政府区）
    'B_Downtown': {'lat': 38.9072, 'lng': -77.0369, 'weight': 0.25},          # 市中心商务区
    'C_Georgetown': {'lat': 38.9076, 'lng': -77.0723, 'weight': 0.15},        # 乔治城（商业/居住）
    'D_Dupont_Circle': {'lat': 38.9097, 'lng': -77.0436, 'weight': 0.15},    # 杜邦环岛（居住/商业）
    'E_Shaw': {'lat': 38.9129, 'lng': -77.0262, 'weight': 0.10},             # 肖区（文化区）
    'F_Navy_Yard': {'lat': 38.8764, 'lng': -76.9951, 'weight': 0.10},        # 海军船坞（滨水区）
}

# 获取脚本所在目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_RAW_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw')
DATA_GEN_DIR = os.path.join(PROJECT_ROOT, 'data', 'generated')

# 确保输出目录存在
os.makedirs(DATA_GEN_DIR, exist_ok=True)

print("="*60)
print("共享单车数据生成脚本 v1.0")
print("="*60)
print(f"目标生成: {TARGET_ORDERS:,} 条订单")
print(f"用户数量: {NUM_USERS:,}")
print(f"单车数量: {NUM_BIKES:,}")
print(f"场景城市: 华盛顿特区 (Washington D.C.)")
print("="*60)

# ==================== 1. 生成用户信息 ====================
print("\n[1/4] 生成用户信息...")

def generate_users(n):
    users = []
    for i in range(n):
        user = {
            'user_id': f'USER{i:05d}',
            'gender': random.choice(['男', '女']),
            'age': random.randint(18, 65),
            'member_type': random.choices(['普通', '会员'], weights=[0.7, 0.3])[0]
        }
        users.append(user)
    return pd.DataFrame(users)

df_users = generate_users(NUM_USERS)
print(f"✓ 生成 {len(df_users):,} 个用户")

# ==================== 2. 生成单车信息 ====================
print("\n[2/4] 生成单车信息...")

def generate_bikes(n):
    bikes = []
    for i in range(n):
        bike = {
            'bike_id': f'BIKE{i:05d}',
            'bike_type': random.choices(['普通车', '助力车'], weights=[0.7, 0.3])[0],
            'status': random.choices(['正常', '维修', '损坏'], weights=[0.9, 0.08, 0.02])[0]
        }
        bikes.append(bike)
    return pd.DataFrame(bikes)

df_bikes = generate_bikes(NUM_BIKES)
print(f"✓ 生成 {len(df_bikes):,} 辆单车")

# ==================== 3. 加载Kaggle数据并建立需求模型 ====================
print("\n[3/4] 加载Kaggle数据并校准需求模型...")

# 读取hour.csv
hour_csv_path = os.path.join(DATA_RAW_DIR, 'hour.csv')
df_hour = pd.read_csv(hour_csv_path)
print(f"✓ 加载 {len(df_hour):,} 条小时级数据")

# 计算需求强度基准值（每小时平均需求）
df_hour['cnt_normalized'] = df_hour['cnt'] / df_hour['cnt'].max()

# 按小时、季节、工作日、天气分组计算平均需求强度
demand_model = df_hour.groupby(['hr', 'season', 'workingday', 'weathersit']).agg({
    'cnt': 'mean',
    'cnt_normalized': 'mean'
}).reset_index()

print(f"✓ 构建需求模型: {len(demand_model)} 种场景组合")

# ==================== 4. 生成订单数据 ====================
print("\n[4/4] 生成订单数据（这可能需要1-2分钟）...")

def get_demand_intensity(hour, season, workingday, weathersit):
    """根据时段、季节、工作日、天气获取需求强度"""
    match = demand_model[
        (demand_model['hr'] == hour) & 
        (demand_model['season'] == season) & 
        (demand_model['workingday'] == workingday) & 
        (demand_model['weathersit'] == weathersit)
    ]
    
    if len(match) > 0:
        return match.iloc[0]['cnt']
    else:
        # 如果没有精确匹配，使用小时维度的平均值
        match_hr = demand_model[demand_model['hr'] == hour]
        if len(match_hr) > 0:
            return match_hr['cnt'].mean()
        return 50  # 默认基准值

def generate_random_datetime(start_date, end_date):
    """生成随机日期时间"""
    time_delta = end_date - start_date
    random_days = random.randint(0, time_delta.days)
    random_seconds = random.randint(0, 86400)
    return start_date + timedelta(days=random_days, seconds=random_seconds)

def calculate_distance(lat1, lng1, lat2, lng2):
    """计算两点距离（简化版，单位km）"""
    # 华盛顿特区纬度约38度，1度纬度≈111km，1度经度≈87km
    dlat = (lat2 - lat1) * 111
    dlng = (lng2 - lng1) * 87
    return np.sqrt(dlat**2 + dlng**2)

def calculate_fee(distance, bike_type, duration_minutes):
    """计算费用"""
    base_fee = 2.0
    if bike_type == '助力车':
        base_fee = 3.0
    
    # 距离费用：每公里0.5元
    distance_fee = distance * 0.5
    
    # 时长费用：超过30分钟每分钟0.05元
    time_fee = max(0, (duration_minutes - 30) * 0.05)
    
    total = base_fee + distance_fee + time_fee
    return round(total, 1)

# 生成订单
orders = []
start_date = datetime(2011, 1, 1)
end_date = datetime(2012, 12, 31)

# 按批次生成，避免内存溢出
batch_size = 10000
num_batches = (TARGET_ORDERS + batch_size - 1) // batch_size

for batch_idx in range(num_batches):
    batch_orders = []
    current_batch_size = min(batch_size, TARGET_ORDERS - len(orders))
    
    for i in range(current_batch_size):
        # 生成随机时间
        start_time = generate_random_datetime(start_date, end_date)
        
        # 提取时间特征
        hour = start_time.hour
        month = start_time.month
        weekday = start_time.weekday()
        is_holiday = 1 if weekday in [5, 6] else 0
        is_workingday = 0 if is_holiday else 1
        
        # 季节：1=冬,2=春,3=夏,4=秋
        if month in [12, 1, 2]:
            season = 1
        elif month in [3, 4, 5]:
            season = 2
        elif month in [6, 7, 8]:
            season = 3
        else:
            season = 4
        
        # 天气：1=晴,2=多云,3=小雨,4=大雨
        weathersit = random.choices([1, 2, 3, 4], weights=[0.5, 0.3, 0.15, 0.05])[0]
        
        # 根据需求模型获取需求强度
        demand_base = get_demand_intensity(hour, season, is_workingday, weathersit)
        
        # 添加随机扰动
        demand_factor = max(0.1, np.random.normal(1.0, 0.2))
        
        # 选择起点和终点（考虑需求强度）
        zone_names = list(ZONES.keys())
        zone_weights = [ZONES[z]['weight'] * demand_factor for z in zone_names]
        start_zone = random.choices(zone_names, weights=zone_weights)[0]
        
        # 终点：70%概率去不同区域，30%概率在同区域
        if random.random() < 0.7:
            end_zone = random.choice([z for z in zone_names if z != start_zone])
        else:
            end_zone = start_zone
        
        # 生成起止坐标（在区域中心附近随机偏移）
        start_lat = ZONES[start_zone]['lat'] + np.random.normal(0, 0.01)
        start_lng = ZONES[start_zone]['lng'] + np.random.normal(0, 0.01)
        end_lat = ZONES[end_zone]['lat'] + np.random.normal(0, 0.01)
        end_lng = ZONES[end_zone]['lng'] + np.random.normal(0, 0.01)
        
        # 计算距离和时长
        distance = calculate_distance(start_lat, start_lng, end_lat, end_lng)
        
        # 时长：基于距离 + 随机因素（假设平均速度12km/h）
        base_duration = (distance / 12) * 3600  # 秒
        duration = int(max(300, np.random.normal(base_duration, base_duration * 0.3)))
        
        end_time = start_time + timedelta(seconds=duration)
        
        # 随机选择用户和单车
        user_id = df_users.sample(1).iloc[0]['user_id']
        bike_row = df_bikes[df_bikes['status'] == '正常'].sample(1).iloc[0]
        bike_id = bike_row['bike_id']
        bike_type = bike_row['bike_type']
        
        # 计算费用
        fee = calculate_fee(distance, bike_type, duration / 60)
        
        # 气温、体感温度、湿度、风速（归一化值）
        # 根据季节和天气生成合理值
        if season == 1:  # 冬季
            temp_base = 0.3
        elif season == 2:  # 春季
            temp_base = 0.5
        elif season == 3:  # 夏季
            temp_base = 0.75
        else:  # 秋季
            temp_base = 0.55
        
        temp_norm = max(0, min(1, np.random.normal(temp_base, 0.1)))
        atemp_norm = max(0, min(1, np.random.normal(temp_norm, 0.05)))
        
        # 天气影响湿度
        if weathersit >= 3:
            hum_norm = max(0.5, min(1, np.random.normal(0.75, 0.1)))
        else:
            hum_norm = max(0, min(1, np.random.normal(0.5, 0.15)))
        
        windspeed_norm = max(0, min(1, np.random.normal(0.2, 0.1)))
        
        # 构建订单记录
        order = {
            'order_id': f'ORD{len(orders) + i:06d}',
            'user_id': user_id,
            'bike_id': bike_id,
            'start_time': start_time.strftime('%Y-%m-%d %H:%M:%S'),
            'end_time': end_time.strftime('%Y-%m-%d %H:%M:%S'),
            'duration_s': duration,
            'season': season,
            'yr': start_time.year - 2011,  # 0=2011, 1=2012
            'mnth': month,
            'hr': hour,
            'holiday': is_holiday,
            'weekday': weekday,
            'workingday': is_workingday,
            'weathersit': weathersit,
            'temp_norm': round(temp_norm, 4),
            'atemp_norm': round(atemp_norm, 4),
            'hum_norm': round(hum_norm, 2),
            'windspeed_norm': round(windspeed_norm, 4),
            'start_zone': start_zone,
            'end_zone': end_zone,
            'start_lat': round(start_lat, 6),
            'start_lng': round(start_lng, 6),
            'end_lat': round(end_lat, 6),
            'end_lng': round(end_lng, 6),
            'distance_km': round(distance, 2),
            'fee': fee
        }
        batch_orders.append(order)
    
    orders.extend(batch_orders)
    print(f"  进度: {len(orders):,} / {TARGET_ORDERS:,} ({len(orders)/TARGET_ORDERS*100:.1f}%)")

df_orders = pd.DataFrame(orders)
print(f"✓ 生成 {len(df_orders):,} 条订单")

# ==================== 5. 保存数据 ====================
print("\n[5/5] 保存数据到CSV文件...")

df_users.to_csv(os.path.join(DATA_GEN_DIR, 'user_info_10k.csv'), index=False, encoding='utf-8-sig')
df_bikes.to_csv(os.path.join(DATA_GEN_DIR, 'bike_info_5k.csv'), index=False, encoding='utf-8-sig')
df_orders.to_csv(os.path.join(DATA_GEN_DIR, 'orders_100k.csv'), index=False, encoding='utf-8-sig')

print(f"✓ 用户信息: data/generated/user_info_10k.csv")
print(f"✓ 单车信息: data/generated/bike_info_5k.csv")
print(f"✓ 订单数据: data/generated/orders_100k.csv")

# ==================== 6. 数据质量检查 ====================
print("\n" + "="*60)
print("数据质量检查报告")
print("="*60)

print("\n【订单数据统计】")
print(f"总订单数: {len(df_orders):,}")
print(f"时间范围: {df_orders['start_time'].min()} ~ {df_orders['start_time'].max()}")
print(f"平均时长: {df_orders['duration_s'].mean()/60:.1f} 分钟")
print(f"平均距离: {df_orders['distance_km'].mean():.2f} km")
print(f"平均费用: ${df_orders['fee'].mean():.2f}")

print("\n【季节分布】")
season_map = {1: '冬季', 2: '春季', 3: '夏季', 4: '秋季'}
season_dist = df_orders['season'].value_counts().sort_index()
for season, count in season_dist.items():
    print(f"  {season_map[season]}: {count:,} ({count/len(df_orders)*100:.1f}%)")

print("\n【天气分布】")
weather_map = {1: '晴天', 2: '多云', 3: '小雨/雪', 4: '大雨/雪'}
weather_dist = df_orders['weathersit'].value_counts().sort_index()
for weather, count in weather_dist.items():
    print(f"  {weather_map[weather]}: {count:,} ({count/len(df_orders)*100:.1f}%)")

print("\n【工作日 vs 假日】")
print(f"  工作日: {(df_orders['workingday']==1).sum():,} ({(df_orders['workingday']==1).sum()/len(df_orders)*100:.1f}%)")
print(f"  假日: {(df_orders['workingday']==0).sum():,} ({(df_orders['workingday']==0).sum()/len(df_orders)*100:.1f}%)")

print("\n【小时分布（Top 5高峰时段）】")
hour_dist = df_orders['hr'].value_counts().head(5)
for hour, count in hour_dist.items():
    print(f"  {hour:02d}:00 - {count:,} 订单")

print("\n【区域分布（起点）】")
zone_dist = df_orders['start_zone'].value_counts()
for zone, count in zone_dist.items():
    print(f"  {zone}: {count:,} ({count/len(df_orders)*100:.1f}%)")

print("\n【单车类型分布】")
bike_types = df_orders['bike_id'].map(df_bikes.set_index('bike_id')['bike_type'])
bike_type_dist = bike_types.value_counts()
for bike_type, count in bike_type_dist.items():
    print(f"  {bike_type}: {count:,} ({count/len(df_orders)*100:.1f}%)")

print("\n" + "="*60)
print("✅ 数据生成完成！")
print("="*60)
print("\n下一步操作：")
print("1. 查看生成的CSV文件: ls -lh data/generated/")
print("2. 上传到HDFS: hdfs dfs -put data/generated/orders_100k.csv /bike_data/raw/")
print("3. 开始需求模型拟合和分析")
print("="*60)
