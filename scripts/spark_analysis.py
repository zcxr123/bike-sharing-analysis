#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Spark数据分析脚本 - 共享单车订单多维度分析
对10万条订单数据进行时间、空间、用户、单车等维度的深度分析
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
import pandas as pd
import os
import json

# 项目路径
PROJECT_ROOT = os.path.expanduser('~/bike-sharing-analysis')
DATA_GEN_DIR = os.path.join(PROJECT_ROOT, 'data', 'generated')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

print("="*70)
print("Spark数据分析 - 共享单车订单多维度分析")
print("="*70)

# ==================== 1. 初始化Spark ====================
print("\n[1/8] 初始化Spark...")

spark = SparkSession.builder \
    .appName("BikeSharing-Analysis") \
    .config("spark.driver.memory", "4g") \
    .config("spark.sql.shuffle.partitions", "8") \
    .getOrCreate()

print(f"✓ Spark版本: {spark.version}")
print(f"✓ Spark UI: http://localhost:4040")

# ==================== 2. 加载数据 ====================
print("\n[2/8] 加载数据...")

# 加载订单数据
orders_path = os.path.join(DATA_GEN_DIR, 'orders_100k.csv')
df_orders = spark.read.csv(orders_path, header=True, inferSchema=True)

# 加载用户数据
users_path = os.path.join(DATA_GEN_DIR, 'user_info_10k.csv')
df_users = spark.read.csv(users_path, header=True, inferSchema=True)

# 加载单车数据
bikes_path = os.path.join(DATA_GEN_DIR, 'bike_info_5k.csv')
df_bikes = spark.read.csv(bikes_path, header=True, inferSchema=True)

print(f"✓ 订单数据: {df_orders.count():,} 条")
print(f"✓ 用户数据: {df_users.count():,} 条")
print(f"✓ 单车数据: {df_bikes.count():,} 条")

# 缓存数据
df_orders.cache()
df_users.cache()
df_bikes.cache()

# ==================== 3. 时间维度分析 ====================
print("\n[3/8] 时间维度分析...")

# 3.1 小时分布
hourly_stats = df_orders.groupBy('hr') \
    .agg(
        count('*').alias('order_count'),
        avg('duration_s').alias('avg_duration'),
        avg('distance_km').alias('avg_distance'),
        avg('fee').alias('avg_fee')
    ) \
    .orderBy('hr')

print("\n【小时分布 Top 5】")
hourly_top5 = hourly_stats.orderBy(desc('order_count')).limit(5)
hourly_top5.show()

# 3.2 工作日 vs 周末
weekday_stats = df_orders.groupBy('workingday') \
    .agg(
        count('*').alias('order_count'),
        avg('duration_s').alias('avg_duration'),
        avg('distance_km').alias('avg_distance')
    ) \
    .withColumn('day_type', when(col('workingday') == 1, 'Weekday').otherwise('Weekend'))

print("\n【工作日 vs 周末对比】")
weekday_stats.select('day_type', 'order_count', 'avg_duration', 'avg_distance').show()

# 3.3 季节分布
season_stats = df_orders.groupBy('season') \
    .agg(
        count('*').alias('order_count'),
        avg('duration_s').alias('avg_duration')
    ) \
    .withColumn('season_name', 
                when(col('season') == 1, 'Winter')
                .when(col('season') == 2, 'Spring')
                .when(col('season') == 3, 'Summer')
                .otherwise('Fall'))

print("\n【季节分布】")
season_stats.select('season_name', 'order_count', 'avg_duration').show()

# 3.4 月度趋势
monthly_stats = df_orders.groupBy('yr', 'mnth') \
    .agg(count('*').alias('order_count')) \
    .withColumn('year', when(col('yr') == 0, 2011).otherwise(2012)) \
    .orderBy('yr', 'mnth')

print("\n【月度趋势（前12个月）】")
monthly_stats.limit(12).show()

# ==================== 4. 天气维度分析 ====================
print("\n[4/8] 天气维度分析...")

weather_stats = df_orders.groupBy('weathersit') \
    .agg(
        count('*').alias('order_count'),
        avg('duration_s').alias('avg_duration'),
        avg('distance_km').alias('avg_distance')
    ) \
    .withColumn('weather_type',
                when(col('weathersit') == 1, 'Clear')
                .when(col('weathersit') == 2, 'Cloudy')
                .when(col('weathersit') == 3, 'Light Rain')
                .otherwise('Heavy Rain'))

print("\n【天气影响分析】")
weather_stats.select('weather_type', 'order_count', 'avg_duration', 'avg_distance').show()

# 温度与需求的关系
temp_bins = df_orders.withColumn('temp_range',
    when(col('temp_norm') < 0.25, 'Cold')
    .when(col('temp_norm') < 0.5, 'Cool')
    .when(col('temp_norm') < 0.75, 'Warm')
    .otherwise('Hot')
)

temp_stats = temp_bins.groupBy('temp_range') \
    .agg(count('*').alias('order_count')) \
    .orderBy('order_count', ascending=False)

print("\n【温度与需求】")
temp_stats.show()

# ==================== 5. 空间维度分析 ====================
print("\n[5/8] 空间维度分析...")

# 5.1 起点区域热度
start_zone_stats = df_orders.groupBy('start_zone') \
    .agg(
        count('*').alias('order_count'),
        avg('distance_km').alias('avg_distance')
    ) \
    .orderBy(desc('order_count'))

print("\n【起点区域热度 Top 6】")
start_zone_stats.show()

# 5.2 终点区域热度
end_zone_stats = df_orders.groupBy('end_zone') \
    .agg(count('*').alias('order_count')) \
    .orderBy(desc('order_count'))

print("\n【终点区域热度 Top 6】")
end_zone_stats.show()

# 5.3 OD流量矩阵（起点-终点）
od_matrix = df_orders.groupBy('start_zone', 'end_zone') \
    .agg(count('*').alias('flow_count')) \
    .orderBy(desc('flow_count'))

print("\n【OD流量 Top 10】")
od_matrix.limit(10).show(truncate=False)

# 5.4 跨区流动分析
cross_zone = df_orders.withColumn('is_cross_zone', 
    when(col('start_zone') != col('end_zone'), 1).otherwise(0)
)

cross_zone_stats = cross_zone.groupBy('is_cross_zone') \
    .agg(count('*').alias('count')) \
    .withColumn('type', when(col('is_cross_zone') == 1, 'Cross-Zone').otherwise('Same-Zone'))

print("\n【跨区 vs 同区】")
cross_zone_stats.select('type', 'count').show()

# ==================== 6. 用户维度分析 ====================
print("\n[6/8] 用户维度分析...")

# 关联用户信息
orders_with_users = df_orders.join(df_users, on='user_id', how='left')

# 6.1 性别分布
gender_stats = orders_with_users.groupBy('gender') \
    .agg(
        count('*').alias('order_count'),
        avg('distance_km').alias('avg_distance'),
        avg('fee').alias('avg_fee')
    )

print("\n【性别分布】")
gender_stats.show()

# 6.2 会员 vs 普通用户
member_stats = orders_with_users.groupBy('member_type') \
    .agg(
        count('*').alias('order_count'),
        avg('duration_s').alias('avg_duration'),
        avg('fee').alias('avg_fee')
    )

print("\n【会员 vs 普通用户】")
member_stats.show()

# 6.3 活跃用户 Top 10
active_users = df_orders.groupBy('user_id') \
    .agg(
        count('*').alias('ride_count'),
        sum('fee').alias('total_spent'),
        avg('distance_km').alias('avg_distance')
    ) \
    .orderBy(desc('ride_count'))

print("\n【活跃用户 Top 10】")
active_users.limit(10).show()

# ==================== 7. 单车维度分析 ====================
print("\n[7/8] 单车维度分析...")

# 关联单车信息
orders_with_bikes = df_orders.join(df_bikes, on='bike_id', how='left')

# 7.1 单车类型对比
bike_type_stats = orders_with_bikes.groupBy('bike_type') \
    .agg(
        count('*').alias('order_count'),
        avg('duration_s').alias('avg_duration'),
        avg('distance_km').alias('avg_distance'),
        avg('fee').alias('avg_fee')
    )

print("\n【单车类型对比】")
bike_type_stats.show()

# 7.2 高频单车 Top 10
freq_bikes = df_orders.groupBy('bike_id') \
    .agg(
        count('*').alias('use_count'),
        sum('distance_km').alias('total_distance'),
        avg('fee').alias('avg_fee')
    ) \
    .orderBy(desc('use_count'))

print("\n【高频单车 Top 10】")
freq_bikes.limit(10).show()

# ==================== 8. 综合统计与导出 ====================
print("\n[8/8] 生成综合报告...")

# 8.1 整体统计
overall_stats = {
    'total_orders': df_orders.count(),
    'total_users': df_users.count(),
    'total_bikes': df_bikes.count(),
    'avg_duration_minutes': df_orders.agg(avg('duration_s')).collect()[0][0] / 60,
    'avg_distance_km': df_orders.agg(avg('distance_km')).collect()[0][0],
    'avg_fee': df_orders.agg(avg('fee')).collect()[0][0],
    'total_revenue': df_orders.agg(sum('fee')).collect()[0][0]
}

print("\n" + "="*70)
print("整体统计摘要")
print("="*70)
for key, value in overall_stats.items():
    if isinstance(value, float):
        print(f"{key:.<40} {value:.2f}")
    else:
        print(f"{key:.<40} {value:,}")

# 8.2 导出分析结果（转为Pandas保存）
print("\n保存分析结果...")

# 转换为Pandas并保存
hourly_stats.toPandas().to_csv(
    os.path.join(RESULTS_DIR, 'hourly_analysis.csv'), index=False
)
season_stats.toPandas().to_csv(
    os.path.join(RESULTS_DIR, 'season_analysis.csv'), index=False
)
weather_stats.toPandas().to_csv(
    os.path.join(RESULTS_DIR, 'weather_analysis.csv'), index=False
)
start_zone_stats.toPandas().to_csv(
    os.path.join(RESULTS_DIR, 'zone_analysis.csv'), index=False
)
od_matrix.toPandas().to_csv(
    os.path.join(RESULTS_DIR, 'od_matrix.csv'), index=False
)
bike_type_stats.toPandas().to_csv(
    os.path.join(RESULTS_DIR, 'bike_type_analysis.csv'), index=False
)

# 保存整体统计为JSON
with open(os.path.join(RESULTS_DIR, 'overall_stats.json'), 'w') as f:
    json.dump(overall_stats, f, indent=2)

print(f"✓ 分析结果已保存到: {RESULTS_DIR}")

# 释放缓存
df_orders.unpersist()
df_users.unpersist()
df_bikes.unpersist()

# 停止Spark
spark.stop()

print("\n" + "="*70)
print("✅ Spark数据分析完成！")
print("="*70)
print("\n生成的分析文件:")
print("  1. hourly_analysis.csv      - 小时分布")
print("  2. season_analysis.csv      - 季节分布")
print("  3. weather_analysis.csv     - 天气影响")
print("  4. zone_analysis.csv        - 区域热度")
print("  5. od_matrix.csv            - OD流量矩阵")
print("  6. bike_type_analysis.csv   - 单车类型对比")
print("  7. overall_stats.json       - 整体统计")
print("\n下一步: 使用Pyecharts进行数据可视化")
print("="*70)