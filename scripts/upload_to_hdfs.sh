#!/bin/bash
# 上传数据到HDFS

echo "正在上传数据到HDFS..."

# 确保HDFS目录存在
hdfs dfs -mkdir -p /bike_data/raw
hdfs dfs -mkdir -p /bike_data/processed
hdfs dfs -mkdir -p /bike_data/results

# 上传生成的数据
cd ~/bike-sharing-analysis/data/generated

echo "上传订单数据..."
hdfs dfs -put -f orders_100k.csv /bike_data/raw/

echo "上传用户数据..."
hdfs dfs -put -f user_info_10k.csv /bike_data/raw/

echo "上传单车数据..."
hdfs dfs -put -f bike_info_5k.csv /bike_data/raw/

echo "验证上传..."
hdfs dfs -ls /bike_data/raw/

echo "✓ 上传完成！"
