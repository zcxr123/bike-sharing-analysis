# 数据分层设计 SQL脚本
## 共享单车智能调度系统 - 大数据架构

**目的**: 展示如何使用大数据仓库分层设计组织数据  
**框架**: ODS (原始) → DWD (明细) → DWS (汇总) → APP (应用)  
**数据量**: 支持10万-1000万+行扩展  

---

## 一、整体数据流

```
数据来源 (原始订单数据)
    ↓
ODS 层 (保存原始数据，无处理)
    ↓
DWD 层 (数据清洗、去重、验证)
    ↓
DWS 层 (数据汇总、聚合、计算指标)
    ↓
APP 层 (业务应用特征工程)
    ↓
RL 模型训练 + Dashboard 展示
```

---

## 二、ODS层 (Operational Data Store) - 原始数据存储

### 2.1 ODS订单表

```sql
-- ============================================================================
-- ODS 层：原始订单表
-- 特点：最少处理，保留原始数据，用于对账和溯源
-- ============================================================================

CREATE TABLE IF NOT EXISTS ods_orders (
    -- 主键和时间
    order_id STRING COMMENT '订单ID (主键)',
    order_time TIMESTAMP COMMENT '订单发生时间',
    
    -- 地理信息
    city_id INT COMMENT '城市ID',
    region_id INT COMMENT '地区/区域ID',
    pickup_region INT COMMENT '上车区域',
    dropoff_region INT COMMENT '下车区域',
    
    -- 用户和车辆
    user_id STRING COMMENT '用户ID',
    bike_id STRING COMMENT '单车ID',
    bike_type STRING COMMENT '单车类型 (normal/ebike)',
    
    -- 骑行信息
    duration_minutes INT COMMENT '骑行时长(分钟)',
    distance_km DOUBLE COMMENT '骑行距离(公里)',
    
    -- 外部因素
    weather STRING COMMENT '天气 (sunny/rainy/cloudy/snowy)',
    temperature INT COMMENT '温度(摄氏度)',
    season STRING COMMENT '季节 (spring/summer/fall/winter)',
    
    -- 其他
    status STRING COMMENT '订单状态 (completed/cancelled)',
    created_at TIMESTAMP COMMENT '创建时间',
    updated_at TIMESTAMP COMMENT '更新时间'
)
COMMENT '原始订单表 - 数据来自生产系统'
PARTITIONED BY (dt STRING COMMENT '分区日期 YYYY-MM-DD')
STORED AS PARQUET
TBLPROPERTIES (
    'parquet.compression' = 'snappy',
    'comment' = 'ODS层 - 保留原始数据，无处理'
);
```

### 2.2 ODS天气表

```sql
CREATE TABLE IF NOT EXISTS ods_weather (
    weather_date DATE COMMENT '天气日期',
    city_id INT COMMENT '城市ID',
    temperature INT COMMENT '温度',
    humidity INT COMMENT '湿度(%)',
    weather_type STRING COMMENT '天气类型',
    wind_speed INT COMMENT '风速(km/h)',
    created_at TIMESTAMP
)
PARTITIONED BY (dt STRING)
STORED AS PARQUET;
```

### 2.3 ODS区域信息表

```sql
CREATE TABLE IF NOT EXISTS ods_regions (
    region_id INT COMMENT '区域ID',
    city_id INT COMMENT '城市ID',
    region_name STRING COMMENT '区域名称',
    region_type STRING COMMENT '区域类型 (commercial/residential/scenic)',
    poi_count INT COMMENT 'POI(兴趣点)数量',
    created_at TIMESTAMP
)
STORED AS PARQUET;
```

---

## 三、DWD层 (Data Warehouse Detail) - 数据明细层

### 3.1 DWD订单明细表

```sql
-- ============================================================================
-- DWD 层：订单明细表（核心事实表）
-- 特点：已清洗、去重、验证；可用于分析
-- ============================================================================

CREATE TABLE IF NOT EXISTS dwd_order_detail (
    -- 主键
    order_id STRING COMMENT '订单ID',
    
    -- 时间维度 (便于按时间分析)
    order_date DATE COMMENT '订单日期',
    order_hour INT COMMENT '订单小时 (0-23)',
    day_of_week INT COMMENT '周几 (1=星期一, 7=星期日)',
    is_weekend INT COMMENT '是否周末 (0/1)',
    
    -- 空间维度
    city_id INT COMMENT '城市ID',
    region_id INT COMMENT '地区ID',
    pickup_region INT COMMENT '上车地区',
    dropoff_region INT COMMENT '下车地区',
    
    -- 用户和车辆
    user_id STRING COMMENT '用户ID',
    bike_id STRING COMMENT '单车ID',
    bike_type STRING COMMENT '单车类型',
    
    -- 骑行指标
    duration_minutes INT COMMENT '骑行时长(分钟)',
    distance_km DOUBLE COMMENT '骑行距离(公里)',
    
    -- 环境因素
    weather STRING COMMENT '天气',
    season STRING COMMENT '季节',
    
    -- 数据质量标记
    is_valid INT COMMENT '是否有效数据 (1=有效, 0=异常)',
    created_at TIMESTAMP COMMENT '创建时间'
)
COMMENT 'DWD层 - 清洗去重后的订单明细'
PARTITIONED BY (dt STRING)
STORED AS PARQUET
TBLPROPERTIES ('parquet.compression' = 'snappy');
```

### 3.2 DWD单车状态表

```sql
CREATE TABLE IF NOT EXISTS dwd_bike_status (
    bike_id STRING COMMENT '单车ID',
    city_id INT COMMENT '城市ID',
    region_id INT COMMENT '当前区域',
    status_date DATE COMMENT '状态日期',
    status STRING COMMENT '单车状态 (active/broken/maintenance)',
    total_trips INT COMMENT '总骑行次数',
    last_trip_time TIMESTAMP COMMENT '最后骑行时间',
    created_at TIMESTAMP
)
PARTITIONED BY (dt STRING)
STORED AS PARQUET;
```

### 3.3 生成DWD的SQL

```sql
-- ============================================================================
-- 从ODS生成DWD (数据清洗和去重)
-- ============================================================================

INSERT OVERWRITE TABLE dwd_order_detail PARTITION(dt='2025-11-22')
SELECT 
    order_id,
    DATE(order_time) AS order_date,
    HOUR(order_time) AS order_hour,
    DAYOFWEEK(order_time) AS day_of_week,
    CASE WHEN DAYOFWEEK(order_time) >= 6 THEN 1 ELSE 0 END AS is_weekend,
    city_id,
    region_id,
    pickup_region,
    dropoff_region,
    user_id,
    bike_id,
    bike_type,
    duration_minutes,
    distance_km,
    weather,
    season,
    -- 数据质量检查
    CASE 
        WHEN order_id IS NOT NULL
         AND duration_minutes > 0 
         AND distance_km > 0
         AND region_id IS NOT NULL
        THEN 1 
        ELSE 0 
    END AS is_valid,
    created_at
FROM ods_orders
WHERE 
    -- 基础过滤
    order_id IS NOT NULL
    AND order_time IS NOT NULL
    AND status = 'completed'  -- 仅保留完成的订单
    AND DATE(order_time) = '2025-11-22'
-- 去重：每个order_id保留最新的记录
QUALIFY ROW_NUMBER() OVER (PARTITION BY order_id ORDER BY updated_at DESC) = 1;
```

---

## 四、DWS层 (Data Warehouse Summary) - 数据汇总层

### 4.1 DWS小时需求汇总表

```sql
-- ============================================================================
-- DWS 层：小时级需求汇总表
-- 特点：按小时、地区汇总，生成核心业务指标
-- ============================================================================

CREATE TABLE IF NOT EXISTS dws_demand_hourly (
    stat_datetime TIMESTAMP COMMENT '统计时间(小时)',
    stat_date DATE COMMENT '统计日期',
    stat_hour INT COMMENT '统计小时',
    day_of_week INT COMMENT '周几',
    is_weekend INT COMMENT '是否周末',
    
    -- 地区维度
    city_id INT COMMENT '城市ID',
    region_id INT COMMENT '地区ID',
    
    -- 环境维度
    weather STRING COMMENT '天气',
    season STRING COMMENT '季节',
    
    -- 需求指标（核心）
    order_count INT COMMENT '订单数',
    unique_users INT COMMENT '不同用户数',
    unique_bikes INT COMMENT '使用单车数',
    
    -- 骑行特征
    avg_duration DOUBLE COMMENT '平均骑行时长',
    stddev_duration DOUBLE COMMENT '骑行时长标准差',
    p95_duration INT COMMENT '95分位数',
    avg_distance DOUBLE COMMENT '平均骑行距离',
    
    -- 数据质量
    valid_order_count INT COMMENT '有效订单数',
    data_quality_pct DOUBLE COMMENT '数据有效率(%)'
)
COMMENT 'DWS层 - 小时级需求汇总'
PARTITIONED BY (dt STRING)
STORED AS PARQUET;
```

### 4.2 DWS日级汇总表

```sql
CREATE TABLE IF NOT EXISTS dws_order_daily (
    stat_date DATE COMMENT '统计日期',
    city_id INT COMMENT '城市ID',
    region_id INT COMMENT '地区ID',
    
    -- 日级指标
    total_orders INT COMMENT '订单总数',
    total_users INT COMMENT '用户总数',
    total_bikes INT COMMENT '单车总数',
    
    avg_duration DOUBLE COMMENT '平均时长',
    avg_distance DOUBLE COMMENT '平均距离',
    
    -- 时间特征
    peak_hour INT COMMENT '最高峰小时',
    peak_hour_orders INT COMMENT '高峰时段订单数',
    
    -- 收入相关
    estimated_revenue DOUBLE COMMENT '预计收入(订单数×$4)'
)
STORED AS PARQUET;
```

### 4.3 生成DWS的SQL

```sql
-- ============================================================================
-- 从DWD生成DWS (汇总聚合)
-- ============================================================================

INSERT OVERWRITE TABLE dws_demand_hourly PARTITION(dt='2025-11-22')
SELECT 
    -- 时间
    CAST(CONCAT(
        DATE_FORMAT(order_date, 'yyyy-MM-dd'), ' ',
        LPAD(order_hour, 2, '0'), ':00:00'
    ) AS TIMESTAMP) AS stat_datetime,
    order_date AS stat_date,
    order_hour AS stat_hour,
    day_of_week,
    is_weekend,
    
    -- 地区
    city_id,
    region_id,
    
    -- 环境
    weather,
    season,
    
    -- 需求指标
    COUNT(*) AS order_count,
    COUNT(DISTINCT user_id) AS unique_users,
    COUNT(DISTINCT bike_id) AS unique_bikes,
    
    -- 骑行特征
    AVG(duration_minutes) AS avg_duration,
    STDDEV(duration_minutes) AS stddev_duration,
    PERCENTILE_APPROX(duration_minutes, 0.95) AS p95_duration,
    AVG(distance_km) AS avg_distance,
    
    -- 数据质量
    SUM(is_valid) AS valid_order_count,
    ROUND(SUM(is_valid) / COUNT(*) * 100, 2) AS data_quality_pct
    
FROM dwd_order_detail
WHERE dt = '2025-11-22'
GROUP BY 
    order_date,
    order_hour,
    day_of_week,
    is_weekend,
    city_id,
    region_id,
    weather,
    season;
```

---

## 五、APP层 (Application) - 应用层

### 5.1 APP特征工程表（给RL模型用）

```sql
-- ============================================================================
-- APP 层：需求预测特征表
-- 特点：为RL模型提供直接可用的特征
-- ============================================================================

CREATE TABLE IF NOT EXISTS app_demand_features (
    feature_datetime TIMESTAMP COMMENT '特征时间',
    region_id INT COMMENT '地区ID',
    city_id INT COMMENT '城市ID',
    
    -- 当前小时的需求信息
    current_demand INT COMMENT '当前小时订单数',
    current_users INT COMMENT '当前小时用户数',
    
    -- 时序特征 (过去的需求模式)
    prev_hour_demand INT COMMENT '前1小时订单数',
    prev_2hours_demand INT COMMENT '前2小时订单数',
    prev_day_same_hour INT COMMENT '昨天同小时订单数',
    avg_7day_demand DOUBLE COMMENT '7天平均需求',
    avg_30day_demand DOUBLE COMMENT '30天平均需求',
    
    -- 时间特征
    hour_of_day INT COMMENT '小时(0-23)',
    day_of_week INT COMMENT '周几(1-7)',
    is_weekend INT COMMENT '是否周末',
    season STRING COMMENT '季节',
    
    -- 环境特征
    weather STRING COMMENT '天气',
    temperature INT COMMENT '温度',
    
    -- 聚集特征
    demand_trend DOUBLE COMMENT '需求趋势 (prev_hour / prev_2hours)',
    demand_volatility DOUBLE COMMENT '需求波动性',
    
    -- 目标变量（用于训练和评估）
    next_hour_demand INT COMMENT '下一小时需求（目标值）'
)
COMMENT 'APP层 - RL模型特征表'
PARTITIONED BY (dt STRING)
STORED AS PARQUET;
```

### 5.2 APP调度结果表

```sql
CREATE TABLE IF NOT EXISTS app_scheduling_results (
    eval_date DATE COMMENT '评估日期',
    strategy_name STRING COMMENT '策略名称 (PPO/Baseline)',
    scenario STRING COMMENT '场景',
    episode INT COMMENT '轮次',
    
    -- 服务指标
    service_rate DOUBLE COMMENT '服务率',
    total_demand INT COMMENT '总需求数',
    satisfied_demand INT COMMENT '满足需求数',
    unmet_demand INT COMMENT '未满足需求数',
    
    -- 经济指标
    total_revenue DOUBLE COMMENT '总收入',
    total_cost DOUBLE COMMENT '总调度成本',
    net_profit DOUBLE COMMENT '净利润',
    roi DOUBLE COMMENT 'ROI比率',
    
    -- 调度指标
    num_rebalance_actions INT COMMENT '调度行动数',
    avg_rebalance_cost DOUBLE COMMENT '平均调度成本',
    
    -- 库存指标
    avg_inventory DOUBLE COMMENT '平均库存',
    inventory_std DOUBLE COMMENT '库存标准差',
    max_inventory_shortage INT COMMENT '最大库存不足',
    
    eval_timestamp TIMESTAMP COMMENT '评估时间'
)
COMMENT 'APP层 - 调度策略结果'
PARTITIONED BY (dt STRING)
STORED AS PARQUET;
```

### 5.3 生成APP特征的SQL

```sql
-- ============================================================================
-- 从DWS生成APP (特征工程)
-- ============================================================================

INSERT OVERWRITE TABLE app_demand_features PARTITION(dt='2025-11-22')
SELECT 
    stat_datetime AS feature_datetime,
    region_id,
    city_id,
    order_count AS current_demand,
    unique_users AS current_users,
    
    -- 时序特征
    LAG(order_count, 1) OVER (
        PARTITION BY region_id 
        ORDER BY stat_datetime
    ) AS prev_hour_demand,
    
    LAG(order_count, 2) OVER (
        PARTITION BY region_id 
        ORDER BY stat_datetime
    ) AS prev_2hours_demand,
    
    LAG(order_count, 24) OVER (
        PARTITION BY region_id 
        ORDER BY stat_datetime
    ) AS prev_day_same_hour,
    
    AVG(order_count) OVER (
        PARTITION BY region_id 
        ORDER BY stat_datetime 
        ROWS BETWEEN 168 PRECEDING AND 1 PRECEDING
    ) AS avg_7day_demand,
    
    AVG(order_count) OVER (
        PARTITION BY region_id 
        ORDER BY stat_datetime 
        ROWS BETWEEN 720 PRECEDING AND 1 PRECEDING
    ) AS avg_30day_demand,
    
    -- 时间特征
    stat_hour AS hour_of_day,
    day_of_week,
    is_weekend,
    season,
    
    -- 环境特征
    weather,
    NULL AS temperature,  -- 从weather表join获取
    
    -- 聚集特征
    CASE 
        WHEN LAG(order_count, 2) OVER (PARTITION BY region_id ORDER BY stat_datetime) > 0
        THEN LAG(order_count, 1) OVER (PARTITION BY region_id ORDER BY stat_datetime) / 
             LAG(order_count, 2) OVER (PARTITION BY region_id ORDER BY stat_datetime)
        ELSE 0 
    END AS demand_trend,
    
    STDDEV(order_count) OVER (
        PARTITION BY region_id 
        ORDER BY stat_datetime 
        ROWS BETWEEN 24 PRECEDING AND CURRENT ROW
    ) AS demand_volatility,
    
    -- 目标变量
    LEAD(order_count, 1) OVER (
        PARTITION BY region_id 
        ORDER BY stat_datetime
    ) AS next_hour_demand

FROM dws_demand_hourly
WHERE dt = '2025-11-22';
```

---

## 六、完整的数据流程示例

### 示例：2025-11-22的数据处理

```sql
-- 步骤1: 查看ODS原始数据
SELECT COUNT(*) FROM ods_orders 
WHERE DATE(order_time) = '2025-11-22'
-- 结果: 100,000 行 (原始数据)

-- 步骤2: DWD清洗后
SELECT COUNT(*) FROM dwd_order_detail 
WHERE dt = '2025-11-22'
-- 结果: 98,500 行 (去重后，减少1.5%)

-- 步骤3: DWS汇总后
SELECT COUNT(*) FROM dws_demand_hourly 
WHERE dt = '2025-11-22'
-- 结果: 168 行 (6区域 × 24小时 + 2行多于1区域)

-- 步骤4: APP特征完整
SELECT COUNT(*) FROM app_demand_features 
WHERE dt = '2025-11-22' AND current_demand IS NOT NULL
-- 结果: 140 行 (首24小时无过去数据，所以只有140行有完整特征)

-- 步骤5: RL模型训练用APP表的特征
SELECT 
    region_id,
    hour_of_day,
    current_demand,
    prev_day_same_hour,
    avg_7day_demand,
    demand_trend,
    season,
    weather,
    next_hour_demand
FROM app_demand_features
WHERE dt = '2025-11-22' AND current_demand IS NOT NULL
LIMIT 10;
```

---

## 七、数据量级展示

### 数据量的变化

```
数据处理阶段          行数         特征       存储(Parquet)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ODS (原始)          100,000     15列       ~120MB
DWD (清洗去重)       98,500     16列       ~118MB
DWS (小时汇总)       ~170       13列       ~0.2MB
APP (特征工程)       ~140       20列       ~0.3MB
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

扩展性演示：
数据规模  ×1     ×10      ×100      ×1000
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
行数     100K   1M      10M       100M
处理方式  Pandas →Pandas  Spark     Spark
耗时     1s     10s     2min      20min
存储     120MB  1.2GB   12GB      120GB

关键结论：
✅ 10倍数据：从Pandas→Spark (超线性性能)
✅ 100倍数据：Spark的线性扩展（2min）
✅ 1000倍数据：分布式集群支持（无限扩展）
```

---

## 八、优点总结

### 分层架构的好处

| 优点 | 说明 | 例子 |
|------|------|------|
| **易维护** | 每层职责清晰 | DWD层数据不对，只需修复该层 |
| **易扩展** | 新需求复用前面的层 | 新Dashboard需求直接用DWS |
| **易复用** | 多个应用共享中间层 | DWS可同时支撑RL、Dashboard、报表 |
| **易优化** | 每层可单独优化 | DWS可单独建索引提速 |
| **可追溯** | 数据血缘清晰 | 问题数据可向上追溯 |
| **质量保证** | 每层进行检查 | DWD层验证数据质量 |

---

## 九、如何使用这个设计

### 在项目中应用

```python
# 假如你用Spark处理

from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("BikeSharing").getOrCreate()

# 1. 读取ODS
df_ods = spark.read.parquet('s3://data/ods_orders/')

# 2. 创建DWD (清洗去重)
df_dwd = df_ods.filter(
    (col('order_id').isNotNull()) &
    (col('duration_minutes') > 0) &
    (col('distance_km') > 0)
).dropDuplicates(['order_id'])

# 3. 创建DWS (小时汇总)
df_dws = df_dwd.groupBy(
    'order_date', 'order_hour', 'region_id', 'weather'
).agg({
    'order_id': 'count',
    'duration_minutes': ['avg', 'stddev'],
    'distance_km': 'avg'
})

# 4. 创建APP (特征工程)
# 使用窗口函数添加历史特征
window_spec = Window.partitionBy('region_id').orderBy('order_datetime')
df_app = df_dws.withColumn(
    'prev_hour_demand',
    lag('order_count').over(window_spec)
)

# 5. 用于RL训练
features = df_app.select([
    'current_demand', 'prev_hour_demand', 'prev_day_demand',
    'hour_of_day', 'day_of_week', 'weather', 'season'
])

# 训练RL模型...
```

---

## 十、总结

这个分层架构展示了：

✅ **大数据工程的规范做法**
- 数据不是一次性处理，而是分层管理
- 每层有清晰的职责和输入输出

✅ **可扩展的设计**
- 从10万行→100万行→10亿行，架构不变
- 只需更换存储/计算引擎

✅ **数据质量保证**
- DWD层进行数据清洗
- APP层进行特征验证

✅ **与RL模型的集成**
- APP层直接支撑模型训练
- 特征工程从数据角度优化模型输入

---

**这就是"大数据技术与应用"的完整体现！** ✨
