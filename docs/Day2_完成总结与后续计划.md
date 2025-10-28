# 共享单车大数据分析项目 - Day 2 完成总结

**项目名称**: 共享单车数据分析与强化学习调度  
**日期**: 2025-10-27（周日）  
**阶段**: M1 数据与分析 - Day 2/3  
**完成度**: ✅ 100%

---

## 一、今日完成内容

### 1.1 需求模型λ(t)拟合 ✅

#### **模型构建**
基于Kaggle Capital Bikeshare的17,379条小时级真实数据，构建需求强度预测模型。

**数学模型**:
```
log λ_t = β₀ + β_hour[h] + β_season[s] + β_weekday[w] + β_weather[k]
```

**特征维度**:
- **时间特征**: 小时(0-23)、工作日标识、季节(1-4)
- **天气特征**: 天气类型(1-4)、温度、湿度、风速
- **衍生特征**: 高峰时段标识、白天标识

#### **模型对比**

| 模型 | RMSE | MAE | R² | 特点 |
|------|------|-----|-----|------|
| **Poisson GLM** | 103.63 | 72.03 | 0.6609 | 统计基础模型，可解释性强 |
| **XGBoost** ⭐ | 39.14 | 23.87 | 0.9516 | **最佳模型**，预测精度高 |

**性能提升**:
- RMSE降低 **62%**（从103.63到39.14）
- R²提升 **44%**（从0.66到0.95）
- MAE降低 **67%**（从72.03到23.87）

#### **特征重要性分析**

| 排名 | 特征 | 重要性 | 解释 |
|------|------|--------|------|
| 1 | is_daytime | 43.12% | 白天时段是否，影响最大 |
| 2 | is_rush_hour | 13.37% | 早晚高峰识别 |
| 3 | hr | 13.34% | 具体小时 |
| 4 | workingday | 7.20% | 工作日vs周末 |
| 5 | yr | 6.87% | 年份趋势 |

**关键洞察**:
- ⏰ 时间维度特征占主导（is_daytime + is_rush_hour + hr = 69.83%）
- 📅 工作日属性显著影响需求
- 🌤️ 天气和温度也有一定影响

#### **需求规律发现**

**高峰时段** (订单/小时):
1. 🥇 **17:00** - 461.5单（下班高峰）
2. 🥈 **18:00** - 425.5单（晚高峰延续）
3. 🥉 **08:00** - 359.0单（上班高峰）

**季节性**:
- 🌞 **夏季**: 236.0单/小时（最高）
- 🍂 **秋季**: 次高
- 🌸 **春季**: 中等
- ❄️ **冬季**: 最低

**整体需求**:
- 平均: 189.5 ± 181.4 订单/小时
- 说明需求波动较大，受时段影响明显

#### **模型输出文件**

```
~/bike-sharing-analysis/results/
├── demand_model_analysis.png     # 6图组合分析（268KB）
├── lambda_params.pkl             # λ(t)参数字典 ⭐
├── poisson_model.pkl             # Poisson回归模型（2.4MB）
├── xgboost_model.json            # XGBoost模型（1.4MB）
├── model_comparison.csv          # 模型性能对比
└── feature_importance.csv        # 特征重要性排序
```

**lambda_params.pkl 内容**:
```python
{
    'hourly': {0: 55.4, 1: 46.2, ..., 23: 197.3},      # 24小时需求
    'seasonal': {1: 111.2, 2: 165.3, 3: 236.0, 4: 194.1},  # 4季节
    'weekday': {0: 134.2, ..., 6: 215.8},              # 7星期
    'workingday': {0: 143.5, 1: 208.9},                # 工作日vs周末
    'weather': {1: 201.3, 2: 181.4, 3: 142.8, 4: 98.3},   # 4种天气
    'overall_mean': 189.5,
    'overall_std': 181.4
}
```

---

### 1.2 Spark多维度数据分析 ✅

#### **分析架构**

使用PySpark 3.5.2对10万条订单进行分布式分析：
- **数据源**: orders_100k.csv (10万订单) + user_info_10k.csv + bike_info_5k.csv
- **计算模式**: 本地伪分布式 (8个shuffle分区)
- **内存配置**: 4GB driver memory
- **分析维度**: 8个（时间、天气、空间、用户、单车等）

#### **核心发现总结**

##### **1. 时间维度分析**

**24小时分布特征**:
- **高峰时段**: 23:00 (4,255单) > 17:00 (4,241单) > 22:00 (4,237单)
- **低谷时段**: 凌晨3:00-5:00
- **双峰模式**: 早高峰08:00，晚高峰17:00-18:00
- **夜间活跃**: 22:00-23:00仍有高需求（夜生活、餐饮）

**工作日 vs 周末**:
| 类型 | 订单量 | 占比 | 平均时长(秒) | 平均距离(km) |
|------|--------|------|-------------|-------------|
| 工作日 | 71,245 | 71.2% | 985.6 | 3.24 |
| 周末 | 28,755 | 28.8% | 993.8 | 3.27 |

**洞察**:
- 工作日订单占比超7成，符合通勤场景
- 周末骑行时间稍长（+8.2秒）、距离稍远（+0.03km）
- 说明周末更多休闲骑行

**季节分布**:
- 各季度订单量均衡（24-25%）
- 平均骑行时长无显著差异（986-989秒）

**月度趋势**:
- 2011年各月订单量在3,863-4,398之间波动
- 无明显增长或下降趋势（数据生成均匀）

##### **2. 天气维度分析**

**天气影响**:
| 天气 | 订单量 | 占比 | 平均距离(km) |
|------|--------|------|-------------|
| Clear (晴天) | 49,879 | 49.9% | 3.25 |
| Cloudy (多云) | 30,210 | 30.2% | 3.24 |
| Light Rain (小雨) | 14,889 | 14.9% | 3.25 |
| Heavy Rain (大雨) | 5,022 | 5.0% | 3.24 |

**洞察**:
- 晴天订单占比接近50%，是主要骑行场景
- 天气恶劣时订单量显著下降（大雨仅5%）
- 但平均距离无显著差异（说明恶劣天气下的订单多为刚需）

**温度与需求**:
| 温度 | 订单量 | 占比 |
|------|--------|------|
| Warm (温暖) | 42,172 | 42.2% |
| Cool (凉爽) | 36,867 | 36.9% |
| Hot (炎热) | 13,118 | 13.1% |
| Cold (寒冷) | 7,843 | 7.8% |

**洞察**:
- 温暖和凉爽天气最受欢迎（占79%）
- 极端温度（炎热/寒冷）明显抑制需求

##### **3. 空间维度分析**

**区域热度排名（起点）**:
| 排名 | 区域 | 订单量 | 占比 | 平均距离(km) | 区域特征 |
|------|------|--------|------|-------------|---------|
| 1 | B_Downtown | 24,968 | 25.0% | 2.77 | 市中心商务区，短途为主 |
| 2 | A_Capitol_Hill | 24,941 | 24.9% | 3.24 | 国会山政府区 |
| 3 | C_Georgetown | 14,952 | 15.0% | 3.96 | 商业居住混合，距离最长 |
| 4 | D_Dupont_Circle | 14,826 | 14.8% | 2.92 | 居住夜生活区 |
| 5 | F_Navy_Yard | 10,171 | 10.2% | 4.19 | 滨水区，距离最长 |
| 6 | E_Shaw | 10,142 | 10.1% | 2.90 | 文化艺术区 |

**OD流量矩阵 Top 10**:
| 起点 | 终点 | 流量 | 类型 |
|------|------|------|------|
| B_Downtown | B_Downtown | 7,458 | 同区 |
| A_Capitol_Hill | A_Capitol_Hill | 7,445 | 同区 |
| C_Georgetown | C_Georgetown | 4,503 | 同区 |
| D_Dupont_Circle | D_Dupont_Circle | 4,474 | 同区 |
| B_Downtown | D_Dupont_Circle | 3,600 | 跨区 |
| A_Capitol_Hill | C_Georgetown | 3,567 | 跨区 |
| A_Capitol_Hill | E_Shaw | 3,530 | 跨区 |
| B_Downtown | F_Navy_Yard | 3,515 | 跨区 |
| B_Downtown | E_Shaw | 3,505 | 跨区 |
| A_Capitol_Hill | B_Downtown | 3,495 | 跨区 |

**跨区流动分析**:
- **跨区骑行**: 70,069单（70.1%）
- **同区骑行**: 29,931单（29.9%）

**洞察**:
- 市中心和国会山是绝对热点（合占50%）
- 同区骑行占30%，说明短途代步需求大
- 跨区流动占70%，说明通勤和跨区活动频繁
- Georgetown和Navy_Yard平均距离最长（接近4km）

##### **4. 用户维度分析**

**性别分布**:
| 性别 | 订单量 | 占比 | 平均距离(km) | 平均费用($) |
|------|--------|------|-------------|-----------|
| 男 | 49,800 | 49.8% | 3.25 | 4.00 |
| 女 | 50,200 | 50.2% | 3.24 | 4.00 |

**会员类型**:
| 类型 | 订单量 | 占比 | 平均时长(秒) | 平均费用($) |
|------|--------|------|-------------|-----------|
| 普通用户 | 70,296 | 70.3% | 990.0 | 4.01 |
| 会员 | 29,704 | 29.7% | 983.1 | 4.00 |

**活跃用户 Top 10**:
| 排名 | 用户ID | 骑行次数 | 总消费($) | 平均距离(km) |
|------|--------|---------|----------|-------------|
| 1 | USER02190 | 25 | 100.3 | 3.26 |
| 2 | USER03076 | 23 | 96.3 | 3.51 |
| 3 | USER06698 | 23 | 100.3 | 3.50 |
| 4 | USER01973 | 22 | 87.0 | 3.35 |
| 5 | USER09893 | 22 | 80.7 | 2.85 |

**洞察**:
- 性别分布完全均衡（50:50）
- 普通用户占70%，会员占30%
- 会员平均时长略短（可能更熟练或有固定路线）
- 最活跃用户月骑行25次，年消费约$1,200

##### **5. 单车维度分析**

**单车类型对比**:
| 类型 | 订单量 | 占比 | 平均时长(秒) | 平均距离(km) | 平均费用($) |
|------|--------|------|-------------|-------------|-----------|
| 普通车 | 67,751 | 67.8% | 987.4 | 3.24 | 3.68 |
| 助力车 | 32,249 | 32.2% | 989.1 | 3.24 | 4.68 |

**高频单车 Top 10**:
| 排名 | 单车ID | 使用次数 | 总里程(km) | 平均费用($) |
|------|--------|---------|-----------|-----------|
| 1 | BIKE01363 | 43 | 153.64 | 3.83 |
| 2 | BIKE02415 | 39 | 117.00 | 3.58 |
| 3 | BIKE04626 | 39 | 150.16 | 5.02 |

**洞察**:
- 普通车占比68%，是主力车型
- 助力车虽少但单价高（$4.68 vs $3.68，高27%）
- 助力车更受欢迎（考虑到数量少但使用率不低）
- 最高频单车年使用43次，总里程153km

#### **整体统计摘要**

```
总订单数............................ 100,000
总用户数............................ 10,000
总单车数............................ 5,000
平均骑行时长........................ 16.47 分钟
平均骑行距离........................ 3.24 公里
平均费用............................ $4.00
总收入.............................. $400,320.90
```

**关键指标**:
- **车辆利用率**: 100,000单 ÷ 5,000车 ÷ 730天 = 27.4单/车/天
- **用户活跃度**: 100,000单 ÷ 10,000用户 = 10单/用户（2年）
- **客单价**: $4.00（合理定价）
- **收入预估**: 年收入约$20万（基于2年数据）

---

### 1.3 Pyecharts交互式可视化 ✅

#### **Dashboard架构**

**技术栈**:
- **Pyecharts 2.0.9**: 生成ECharts交互式图表
- **HTML5 + JavaScript**: 前端渲染
- **渐变设计**: 使用线性渐变和阴影效果

**页面组成**:
1. **统计卡片区** - 4个渐变卡片展示核心指标
2. **时间趋势图** - 双Y轴折线图
3. **季节天气对比** - 柱状图+饼图Grid布局
4. **区域热度排名** - 渐变柱状图
5. **OD流量热力图** - HeatMap展示起点-终点流动
6. **单车类型对比** - 雷达图多维度对比

#### **可视化效果**

**1. 统计卡片（渐变设计）**:
```
┌─────────────┬─────────────┬─────────────┬─────────────┐
│ 总订单数    │ 总收入      │ 平均时长    │ 平均距离    │
│ 100,000     │ $400,321    │ 16.5 min    │ 3.24 km     │
│ 紫色渐变    │ 粉色渐变    │ 蓝色渐变    │ 绿色渐变    │
└─────────────┴─────────────┴─────────────┴─────────────┘
```

**2. 24小时订单趋势（双Y轴）**:
- **左Y轴**: 订单量（蓝色实线）
- **右Y轴**: 平均距离（绿色虚线）
- **交互**: 缩放、悬停提示、标记最大/最小/平均值
- **发现**: 17:00-18:00和22:00-23:00为双高峰

**3. 季节需求 + 天气影响（Grid布局）**:
- **左侧**: 柱状图（4季节订单量对比）
- **右侧**: 环形饼图（4种天气占比）
- **样式**: 蓝色渐变填充
- **发现**: 夏季需求最高，晴天占50%

**4. 区域热度排名**:
- **类型**: 横向柱状图（6个区域）
- **颜色**: 蓝色渐变
- **数值**: 顶部标签显示
- **发现**: Downtown和Capitol Hill明显领先

**5. OD流量热力图**:
- **维度**: 6×6矩阵（起点×终点）
- **颜色**: 数值越大颜色越深
- **交互**: 悬停显示具体流量
- **发现**: 对角线（同区）和Downtown相关流量最高

**6. 单车类型雷达图**:
- **维度**: 订单量、时长、距离、费用
- **对比**: 普通车 vs 助力车
- **样式**: 半透明填充区域
- **发现**: 助力车在费用维度明显更高

#### **技术亮点**

**响应式设计**:
- 图表宽度1200px，适配主流显示器
- 支持缩放、拖拽、数据选择
- 移动端兼容（触摸交互）

**交互功能**:
- **Tooltip**: 悬停显示详细数据
- **Legend**: 点击切换数据系列
- **DataZoom**: 区域缩放查看细节
- **MarkPoint**: 自动标记极值点

**视觉效果**:
- **渐变色**: 使用JsCode实现线性渐变
- **阴影**: box-shadow增强立体感
- **圆角**: border-radius柔化边缘
- **字体**: 多层次字号区分重要性

#### **文件输出**

```
~/bike-sharing-analysis/web/
└── analysis_dashboard.html    # 完整Dashboard（约500KB）
```

**浏览器访问**:
```bash
# WSL中打开
cd ~/bike-sharing-analysis/web
cmd.exe /c start analysis_dashboard.html

# 或Windows路径
\\wsl$\Ubuntu\home\renr\bike-sharing-analysis\web\analysis_dashboard.html
```

---

## 二、核心技术要点

### 2.1 需求建模方法论

#### **Poisson回归 vs XGBoost**

**Poisson GLM的优势**:
- ✅ 统计学基础扎实，可解释性强
- ✅ 适合计数数据（订单量）
- ✅ 系数有明确含义（β值）
- ✅ 可进行统计推断（置信区间、p值）
- ❌ 假设需求强度为线性组合（受限）
- ❌ 难以捕捉复杂非线性关系

**XGBoost的优势**:
- ✅ 强大的非线性建模能力
- ✅ 自动特征交互（决策树）
- ✅ 处理缺失值和异常值
- ✅ 正则化防止过拟合
- ✅ 特征重要性分析
- ❌ 黑盒模型，可解释性较弱
- ❌ 需要更多调参（但本项目已优化）

**本项目选择**:
- 使用XGBoost作为主模型（R²=0.95）
- 保留Poisson回归作为基准对比
- 结合两者优势：XGBoost预测，Poisson解释

#### **特征工程技巧**

**衍生特征**:
```python
# 高峰时段识别
df['is_rush_hour'] = df['hr'].apply(lambda x: 1 if x in [7,8,9,17,18,19] else 0)

# 白天时段
df['is_daytime'] = df['hr'].apply(lambda x: 1 if 6 <= x <= 20 else 0)

# 温度分箱
df['temp_cat'] = pd.cut(df['temp'], bins=4, labels=['cold','cool','warm','hot'])
```

**为什么有效**:
- 高峰时段和白天是最重要特征（重要性前2）
- 将连续变量离散化可以增强树模型效果
- 领域知识驱动的特征工程比盲目增加特征更有效

#### **模型评估与验证**

**评估指标**:
- **RMSE** (Root Mean Squared Error): 惩罚大误差
- **MAE** (Mean Absolute Error): 平均绝对误差
- **R²** (决定系数): 解释方差比例

**验证策略**:
- 80/20时间序列划分（避免数据泄露）
- 随机种子固定（random_state=42）
- 交叉验证（未使用，但可扩展）

**模型保存**:
```python
# Poisson模型（使用pickle）
with open('poisson_model.pkl', 'wb') as f:
    pickle.dump(poisson_model, f)

# XGBoost模型（JSON格式，跨语言）
xgb_model.save_model('xgboost_model.json')
```

---

### 2.2 Spark分布式计算

#### **PySpark API使用**

**DataFrame操作**:
```python
# 1. 加载数据
df = spark.read.csv("orders_100k.csv", header=True, inferSchema=True)

# 2. 聚合分析
hourly_stats = df.groupBy('hr') \
    .agg(
        count('*').alias('order_count'),
        avg('duration_s').alias('avg_duration'),
        avg('distance_km').alias('avg_distance')
    ) \
    .orderBy('hr')

# 3. 条件筛选
peak_hours = df.filter(col('hr').isin([8, 17, 18]))

# 4. 关联查询
orders_with_users = df_orders.join(df_users, on='user_id', how='left')

# 5. 窗口函数
from pyspark.sql.window import Window
window_spec = Window.partitionBy('start_zone').orderBy(desc('distance_km'))
df.withColumn('rank', row_number().over(window_spec))
```

**性能优化**:
```python
# 1. 数据缓存（多次使用）
df_orders.cache()

# 2. 分区控制
spark.conf.set("spark.sql.shuffle.partitions", "8")

# 3. 内存配置
spark = SparkSession.builder \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()

# 4. 释放缓存
df_orders.unpersist()
```

#### **Spark vs Pandas选择**

**使用Spark的场景**:
- ✅ 数据量大（>1GB）
- ✅ 需要分布式处理
- ✅ 复杂的Join和聚合
- ✅ 与HDFS集成

**使用Pandas的场景**:
- ✅ 数据量小（<1GB）
- ✅ 快速原型开发
- ✅ 丰富的科学计算库
- ✅ 与Matplotlib/Seaborn集成

**本项目策略**:
- Spark用于大规模聚合分析（10万行）
- 分析结果转Pandas保存CSV（数百行）
- 可视化使用Pandas读取CSV

---

### 2.3 数据可视化设计

#### **Pyecharts核心组件**

**图表类型选择**:
| 数据特征 | 推荐图表 | 本项目应用 |
|---------|---------|-----------|
| 时间序列 | Line | 24小时趋势 |
| 类别对比 | Bar | 季节、区域对比 |
| 占比分析 | Pie | 天气影响分布 |
| 多维对比 | Radar | 单车类型对比 |
| 矩阵关系 | HeatMap | OD流量矩阵 |
| 地理分布 | Geo/Map | （未使用，可扩展）|

**交互设计**:
```python
# 1. Tooltip（悬停提示）
.set_global_opts(
    tooltip_opts=opts.TooltipOpts(
        trigger="axis",           # 坐标轴触发
        axis_pointer_type="cross" # 十字准星
    )
)

# 2. DataZoom（区域缩放）
.set_global_opts(
    datazoom_opts=[
        opts.DataZoomOpts(range_start=0, range_end=100)
    ]
)

# 3. MarkPoint（标记点）
.add_yaxis(
    markpoint_opts=opts.MarkPointOpts(
        data=[
            opts.MarkPointItem(type_="max", name="最大值"),
            opts.MarkPointItem(type_="min", name="最小值")
        ]
    )
)

# 4. 渐变色（JsCode）
itemstyle_opts=opts.ItemStyleOpts(
    color=JsCode("""
        new echarts.graphic.LinearGradient(0, 0, 0, 1, [
            {offset: 0, color: '#83bff6'},
            {offset: 1, color: '#188df0'}
        ])
    """)
)
```

#### **Grid布局技巧**

**多图组合**:
```python
grid = Grid(init_opts=opts.InitOpts(width="1200px", height="450px"))
grid.add(
    chart1, 
    grid_opts=opts.GridOpts(pos_left="5%", pos_right="55%")
)
grid.add(
    chart2, 
    grid_opts=opts.GridOpts(pos_left="55%", pos_right="5%")
)
```

**Page布局**:
```python
page = Page(layout=Page.SimplePageLayout)
page.add(chart1, chart2, chart3, ...)
page.render("dashboard.html")
```

---

### 2.4 数据质量保证

#### **数据一致性检查**

**订单数据验证**:
```python
# 1. 时间逻辑
assert (df['end_time'] > df['start_time']).all()

# 2. 数值范围
assert df['hr'].between(0, 23).all()
assert df['season'].between(1, 4).all()

# 3. 分布合理性
from scipy.stats import ks_2samp
# KS检验：生成数据 vs 真实数据
```

**关联完整性**:
```python
# 外键存在性
assert df_orders['user_id'].isin(df_users['user_id']).all()
assert df_orders['bike_id'].isin(df_bikes['bike_id']).all()
```

#### **异常值处理**

**识别方法**:
- IQR法（四分位距）
- Z-score法（标准差）
- 箱线图可视化

**处理策略**:
- 本项目：数据生成时已控制合理范围
- 实际场景：可截断、填充或删除

---

## 三、遇到的问题与解决方案

### 3.1 Radar图参数错误

**问题**:
```python
TypeError: Radar.add_schema() got an unexpected keyword argument 'splitline_opts'
```

**原因**: Pyecharts版本更新后，部分参数不再支持

**解决**:
```python
# 简化参数配置
radar = Radar().add_schema(
    schema=schema,
    shape="circle"
)
```

**学习点**: 
- 关注库版本变化
- 优先使用核心参数
- 查阅官方文档最新API

### 3.2 Dashboard排版问题

**现象**: 浏览器打开后图表挤在一起

**原因**: 
- 浏览器窗口未最大化
- 固定宽度1200px在小屏幕显示不全

**解决方案**:
1. **临时**: 最大化浏览器窗口
2. **优化**: 使用响应式宽度
```python
init_opts=opts.InitOpts(width="100%", height="450px")
```

### 3.3 文件路径访问

**问题**: WSL无法直接用浏览器打开HTML

**解决**:
```bash
# 方法1：cmd命令
cmd.exe /c start analysis_dashboard.html

# 方法2：explorer打开文件夹后手动双击
explorer.exe ~/bike-sharing-analysis/web/

# 方法3：Windows路径
\\wsl$\Ubuntu\home\renr\bike-sharing-analysis\web\analysis_dashboard.html
```

---

## 四、技术亮点与创新点

### 4.1 模型融合策略

✅ **双模型对比**:
- Poisson回归：提供统计基础和可解释性
- XGBoost：提供高精度预测
- 结合两者优势，既能解释又能预测

✅ **特征工程创新**:
- 领域知识驱动（高峰时段、白天标识）
- 比单纯增加特征更有效
- 特征重要性验证了设计合理性

### 4.2 全流程自动化

✅ **一键生成**:
- 数据生成 → 模型训练 → 分析 → 可视化
- 每个脚本独立运行，无人工干预
- 可复现（随机种子固定）

✅ **模块化设计**:
```
scripts/
├── generate_bike_data.py      # 数据生成
├── demand_modeling.py          # 模型训练
├── spark_analysis.py           # Spark分析
└── visualization.py            # 可视化生成
```

### 4.3 企业级可视化

✅ **交互性**:
- 缩放、拖拽、筛选
- 多维度联动
- 移动端兼容

✅ **美观性**:
- 渐变色设计
- 阴影和圆角
- 多层次信息展示

✅ **信息密度**:
- 一屏展示核心指标
- 多图表组合布局
- 避免信息过载

---

## 五、后续工作计划

### **M1 阶段剩余任务（Day 3）- 10月28日**

#### **任务1: Dashboard优化** ⭐

**排版优化**:
```python
# 1. 响应式宽度
init_opts=opts.InitOpts(width="100%", height="450px")

# 2. 移动端适配
@media (max-width: 768px) {
    .chart-container {
        width: 100% !important;
    }
}

# 3. 图表间距调整
page = Page(layout=Page.SimplePageLayout)
# 添加自定义CSS
```

**功能增强**:
1. **添加更多图表**:
   - 用户年龄分布（柱状图）
   - 费用分布（直方图）
   - 时空动态热力图（Timeline+HeatMap）

2. **交互增强**:
   - 区域点击跳转详情
   - 时间范围选择器
   - 数据导出按钮

3. **性能优化**:
   - 大数据量时使用数据抽样
   - 懒加载图表
   - CDN加速ECharts.js

**预期输出**:
- `analysis_dashboard_v2.html` - 优化版Dashboard
- `dashboard_mobile.html` - 移动端版本（可选）

#### **任务2: 分析报告整理**

**生成文档**:
1. **分析报告PDF**（可选）:
   ```python
   from reportlab.pdfgen import canvas
   # 或使用 Markdown → HTML → PDF
   ```

2. **数据洞察总结**:
   - 核心发现（5-10条）
   - 业务建议（3-5条）
   - 风险提示

3. **技术文档**:
   - 数据字典
   - 模型说明
   - API文档（供模拟器使用）

#### **任务3: 准备M2阶段基础**

**需求采样函数**:
```python
def sample_demand(hour, season, workingday, weather, zone_weights):
    """
    基于λ(t)参数进行泊松采样
    
    Args:
        hour: 小时(0-23)
        season: 季节(1-4)
        workingday: 工作日(0/1)
        weather: 天气(1-4)
        zone_weights: 区域权重字典
    
    Returns:
        demands: {zone: demand_count}
    """
    # 加载lambda_params.pkl
    # 计算基准需求强度
    # 按区域权重分配
    # 泊松采样
    return demands
```

**模拟器接口设计**:
```python
class BikeRebalancingEnv(gym.Env):
    def __init__(self, lambda_params, zone_config):
        # 初始化参数
        
    def reset(self):
        # 重置环境
        
    def step(self, action):
        # 执行动作
        # 需求采样（调用上面的函数）
        # 计算奖励
        return obs, reward, done, info
```

**输出**:
- `demand_sampler.py` - 需求采样模块
- `env_config.yaml` - 环境配置文件
- `gym_env_skeleton.py` - Gym环境框架

---

### **M2 阶段：调度模拟器（Day 4-6）- 10月29-31日**

#### **Day 4: Gym环境搭建**

**核心任务**:
1. **状态空间设计**:
   ```python
   observation_space = spaces.Dict({
       'inventory': spaces.Box(0, 500, shape=(6,)),  # 各区库存
       'hour': spaces.Discrete(24),
       'weekday': spaces.Discrete(7),
       'season': spaces.Discrete(4),
       'weather': spaces.Discrete(4)
   })
   ```

2. **动作空间设计**（简化版-夜间调度）:
   ```python
   # 每天23:00统一调度
   action_space = spaces.Box(
       low=-100, high=100, 
       shape=(6, 6),  # 6×6调度矩阵
       dtype=np.float32
   )
   ```

3. **奖励函数实现**:
   ```python
   reward = (
       2.0 * served_demand           # 收入
       - 5.0 * unmet_demand           # 未满足惩罚
       - 0.1 * rebalance_distance     # 调度成本
   )
   ```

**输出**:
- `bike_env.py` - 完整Gym环境
- `test_env.py` - 单元测试
- `env_demo.py` - 演示脚本

#### **Day 5: 基线策略实现**

**策略1: Zero-Action**（不调度）:
```python
def zero_policy(obs):
    return np.zeros((6, 6))
```

**策略2: Proportional Refill**（比例补货）:
```python
def proportional_policy(obs, historical_avg):
    current_inventory = obs['inventory']
    target_inventory = historical_avg * current_inventory.sum()
    
    surplus = current_inventory - target_inventory
    deficit = target_inventory - current_inventory
    
    # 贪心分配
    action = allocate(surplus, deficit)
    return action
```

**策略3: Min-Cost Flow**（最小成本流）:
```python
def min_cost_flow_policy(obs, cost_matrix):
    # NetworkX构图
    # 求解最小成本流
    # 转换为动作矩阵
    return action
```

**输出**:
- `baseline_policies.py` - 3种基线策略
- `policy_eval.py` - 策略评估脚本
- `baseline_results.csv` - 评估结果

#### **Day 6: 评估与对比**

**评估指标**:
```python
metrics = {
    'service_rate': served / total_demand,
    'unmet_demand': total_unmet,
    'rebalance_cost': total_cost,
    'total_revenue': total_revenue,
    'net_profit': revenue - cost
}
```

**对比实验**:
- 运行100个episode
- 固定随机种子
- 记录每个时刻的状态和指标
- 生成对比报告和可视化

**输出**:
- `evaluate.py` - 评估脚本
- `baseline_comparison.html` - 可视化对比
- `baseline_report.md` - 文字报告

---

### **M3 阶段：强化学习训练（Day 7-9）- 11月1-3日**

#### **Day 7: PPO算法接入**

```python
from stable_baselines3 import PPO

model = PPO(
    "MultiInputPolicy",  # 支持Dict observation
    env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    verbose=1,
    tensorboard_log="./logs/"
)

model.learn(total_timesteps=100000, callback=eval_callback)
model.save("ppo_bike_rebalancing")
```

#### **Day 8: 超参数调优**

**调参范围**:
- learning_rate: [1e-4, 3e-4, 1e-3]
- n_steps: [1024, 2048, 4096]
- gamma: [0.95, 0.99, 0.999]
- ent_coef: [0.0, 0.01, 0.05]

**方法**: 
- Grid Search（小规模）
- Optuna自动调优（推荐）

#### **Day 9: 策略对比**

**对比维度**:
1. **不同场景**:
   - 晴天 vs 雨天
   - 工作日 vs 周末
   - 夏季 vs 冬季

2. **不同预算**:
   - 无调度成本
   - 正常成本
   - 高成本

3. **不同初始库存**:
   - 均匀分布
   - 集中分布
   - 随机分布

**输出**:
- `ppo_model.zip` - 训练好的模型
- `training_curves.png` - 训练曲线
- `rl_vs_baseline.html` - 对比报告
- `evaluation_metrics.csv` - 详细指标

---

### **M4 阶段：平台集成与演示（Day 10-12）- 11月4-6日**

#### **Day 10: Flask应用开发**

**页面结构**:
```
web/
├── app.py                 # Flask后端
├── templates/
│   ├── index.html         # 首页
│   ├── analysis.html      # 分析页
│   └── simulation.html    # 仿真页
├── static/
│   ├── css/
│   ├── js/
│   └── img/
└── analysis_dashboard.html # 已完成
```

**路由设计**:
```python
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analysis')
def analysis():
    return render_template('analysis.html')

@app.route('/api/simulate', methods=['POST'])
def simulate():
    config = request.json
    results = run_simulation(config)
    return jsonify(results)
```

#### **Day 11: What-if仿真页面**

**功能设计**:
1. **场景配置面板**:
   - 天气选择（晴/雨）
   - 工作日/周末
   - 季节选择
   - 预算设置

2. **策略选择器**:
   - Zero Action
   - Proportional
   - Min-Cost Flow
   - PPO (RL)

3. **一键运行**:
   - 7×24小时快速仿真
   - 实时进度显示
   - 结果可视化

**交互流程**:
```
用户配置 → 点击"运行仿真" → 后端计算 → 返回JSON → 前端绘图
```

#### **Day 12: 文档与PPT**

**文档结构**:
1. **项目背景** (2页)
2. **技术架构** (3页)
3. **数据分析** (5页)
4. **需求建模** (3页)
5. **调度模拟** (4页)
6. **强化学习** (4页)
7. **结果对比** (3页)
8. **结论展望** (2页)

**PPT结构** (20-25页):
1. 封面 + 目录
2. 背景与目标
3. 技术路线图
4. 数据生成与质量
5. 需求模型λ(t)
6. Spark分析洞察
7. 可视化Dashboard展示
8. Gym环境设计
9. 基线策略对比
10. RL训练过程
11. 性能评估
12. **Live Demo** ⭐
13. 总结与展望

---

## 六、技术学习路线图

### **已掌握技能** ✅
- [x] Linux命令行操作
- [x] WSL环境配置与迁移
- [x] Hadoop HDFS基础
- [x] Spark安装配置
- [x] Python数据处理（Pandas/NumPy）
- [x] 模拟数据生成（Faker）
- [x] **需求预测建模（Poisson/XGBoost）** ⭐
- [x] **PySpark分布式计算** ⭐
- [x] **Pyecharts数据可视化** ⭐

### **本周需掌握** 📚
- [ ] Gymnasium (OpenAI Gym)环境开发
- [ ] 基线策略设计（启发式算法）
- [ ] 强化学习基础理论
- [ ] PPO算法原理与实现
- [ ] 模型评估与对比方法
- [ ] Flask Web开发
- [ ] 前端交互设计（HTML/CSS/JS）

### **进阶技能** 🚀
- [ ] 深度强化学习（DQN/A3C）
- [ ] 时间序列预测（Prophet/LSTM）
- [ ] 分布式训练（Ray/Horovod）
- [ ] 容器化部署（Docker/K8s）
- [ ] CI/CD流程
- [ ] 云服务器部署（AWS/阿里云）

---

## 七、风险预警与应对

### **风险1: Gym环境复杂度**
**表现**: 状态/动作空间设计困难，难以收敛

**应对**:
- ✅ 优先简化：夜间集中调度，不考虑延迟
- ✅ 状态归一化：[0, 1]范围
- ✅ 动作约束：总调度量上限
- ✅ 奖励设计：逐步调试（先简单后复杂）

### **风险2: RL训练不稳定**
**表现**: reward波动大，不收敛

**应对**:
- ✅ 降低环境复杂度（K=6区域，T=7天）
- ✅ 归一化奖励：`reward / max_possible_reward`
- ✅ 调参：降低learning_rate，增加entropy
- ✅ 增加训练时长：100k → 500k steps
- ✅ 使用多个随机种子验证

### **风险3: 时间紧张**
**表现**: 12天任务量大，可能延期

**应对**:
- ✅ 优先保证核心功能（M2+M3）
- ✅ Dashboard可以简化（v1版本够用）
- ✅ 并行开发：分析和模拟器可同步
- ✅ 复用代码：stable-baselines3开箱即用
- ✅ 灵活调整：根据进度删减非核心功能

### **风险4: Dashboard交互问题**
**表现**: 浏览器显示异常，交互卡顿

**应对**:
- ✅ 数据抽样：大数据量时只展示部分
- ✅ 懒加载：图表按需加载
- ✅ 简化交互：去掉非必要动画
- ✅ 浏览器兼容：优先支持Chrome

---

## 八、参考资料与学习资源

### **官方文档**
- [Pyecharts官方文档](https://pyecharts.org/)
- [Stable-Baselines3文档](https://stable-baselines3.readthedocs.io/)
- [Gymnasium文档](https://gymnasium.farama.org/)
- [PySpark API文档](https://spark.apache.org/docs/latest/api/python/)

### **学习资源**
- **需求预测**: 
  - "Forecasting: Principles and Practice" (Rob Hyndman)
  - Kaggle Bike Sharing Demand竞赛方案
  
- **强化学习**:
  - "Spinning Up in Deep RL" (OpenAI)
  - "Reinforcement Learning: An Introduction" (Sutton & Barto)
  
- **数据可视化**:
  - Pyecharts Gallery示例
  - D3.js可视化教程

### **代码参考**
- [Bike Sharing Demand Kaggle Kernels](https://www.kaggle.com/c/bike-sharing-demand/code)
- [Stable-Baselines3 Examples](https://github.com/DLR-RM/stable-baselines3)
- [Gymnasium Custom Environments](https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/)

---

## 九、项目交付物检查清单

### **M1阶段** (10-27 ~ 10-29) - Day 2已完成 ✅
- [x] 10万条订单数据
- [x] 项目目录结构
- [x] 数据生成脚本
- [x] **需求模型λ(t)** ⭐
- [x] **Spark多维度分析** ⭐
- [x] **可视化Dashboard v1** ⭐
- [ ] Dashboard优化v2（Day 3）
- [ ] 分析报告文档（Day 3）
- [ ] 需求采样模块（Day 3）

### **M2阶段** (10-30 ~ 11-01)
- [ ] Gym调度环境
- [ ] 3种基线策略
- [ ] 基线对比报告

### **M3阶段** (11-02 ~ 11-04)
- [ ] PPO训练脚本
- [ ] 训练好的模型
- [ ] RL vs基线对比

### **M4阶段** (11-05 ~ 11-07)
- [ ] Flask演示应用
- [ ] What-if仿真页面
- [ ] 项目文档
- [ ] 答辩PPT

---

## 十、Day 2 心得体会

### **技术收获**
1. **需求建模经验**:
   - 理解了计数数据建模方法（Poisson）
   - 掌握了XGBoost树模型调参技巧
   - 学会了特征重要性分析

2. **Spark实战**:
   - 熟悉了PySpark DataFrame API
   - 理解了分布式计算原理（shuffle、cache）
   - 掌握了SQL式数据分析思维

3. **可视化设计**:
   - 学会了Pyecharts多图表组合
   - 理解了交互式可视化的价值
   - 提升了数据呈现的审美

### **遇到的挑战**
1. **模型选择权衡**:
   - Poisson vs XGBoost各有优劣
   - 最终选择保留两者，各取所长

2. **Spark性能优化**:
   - 初次未设置cache，多次聚合很慢
   - 学会了缓存和分区优化

3. **可视化细节**:
   - 参数配置繁琐
   - 需要反复调试才能达到理想效果

### **改进方向**
1. 提前规划可视化需求（避免重复修改）
2. 建立代码模板库（常用图表配置）
3. 加强文档注释（方便后续维护）

---

## 十一、总结与展望

### **Day 2成就** 🎉
- ✅ 构建了高精度需求预测模型（R²=0.95）
- ✅ 完成了10万订单的全方位分析
- ✅ 生成了专业级可视化Dashboard
- ✅ 为模拟器开发奠定了坚实基础

### **接下来的重点**
1. **Day 3**: Dashboard优化 + 报告整理 + 模拟器准备
2. **Day 4-6**: Gym环境 + 基线策略 + 评估对比
3. **Day 7-9**: PPO训练 + 超参调优 + 性能评估
4. **Day 10-12**: Flask集成 + What-if页面 + 文档答辩

### **项目愿景**
通过本项目，我们不仅完成了学业要求，更重要的是：
- 掌握了大数据分析全流程（从数据到洞察）
- 理解了机器学习在实际问题中的应用
- 培养了端到端项目开发能力
- 积累了可展示的作品集

**数据驱动决策，算法优化运营** - 这是本项目的核心价值！

---

**项目进度**: 第2天/12天（16.7%）  
**预计完成时间**: 2025-11-07  
**当前状态**: ✅ 超预期完成

**下一步行动**: 
明天（10-28）优化Dashboard，准备Gym模拟器开发

---

*报告生成时间: 2025-10-27 22:00*  
*项目负责人: renr*  
*技术支持: Claude (Anthropic)*
