#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
需求模型拟合脚本 - λ(t) 建模
使用Poisson回归和XGBoost对共享单车需求进行建模
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
import os
import pickle

warnings.filterwarnings('ignore')

# 设置matplotlib中文显示
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 项目路径
PROJECT_ROOT = os.path.expanduser('~/bike-sharing-analysis')
DATA_RAW_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

print("="*70)
print("需求模型拟合 - λ(t) Demand Modeling")
print("="*70)

# ===== 1. 加载数据 =====
print("\n[1/6] 加载Kaggle数据...")

hour_csv = os.path.join(DATA_RAW_DIR, 'hour.csv')
df = pd.read_csv(hour_csv)

print(f"✓ 加载 {len(df):,} 条小时级数据")
print(f"✓ 时间范围: {df['dteday'].min()} ~ {df['dteday'].max()}")
print(f"✓ 特征列: {list(df.columns)}")

# ===== 2. 特征工程 =====
print("\n[2/6] 特征工程...")

# 创建衍生特征
df['is_rush_hour'] = df['hr'].apply(lambda x: 1 if x in [7, 8, 9, 17, 18, 19] else 0)
df['is_daytime'] = df['hr'].apply(lambda x: 1 if 6 <= x <= 20 else 0)
df['temp_cat'] = pd.cut(df['temp'], bins=4, labels=['cold', 'cool', 'warm', 'hot'])
df['hum_cat'] = pd.cut(df['hum'], bins=3, labels=['dry', 'moderate', 'humid'])

# 目标变量
y = df['cnt'].values

# 特征选择
feature_cols = ['hr', 'season', 'yr', 'mnth', 'holiday', 'weekday', 
                'workingday', 'weathersit', 'temp', 'atemp', 'hum', 'windspeed',
                'is_rush_hour', 'is_daytime']

X = df[feature_cols].values
feature_names = feature_cols

print(f"✓ 特征维度: {X.shape}")
print(f"✓ 目标变量范围: {y.min():.0f} ~ {y.max():.0f}")
print(f"✓ 目标变量均值: {y.mean():.1f} ± {y.std():.1f}")

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"✓ 训练集: {len(X_train):,} 样本")
print(f"✓ 测试集: {len(X_test):,} 样本")

# ==================== 3. Poisson回归 ====================
print("\n[3/6] Poisson回归建模...")

try:
    import statsmodels.api as sm
    
    # 添加截距项
    X_train_sm = sm.add_constant(X_train)
    X_test_sm = sm.add_constant(X_test)
    
    # Poisson GLM
    poisson_model = sm.GLM(
        y_train, 
        X_train_sm, 
        family=sm.families.Poisson()
    ).fit()
    
    # 预测
    y_pred_train_poisson = poisson_model.predict(X_train_sm)
    y_pred_test_poisson = poisson_model.predict(X_test_sm)
    
    # 评估
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train_poisson))
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test_poisson))
    mae_test = mean_absolute_error(y_test, y_pred_test_poisson)
    r2_test = r2_score(y_test, y_pred_test_poisson)
    
    print(f"✓ Poisson回归训练完成")
    print(f"  训练集 RMSE: {rmse_train:.2f}")
    print(f"  测试集 RMSE: {rmse_test:.2f}")
    print(f"  测试集 MAE:  {mae_test:.2f}")
    print(f"  测试集 R²:   {r2_test:.4f}")
    
    # 保存模型
    with open(os.path.join(RESULTS_DIR, 'poisson_model.pkl'), 'wb') as f:
        pickle.dump(poisson_model, f)
    
    poisson_available = True
    
except ImportError:
    print("⚠ statsmodels未安装，跳过Poisson回归")
    print("  可安装: pip3 install statsmodels --break-system-packages")
    poisson_available = False
    y_pred_test_poisson = None

# ==================== 4. XGBoost回归 ====================
print("\n[4/6] XGBoost回归建模...")

try:
    import xgboost as xgb
    
    # XGBoost参数
    params = {
        'objective': 'count:poisson',
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 200,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42
    }
    
    xgb_model = xgb.XGBRegressor(**params)
    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )
    
    # 预测
    y_pred_train_xgb = xgb_model.predict(X_train)
    y_pred_test_xgb = xgb_model.predict(X_test)
    
    # 评估
    rmse_train_xgb = np.sqrt(mean_squared_error(y_train, y_pred_train_xgb))
    rmse_test_xgb = np.sqrt(mean_squared_error(y_test, y_pred_test_xgb))
    mae_test_xgb = mean_absolute_error(y_test, y_pred_test_xgb)
    r2_test_xgb = r2_score(y_test, y_pred_test_xgb)
    
    print(f"✓ XGBoost训练完成")
    print(f"  训练集 RMSE: {rmse_train_xgb:.2f}")
    print(f"  测试集 RMSE: {rmse_test_xgb:.2f}")
    print(f"  测试集 MAE:  {mae_test_xgb:.2f}")
    print(f"  测试集 R²:   {r2_test_xgb:.4f}")
    
    # 特征重要性
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': xgb_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n  特征重要性 Top 5:")
    for idx, row in feature_importance.head(5).iterrows():
        print(f"    {row['feature']}: {row['importance']:.4f}")
    
    # 保存模型
    xgb_model.save_model(os.path.join(RESULTS_DIR, 'xgboost_model.json'))
    feature_importance.to_csv(
        os.path.join(RESULTS_DIR, 'feature_importance.csv'), 
        index=False
    )
    
    xgb_available = True
    
except ImportError:
    print("⚠ xgboost未安装，跳过XGBoost回归")
    print("  可安装: pip3 install xgboost --break-system-packages")
    xgb_available = False
    y_pred_test_xgb = None

# ====== 5. 模型对比与选择 =====
print("\n[5/6] 模型对比...")

comparison = []

if poisson_available:
    comparison.append({
        'Model': 'Poisson GLM',
        'RMSE': rmse_test,
        'MAE': mae_test,
        'R²': r2_test
    })

if xgb_available:
    comparison.append({
        'Model': 'XGBoost',
        'RMSE': rmse_test_xgb,
        'MAE': mae_test_xgb,
        'R²': r2_test_xgb
    })

if comparison:
    comparison_df = pd.DataFrame(comparison)
    print("\n模型性能对比:")
    print(comparison_df.to_string(index=False))
    
    # 选择最佳模型
    best_model_name = comparison_df.loc[comparison_df['RMSE'].idxmin(), 'Model']
    print(f"\n✓ 最佳模型: {best_model_name}")
    
    # 保存对比结果
    comparison_df.to_csv(
        os.path.join(RESULTS_DIR, 'model_comparison.csv'),
        index=False
    )

# ====== 6. 可视化 =====
print("\n[6/6] 生成可视化图表...")

# 创建图表
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Demand Model Analysis - λ(t)', fontsize=16, fontweight='bold')

# 6.1 实际值 vs 预测值 (XGBoost)
if xgb_available:
    ax = axes[0, 0]
    sample_idx = np.random.choice(len(y_test), 500, replace=False)
    ax.scatter(y_test[sample_idx], y_pred_test_xgb[sample_idx], 
               alpha=0.5, s=20, c='steelblue')
    ax.plot([y_test.min(), y_test.max()], 
            [y_test.min(), y_test.max()], 
            'r--', lw=2, label='Perfect Prediction')
    ax.set_xlabel('Actual Demand', fontsize=11)
    ax.set_ylabel('Predicted Demand', fontsize=11)
    ax.set_title('XGBoost: Actual vs Predicted', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

# 6.2 按小时的平均需求
ax = axes[0, 1]
hourly_demand = df.groupby('hr')['cnt'].mean()
ax.plot(hourly_demand.index, hourly_demand.values, 
        marker='o', linewidth=2, markersize=8, color='darkgreen')
ax.fill_between(hourly_demand.index, hourly_demand.values, alpha=0.3, color='lightgreen')
ax.set_xlabel('Hour of Day', fontsize=11)
ax.set_ylabel('Average Demand', fontsize=11)
ax.set_title('Hourly Demand Pattern', fontsize=12, fontweight='bold')
ax.set_xticks(range(0, 24, 2))
ax.grid(True, alpha=0.3)

# 6.3 按季节的需求分布
ax = axes[0, 2]
season_labels = ['Winter', 'Spring', 'Summer', 'Fall']
season_demand = df.groupby('season')['cnt'].mean()
colors = ['skyblue', 'lightgreen', 'gold', 'coral']
bars = ax.bar(season_labels, season_demand.values, color=colors, edgecolor='black', linewidth=1.5)
ax.set_ylabel('Average Demand', fontsize=11)
ax.set_title('Seasonal Demand', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
# 添加数值标签
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.0f}', ha='center', va='bottom', fontsize=10)

# 6.4 工作日 vs 周末
ax = axes[1, 0]
workday_demand = df.groupby('workingday')['cnt'].mean()
labels = ['Weekend', 'Weekday']
bars = ax.bar(labels, workday_demand.values, 
              color=['coral', 'steelblue'], edgecolor='black', linewidth=1.5)
ax.set_ylabel('Average Demand', fontsize=11)
ax.set_title('Weekday vs Weekend Demand', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.0f}', ha='center', va='bottom', fontsize=10)

# 6.5 天气影响
ax = axes[1, 1]
weather_labels = ['Clear', 'Cloudy', 'Light Rain', 'Heavy Rain']
weather_demand = df.groupby('weathersit')['cnt'].mean()
bars = ax.bar(weather_labels, weather_demand.values, 
              color=['gold', 'lightgray', 'lightblue', 'darkblue'],
              edgecolor='black', linewidth=1.5)
ax.set_ylabel('Average Demand', fontsize=11)
ax.set_title('Weather Impact on Demand', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
plt.setp(ax.xaxis.get_majorticklabels(), rotation=15, ha='right')
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.0f}', ha='center', va='bottom', fontsize=9)

# 6.6 特征重要性 (如果有XGBoost)
ax = axes[1, 2]
if xgb_available:
    top_features = feature_importance.head(10)
    ax.barh(range(len(top_features)), top_features['importance'].values, 
            color='steelblue', edgecolor='black', linewidth=1)
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['feature'].values, fontsize=10)
    ax.set_xlabel('Importance', fontsize=11)
    ax.set_title('Feature Importance (XGBoost)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    ax.invert_yaxis()
else:
    ax.text(0.5, 0.5, 'XGBoost not available', 
            ha='center', va='center', fontsize=12, transform=ax.transAxes)
    ax.axis('off')

plt.tight_layout()
plot_path = os.path.join(RESULTS_DIR, 'demand_model_analysis.png')
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
print(f"✓ 可视化图表已保存: {plot_path}")

# ===== 7. 导出λ(t)参数 =====
print("\n[7/7] 导出需求模型参数...")

# 计算各维度的平均需求强度
lambda_params = {
    'hourly': df.groupby('hr')['cnt'].mean().to_dict(),
    'seasonal': df.groupby('season')['cnt'].mean().to_dict(),
    'weekday': df.groupby('weekday')['cnt'].mean().to_dict(),
    'workingday': df.groupby('workingday')['cnt'].mean().to_dict(),
    'weather': df.groupby('weathersit')['cnt'].mean().to_dict(),
    'overall_mean': df['cnt'].mean(),
    'overall_std': df['cnt'].std()
}

# 保存参数
with open(os.path.join(RESULTS_DIR, 'lambda_params.pkl'), 'wb') as f:
    pickle.dump(lambda_params, f)

print(f"✓ λ(t)参数已保存: lambda_params.pkl")

# 打印摘要
print("\n" + "="*70)
print("需求模型参数摘要")
print("="*70)
print(f"整体平均需求: {lambda_params['overall_mean']:.1f} ± {lambda_params['overall_std']:.1f}")
print(f"\n高峰小时 Top 3:")
top_hours = sorted(lambda_params['hourly'].items(), key=lambda x: x[1], reverse=True)[:3]
for hr, demand in top_hours:
    print(f"  {hr:02d}:00 - {demand:.1f} 订单/小时")

print(f"\n最佳季节:")
best_season = max(lambda_params['seasonal'].items(), key=lambda x: x[1])
season_map = {1: 'Winter', 2: 'Spring', 3: 'Summer', 4: 'Fall'}
print(f"  {season_map[best_season[0]]}: {best_season[1]:.1f} 订单/小时")

print("\n" + "="*70)
print("✅ 需求模型拟合完成！")
print("="*70)
print("\n生成的文件:")
print(f"  1. {os.path.join(RESULTS_DIR, 'demand_model_analysis.png')}")
print(f"  2. {os.path.join(RESULTS_DIR, 'lambda_params.pkl')}")
print(f"  3. {os.path.join(RESULTS_DIR, 'model_comparison.csv')}")
if xgb_available:
    print(f"  4. {os.path.join(RESULTS_DIR, 'feature_importance.csv')}")
print("\n下一步: 使用lambda_params.pkl在模拟器中进行需求采样")
print("="*70)