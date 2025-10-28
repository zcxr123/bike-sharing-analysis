# Day 7 快速启动指南

**目标**: PPO算法接入与训练  
**时间**: 5-7小时  
**日期**: 2025-10-31

---

## 🚀 快速开始（3步）

### **步骤1: 安装依赖**

```bash
# 安装stable-baselines3
pip3 install stable-baselines3[extra] --break-system-packages

# 或者使用清华镜像加速
pip3 install stable-baselines3[extra] -i https://pypi.tuna.tsinghua.edu.cn/simple --break-system-packages
```

### **步骤2: 复制脚本到项目**

```bash
cd ~/bike-sharing-analysis

# 创建scripts目录（如果不存在）
mkdir -p scripts

# 复制Day 7脚本
cp /mnt/user-data/outputs/day7_check_env.py scripts/
cp /mnt/user-data/outputs/day7_train_ppo.py scripts/
cp /mnt/user-data/outputs/day7_evaluate_ppo.py scripts/
```

### **步骤3: 环境检查**

```bash
cd ~/bike-sharing-analysis
python3 scripts/day7_check_env.py
```

**预期输出**:
```
✅ BikeRebalancingEnv导入成功
✅ 配置加载成功
✅ 环境创建成功
✅ 环境通过SB3兼容性检查！
✅ 所有检查通过！
```

---

## 📋 详细任务流程

### **任务1: PPO训练（核心）** [2-3小时]

#### **1.1 快速训练（推荐）**

```bash
cd ~/bike-sharing-analysis

# 默认配置训练（10万步，约30-60分钟）
python3 scripts/day7_train_ppo.py --timesteps 100000 --quick-test
```

**训练参数说明**:
- `--timesteps 100000`: 总训练步数（可调整）
- `--lr 3e-4`: 学习率（默认）
- `--n-steps 2048`: 每次更新步数（默认）
- `--batch-size 64`: 批大小（默认）
- `--quick-test`: 训练后进行快速测试

#### **1.2 小规模测试（快速验证）**

如果时间紧张，先用小规模验证：

```bash
# 1万步快速测试（约3-5分钟）
python3 scripts/day7_train_ppo.py --timesteps 10000 --quick-test
```

#### **1.3 调整训练规模**

根据时间和硬件选择：

```bash
# 小规模（5分钟）- 快速验证
python3 scripts/day7_train_ppo.py --timesteps 10000

# 中等规模（30分钟）- 平衡性能
python3 scripts/day7_train_ppo.py --timesteps 50000

# 大规模（1-2小时）- 最佳性能
python3 scripts/day7_train_ppo.py --timesteps 200000
```

#### **1.4 查看训练日志**

训练过程中打开新终端：

```bash
# 安装tensorboard（如果没有）
pip3 install tensorboard --break-system-packages

# 查看训练曲线
cd ~/bike-sharing-analysis
tensorboard --logdir results/ppo_training/logs
```

然后在浏览器打开: http://localhost:6006

---

### **任务2: PPO评估与对比** [1小时]

训练完成后，评估PPO性能并与基线对比。

#### **2.1 找到训练好的模型**

```bash
# 查看训练的模型
ls -lh ~/bike-sharing-analysis/results/ppo_training/models/

# 输出示例：
# best_model/best_model.zip      ← 最佳模型
# ppo_final_20251031_*.zip       ← 最终模型
# checkpoints/ppo_bike_*.zip     ← 检查点
```

#### **2.2 完整评估（推荐）**

```bash
cd ~/bike-sharing-analysis

# 使用最佳模型进行完整评估
python3 scripts/day7_evaluate_ppo.py \
    --model results/ppo_training/models/best_model/best_model.zip \
    --episodes 10
```

**输出内容**:
- PPO在5个场景的表现
- Proportional-Optimized基线表现
- 详细对比分析
- 总结报告

#### **2.3 快速评估（省时）**

如果时间紧张：

```bash
# 只评估3个核心场景，每个场景5轮
python3 scripts/day7_evaluate_ppo.py \
    --model results/ppo_training/models/best_model/best_model.zip \
    --episodes 5 \
    --scenarios default sunny_weekday rainy_weekend
```

#### **2.4 只评估PPO**

跳过基线对比（省时间）：

```bash
python3 scripts/day7_evaluate_ppo.py \
    --model results/ppo_training/models/best_model/best_model.zip \
    --episodes 5 \
    --ppo-only
```

---

### **任务3: 结果分析** [30分钟]

#### **3.1 查看评估结果**

```bash
cd ~/bike-sharing-analysis/results/ppo_evaluation

# 查看生成的文件
ls -lh

# 文件说明：
# ppo_detail_*.csv              - PPO详细结果
# baseline_detail_*.csv         - 基线详细结果
# ppo_vs_baseline_*.csv         - 对比结果
# evaluation_summary_*.txt      - 总结报告
```

#### **3.2 查看总结报告**

```bash
# 查看总结
cat results/ppo_evaluation/evaluation_summary_*.txt
```

**关键指标**:
- ✅ **服务率**: PPO vs Proportional-Optimized (99.99%)
- ✅ **净利润**: PPO vs Proportional-Optimized ($125,149)
- ✅ **调度成本**: PPO vs Proportional-Optimized ($1,147)

#### **3.3 判断训练效果**

**情况1: PPO > Proportional** ✨
- 服务率 > 99.99% 或 净利润 > $125,149
- **结论**: RL成功！比启发式更好

**情况2: PPO ≈ Proportional** 🤝
- 性能接近（差异<1%）
- **结论**: RL达到启发式水平，都很优秀

**情况3: PPO < Proportional** 🤔
- 性能不及启发式
- **可能原因**:
  - 训练步数不够
  - 超参数需要调整
  - 奖励函数设计问题
  - 该问题更适合启发式

---

## ⚡ 时间紧张方案

### **最小可行方案（2小时）**

```bash
# 1. 环境检查（5分钟）
python3 scripts/day7_check_env.py

# 2. 小规模训练（30分钟）
python3 scripts/day7_train_ppo.py --timesteps 50000 --quick-test

# 3. 快速评估（30分钟）
python3 scripts/day7_evaluate_ppo.py \
    --model results/ppo_training/models/best_model/best_model.zip \
    --episodes 5 \
    --scenarios default sunny_weekday rainy_weekend

# 4. 查看结果（5分钟）
cat results/ppo_evaluation/evaluation_summary_*.txt
```

### **标准方案（4小时）**

```bash
# 1. 环境检查（5分钟）
python3 scripts/day7_check_env.py

# 2. 中等规模训练（1.5小时）
python3 scripts/day7_train_ppo.py --timesteps 100000 --quick-test

# 3. 完整评估（1.5小时）
python3 scripts/day7_evaluate_ppo.py \
    --model results/ppo_training/models/best_model/best_model.zip \
    --episodes 10

# 4. 分析结果（30分钟）
cat results/ppo_evaluation/evaluation_summary_*.txt
# 查看详细CSV文件
```

---

## 🔧 故障排查

### **问题1: 导入错误**

```
ImportError: No module named 'stable_baselines3'
```

**解决**:
```bash
pip3 install stable-baselines3[extra] --break-system-packages
```

### **问题2: 环境不兼容**

```
AssertionError: observation_space mismatch
```

**解决**:
- 检查环境是否正确初始化
- 运行 `python3 scripts/day7_check_env.py` 查看详细错误

### **问题3: 训练不收敛**

**现象**: reward一直不增长

**解决**:
1. 检查奖励函数设计
2. 降低学习率: `--lr 1e-4`
3. 增加训练步数: `--timesteps 200000`

### **问题4: GPU/内存不足**

**解决**:
```bash
# 减少batch size
python3 scripts/day7_train_ppo.py --batch-size 32

# 减少n_steps
python3 scripts/day7_train_ppo.py --n-steps 1024
```

---

## 📊 预期输出文件

训练和评估完成后，应该有以下文件：

```
bike-sharing-analysis/
├── results/
│   ├── ppo_training/
│   │   ├── models/
│   │   │   ├── best_model/
│   │   │   │   └── best_model.zip         ⭐ 最佳模型
│   │   │   ├── ppo_final_*.zip            ⭐ 最终模型
│   │   │   └── checkpoints/
│   │   │       └── ppo_bike_*.zip         ⭐ 训练检查点
│   │   └── logs/
│   │       ├── PPO_*/                     ⭐ TensorBoard日志
│   │       └── eval/                      ⭐ 评估日志
│   └── ppo_evaluation/
│       ├── ppo_detail_*.csv               ⭐ PPO详细结果
│       ├── baseline_detail_*.csv          ⭐ 基线详细结果
│       ├── ppo_vs_baseline_*.csv          ⭐ 对比结果
│       └── evaluation_summary_*.txt       ⭐ 总结报告
```

---

## 🎯 成功标准

Day 7任务成功的标志：

- ✅ PPO模型训练完成（无错误）
- ✅ 训练曲线收敛（reward增长趋于稳定）
- ✅ 评估完成，得到性能数据
- ✅ 与Proportional-Optimized基线对比
- ✅ 生成评估总结报告

---

## 📝 下一步（Day 8）

如果PPO性能不理想，Day 8可以：

1. **超参数调优**
   - 网格搜索
   - 学习率调整
   - 网络结构调整

2. **训练策略优化**
   - Reward Shaping
   - Curriculum Learning
   - 增加训练步数

3. **尝试其他算法**
   - DQN（离散动作）
   - SAC（连续动作）
   - A2C（更稳定）

---

## 💡 小贴士

1. **先小规模测试**: 用1万步先验证流程通畅
2. **利用TensorBoard**: 实时监控训练进度
3. **保存检查点**: 避免训练中断丢失进度
4. **多次实验**: 不同随机种子可能有不同效果
5. **记录实验**: 记下每次实验的参数和结果

---

**祝训练顺利！有问题随时问我 😊**