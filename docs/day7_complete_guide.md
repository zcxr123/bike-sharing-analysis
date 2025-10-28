# Day 7 完整工作包 - PPO训练与评估

**日期**: 2025-10-31  
**阶段**: M3 RL训练 - Day 1/3  
**预计时间**: 5-7小时

---

## 📦 包含文件清单

我已经为你准备了完整的Day 7工作包，包含以下文件：

### **核心脚本**（3个）
1. ✅ `day7_check_env.py` - 环境兼容性检查
2. ✅ `day7_train_ppo.py` - PPO训练主脚本（完整）
3. ✅ `day7_evaluate_ppo.py` - PPO评估与对比脚本

### **配置文件**（1个）
4. ✅ `ppo_training_config.yaml` - 训练超参数配置

### **辅助脚本**（1个）
5. ✅ `day7_run_all.sh` - 一键运行脚本

### **文档**（1个）
6. ✅ `Day7_Quick_Start.md` - 快速启动指南

---

## 🚀 三种启动方式

### **方式1: 一键运行（最简单）** ⭐推荐

```bash
cd ~/bike-sharing-analysis

# 复制所有脚本
cp /mnt/user-data/outputs/day7_*.py scripts/
cp /mnt/user-data/outputs/day7_run_all.sh scripts/
chmod +x scripts/day7_run_all.sh

# 一键运行（完整模式）
./scripts/day7_run_all.sh

# 或快速模式（省时间）
./scripts/day7_run_all.sh 50000 5 quick
```

### **方式2: 分步运行（标准）**

```bash
cd ~/bike-sharing-analysis

# 步骤1: 环境检查
python3 scripts/day7_check_env.py

# 步骤2: PPO训练
python3 scripts/day7_train_ppo.py --timesteps 100000 --quick-test

# 步骤3: 评估对比
python3 scripts/day7_evaluate_ppo.py \
    --model results/ppo_training/models/best_model/best_model.zip \
    --episodes 10
```

### **方式3: 自定义训练（高级）**

```bash
# 自定义超参数
python3 scripts/day7_train_ppo.py \
    --timesteps 200000 \
    --lr 0.0001 \
    --n-steps 4096 \
    --batch-size 128 \
    --quick-test

# 只评估特定场景
python3 scripts/day7_evaluate_ppo.py \
    --model results/ppo_training/models/best_model/best_model.zip \
    --episodes 5 \
    --scenarios default sunny_weekday
```

---

## ⏱️ 时间预算方案

### **极速方案（1.5小时）** - 时间非常紧张时

```bash
# 1. 环境检查（5分钟）
python3 scripts/day7_check_env.py

# 2. 快速训练（30分钟，1万步）
python3 scripts/day7_train_ppo.py --timesteps 10000 --quick-test

# 3. 快速评估（30分钟，3场景×3轮）
python3 scripts/day7_evaluate_ppo.py \
    --model results/ppo_training/models/best_model/best_model.zip \
    --episodes 3 \
    --scenarios default sunny_weekday rainy_weekend

# 4. 查看结果（5分钟）
cat results/ppo_evaluation/evaluation_summary_*.txt
```

### **标准方案（4小时）** - 平衡方案

```bash
# 1. 环境检查（5分钟）
python3 scripts/day7_check_env.py

# 2. 标准训练（1.5小时，10万步）
python3 scripts/day7_train_ppo.py --timesteps 100000 --quick-test

# 3. 完整评估（1.5小时，5场景×10轮）
python3 scripts/day7_evaluate_ppo.py \
    --model results/ppo_training/models/best_model/best_model.zip \
    --episodes 10

# 4. 分析结果（30分钟）
cat results/ppo_evaluation/evaluation_summary_*.txt
# 查看详细CSV
# 生成可视化
```

### **完整方案（6-8小时）** - 最佳性能

```bash
# 1. 环境检查（5分钟）
python3 scripts/day7_check_env.py

# 2. 深度训练（3-4小时，20-30万步）
python3 scripts/day7_train_ppo.py --timesteps 300000 --quick-test

# 3. 完整评估（2小时，5场景×20轮）
python3 scripts/day7_evaluate_ppo.py \
    --model results/ppo_training/models/best_model/best_model.zip \
    --episodes 20

# 4. 深度分析（1小时）
# 生成可视化图表
# 编写分析报告
```

---

## 📊 关键指标对比

训练完成后，重点关注这些指标：

### **Proportional-Optimized基线（Day 6结果）**
```
服务率: 99.99% ± 0.01%
净利润: $125,149 ± $9,677
调度成本: $1,147 ± $75
ROI: 351%
```

### **PPO目标**
```
最低目标: 服务率 > 95%，净利润 > $120,000
良好目标: 服务率 > 99%，净利润 > $125,000
优秀目标: 服务率 > 99.99%，净利润 > $130,000
```

### **判断标准**

**🎯 成功场景**:
- PPO服务率 ≥ 99%
- PPO净利润 ≥ $120,000
- 训练曲线收敛

**✨ 优秀场景**:
- PPO服务率 > 99.99%
- PPO净利润 > $125,149
- 超越基线策略

**🤔 需改进场景**:
- PPO服务率 < 95%
- 训练不收敛
- 需要超参数调优（Day 8）

---

## 🔧 常见问题与解决

### **问题1: 训练很慢**

**原因**: CPU训练，大规模timesteps

**解决方案**:
```bash
# 方案A: 减少timesteps
python3 scripts/day7_train_ppo.py --timesteps 50000

# 方案B: 减少batch size
python3 scripts/day7_train_ppo.py --batch-size 32 --n-steps 1024

# 方案C: 后台运行
nohup python3 scripts/day7_train_ppo.py --timesteps 200000 > train.log 2>&1 &
```

### **问题2: 训练不收敛**

**现象**: reward波动大，不增长

**解决方案**:
```bash
# 降低学习率
python3 scripts/day7_train_ppo.py --lr 0.0001 --timesteps 200000

# 或者增加训练步数
python3 scripts/day7_train_ppo.py --timesteps 300000
```

### **问题3: 性能不如基线**

**现象**: PPO服务率/利润低于Proportional-Optimized

**分析**:
1. 训练步数可能不够 → 增加到20-30万步
2. 奖励函数可能需要调整 → Day 8任务
3. 超参数需要优化 → Day 8任务
4. 该问题可能更适合启发式 → 也是有价值的发现

**这不是失败！** 即使PPO不如启发式，这也是重要的研究发现。

### **问题4: 内存不足**

**错误**: `MemoryError` 或 `Killed`

**解决方案**:
```bash
# 减少并行环境
# 减少batch size
python3 scripts/day7_train_ppo.py --batch-size 32

# 减少n_steps
python3 scripts/day7_train_ppo.py --n-steps 1024
```

---

## 📈 实时监控训练

### **使用TensorBoard**

```bash
# 安装（如果没有）
pip3 install tensorboard --break-system-packages

# 启动TensorBoard
cd ~/bike-sharing-analysis
tensorboard --logdir results/ppo_training/logs

# 在浏览器打开
# http://localhost:6006
```

**重点观察指标**:
- `rollout/ep_rew_mean`: Episode平均奖励（应该上升）
- `train/loss`: 训练损失
- `train/policy_gradient_loss`: 策略梯度损失
- `train/value_loss`: 价值函数损失

**健康训练的特征**:
- ✅ ep_rew_mean 逐渐上升
- ✅ loss 逐渐下降并稳定
- ✅ 没有突然的震荡

---

## 🎯 今日目标（Day 7）

### **必须完成** ✅
- [ ] 环境兼容性检查通过
- [ ] PPO模型训练完成（至少5万步）
- [ ] 评估得到性能数据
- [ ] 与基线对比分析

### **建议完成** 🎁
- [ ] 训练至少10万步
- [ ] 完整5场景评估
- [ ] TensorBoard曲线分析
- [ ] 生成评估总结

### **可选完成** ⭐
- [ ] 训练20万步以上
- [ ] 多次实验（不同随机种子）
- [ ] 超参数初步调优
- [ ] 编写Day 7完成总结

---

## 📝 下一步预告（Day 8）

根据Day 7的结果，Day 8可以：

### **如果PPO表现良好**（≥基线）
- 进一步优化超参数
- 尝试更复杂的网络结构
- 深度分析决策模式

### **如果PPO表现一般**（<基线）
- 诊断问题（奖励函数、状态空间等）
- 超参数网格搜索
- Reward Shaping
- 尝试其他算法（DQN、SAC）

### **如果训练不收敛**
- 检查环境实现
- 简化动作空间
- 调整学习率和训练规模

---

## 💡 重要提醒

1. **先小规模测试**: 用1万步验证流程（5-10分钟）
2. **保存中间结果**: 训练过程中按Ctrl+C也会保存模型
3. **实时监控**: 用TensorBoard观察训练状态
4. **记录实验**: 记下参数设置和结果
5. **接受所有结果**: PPO不一定比启发式好，都是有价值的发现

---

## 🚀 现在开始！

### **推荐执行流程**:

```bash
# 1. 复制脚本
cd ~/bike-sharing-analysis
cp /mnt/user-data/outputs/day7_*.py scripts/
cp /mnt/user-data/outputs/day7_*.sh scripts/
chmod +x scripts/day7_run_all.sh

# 2. 安装依赖
pip3 install stable-baselines3[extra] --break-system-packages

# 3. 运行！
# 选择你的方案：

# 方案A: 一键运行（完整）
./scripts/day7_run_all.sh

# 方案B: 一键运行（快速）
./scripts/day7_run_all.sh 50000 5 quick

# 方案C: 分步运行
python3 scripts/day7_check_env.py
python3 scripts/day7_train_ppo.py --timesteps 100000 --quick-test
python3 scripts/day7_evaluate_ppo.py --model results/ppo_training/models/best_model/best_model.zip
```

---

**时间紧张，让我们开始吧！Good luck! 🚀**

有问题随时问我！