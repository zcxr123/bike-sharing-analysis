#!/bin/bash

# Day 10 一键部署脚本
# 自动设置Dashboard并启动

echo "======================================================================"
echo "Day 10 - Dashboard自动部署"
echo "======================================================================"
echo ""

# 检查Python
if ! command -v python3 &> /dev/null; then
    echo "❌ 错误: Python 3未安装"
    exit 1
fi

echo "✅ Python 3已安装"
echo ""

# 1. 安装依赖
echo "======================================================================"
echo "步骤 1/4: 安装依赖包"
echo "======================================================================"
echo ""

echo "📦 安装Streamlit和相关包..."
pip3 install streamlit plotly kaleido pandas numpy --break-system-packages

if [ $? -ne 0 ]; then
    echo "❌ 依赖安装失败"
    exit 1
fi

echo "✅ 依赖安装成功"
echo ""

# 2. 创建目录结构
echo "======================================================================"
echo "步骤 2/4: 创建Dashboard目录结构"
echo "======================================================================"
echo ""

mkdir -p dashboard/{data,assets/plots,pages,utils}

echo "✅ 目录创建完成"
echo ""

# 3. 复制文件
echo "======================================================================"
echo "步骤 3/4: 复制Dashboard文件"
echo "======================================================================"
echo ""

# 复制主应用
if [ -f "/mnt/user-data/outputs/dashboard_app.py" ]; then
    cp /mnt/user-data/outputs/dashboard_app.py dashboard/app.py
    echo "✅ 主应用: app.py"
fi

# 复制页面
if [ -f "/mnt/user-data/outputs/dashboard_page_comparison.py" ]; then
    cp /mnt/user-data/outputs/dashboard_page_comparison.py dashboard/pages/2_📈_策略对比.py
    echo "✅ 页面: 2_📈_策略对比.py"
fi

if [ -f "/mnt/user-data/outputs/dashboard_page_roi.py" ]; then
    cp /mnt/user-data/outputs/dashboard_page_roi.py dashboard/pages/4_💰_ROI计算器.py
    echo "✅ 页面: 4_💰_ROI计算器.py"
fi

# 复制数据准备脚本
if [ -f "/mnt/user-data/outputs/day10_prepare_data.py" ]; then
    cp /mnt/user-data/outputs/day10_prepare_data.py scripts/
    echo "✅ 脚本: day10_prepare_data.py"
fi

echo ""

# 4. 准备数据
echo "======================================================================"
echo "步骤 4/4: 准备Dashboard数据"
echo "======================================================================"
echo ""

python3 scripts/day10_prepare_data.py

if [ $? -ne 0 ]; then
    echo "❌ 数据准备失败"
    exit 1
fi

echo ""

# 完成
echo "======================================================================"
echo "✅ Dashboard部署完成！"
echo "======================================================================"
echo ""
echo "🚀 启动Dashboard:"
echo ""
echo "  cd dashboard"
echo "  streamlit run app.py"
echo ""
echo "📝 然后在浏览器中访问:"
echo "  http://localhost:8501"
echo ""
echo "💡 提示:"
echo "  - 使用 Ctrl+C 停止Dashboard"
echo "  - 修改代码后会自动重载"
echo "  - 数据文件位于 dashboard/data/"
echo ""