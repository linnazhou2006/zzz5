#!/usr/bin/env bash
# 移除 sudo 命令，直接执行

# 清理旧的编译产物
echo "Cleaning previous build artifacts..."
rm -f *.so      # 删除 .so 文件
rm -rf build/   # 删除 build 目录

# 检查 Python 3 和 ninja 是否可用
echo "Checking for python3 and ninja..."
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: python3 command not found. Please ensure python3 is installed and in PATH."
    exit 1
fi

if ! command -v ninja &> /dev/null; then
    echo "⚠️ Warning: ninja build system not found. Attempting to install ninja..."
    # 尝试在用户级别安装 ninja，如果用户权限允许
    # 注意：如果 pip install ninja 也失败，那可能是因为没有 pip 的写权限
    if python3 -m pip install ninja --user; then
        echo "✅ Ninja installed successfully."
    else
        echo "❌ Failed to install ninja. DCNv2 compilation might fail if it relies heavily on ninja."
        echo "   Falling back to distutils backend if ninja is truly unavailable."
    fi
fi

# 编译 DCNv2 扩展
echo "Compiling DCNv2 extension..."
# 使用python3直接执行setup.py
# build develop 参数会将其安装到 site-packages
python3 setup.py build develop

if [ $? -eq 0 ]; then
    echo "✅ DCNv2 compilation and installation successful."
else
    echo "❌ DCNv2 compilation and installation failed. Check previous errors."
    exit 1
fi

# 可选：编译 NMS 扩展 (如果 setup.py 配置了)
# 检查 setup.py 中的 nms 模块配置
if grep -q "nms.pyx" "setup.py"; then
    echo "Compiling NMS extension..."
    python3 setup.py build develop # 再次执行build develop，通常会包含所有扩展
    if [ $? -eq 0 ]; then
        echo "✅ NMS compilation and installation successful."
    else
        echo "❌ NMS compilation failed."
    fi
fi

echo "DCNv2 build process finished."