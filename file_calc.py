#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化版本：直接删除所有txt文件
"""

import os
import glob


def delete_all_txt_files():
    """直接删除image文件夹中所有txt文件"""
    base_dir = "/mnt/d/BaiduNetdiskDownload/s2/images"

    if not os.path.exists(base_dir):
        print(f"错误：目录 '{base_dir}' 不存在！")
        return

    total_deleted = 0

    # 遍历data1到data93
    for i in range(1, 94):
        folder_path = os.path.join(base_dir, f"data{i}")

        if os.path.exists(folder_path):
            txt_files = glob.glob(os.path.join(folder_path, "*.txt"))

            for txt_file in txt_files:
                try:
                    os.remove(txt_file)
                    print(f"删除: {txt_file}")
                    total_deleted += 1
                except Exception as e:
                    print(f"删除失败: {txt_file} - {e}")

    print(f"总共删除了{total_deleted}个txt文件")

if __name__ == "__main__":
    # 确认删除
    confirm = input("确定要删除所有txt文件吗？(y/N): ")
    if confirm.lower() in ['y', 'yes']:
        delete_all_txt_files()
    else:
        print("操作已取消")
