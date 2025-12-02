import os
import re

def add_underscores_after_numbers():
    # 获取当前脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 遍历当前目录下的所有文件
    for filename in os.listdir(current_dir):
        # 获取文件的完整路径
        old_path = os.path.join(current_dir, filename)
        
        # 跳过目录，只处理文件
        if os.path.isfile(old_path):
            # 分离文件名和扩展名
            name, ext = os.path.splitext(filename)
            
            # 使用正则表达式在数字后面添加下划线
            # (\d+) 匹配一个或多个数字，(?!\d) 确保后面不是数字
            new_name = re.sub(r'(\d+)(?!\d)', r'\1_', name)
            new_filename = new_name + ext
            new_path = os.path.join(current_dir, new_filename)
            
            # 如果新文件名与原文件名不同，则重命名
            if filename != new_filename:
                try:
                    # 检查新文件名是否已存在
                    if os.path.exists(new_path):
                        print(f"警告: {new_filename} 已存在，跳过重命名 {filename}")
                    else:
                        os.rename(old_path, new_path)
                        print(f"重命名: {filename} -> {new_filename}")
                except Exception as e:
                    print(f"重命名失败 {filename}: {e}")

def remove_underscores():
    # 获取当前脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 遍历当前目录下的所有文件
    for filename in os.listdir(current_dir):
        # 获取文件的完整路径
        file_path = os.path.join(current_dir, filename)
        
        # 检查是否是文件
        if os.path.isfile(file_path):
            # 如果文件名中包含下划线
            if "_" in filename:
                # 新文件名：去掉下划线
                new_filename = filename.replace("_", "")
                new_path = os.path.join(current_dir, new_filename)
                
                # 尝试重命名文件
                try:
                    os.rename(file_path, new_path)
                    print(f"重命名: {filename} -> {new_filename}")
                except Exception as e:
                    print(f"重命名失败 {filename}: {e}")

if __name__ == "__main__":
    print("开始处理文件名...")
    add_underscores_after_numbers()
    remove_underscores()
    print("完成!")