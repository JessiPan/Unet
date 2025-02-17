import os
import shutil

# 指定源目录和目标目录
source_directory = "./VOCdevkit/VOC2007/SegmentationClass(original)"
target_directory = "VOCdevkit/VOC2007/SegmentationClass"

# 确保目标目录存在
if not os.path.exists(target_directory):
    os.makedirs(target_directory)

# 遍历源目录中的所有文件
for filename in os.listdir(source_directory):
    # 检查文件名是否包含 '_lab'
    if '_lab' in filename:
        # 构建新的文件名
        new_filename = filename.replace('_lab', '')
        # 构建完整的源文件路径和目标文件路径
        source_file = os.path.join(source_directory, filename)
        target_file = os.path.join(target_directory, new_filename)
        # 复制文件到目标目录并重命名
        shutil.copy2(source_file, target_file)
        print(f"文件 {filename} 已重命名为 {new_filename} 并保存到 {target_directory}")

print("所有文件处理完成")