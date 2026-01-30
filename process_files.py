from pathlib import Path
import os
import shutil
import pydicom
from tqdm import tqdm
import uuid
import cv2
import numpy as np
import matplotlib.pyplot as plt


def delete_large_dicoms(folder_path, size_limit_kb=10000):
    folder = Path(folder_path)
    # 计算字节限制 (1 KB = 1024 Bytes)
    size_limit_bytes = size_limit_kb * 1024
    
    # 查找所有 .dcm 和 .DCM 文件
    files = list(folder.rglob("*.dcm")) + list(folder.rglob("*.DCM"))
    
    deleted_count = 0
    total_size_freed = 0
    
    print(f"正在扫描文件夹: {folder.absolute()}")
    
    for f in files:
        # 获取文件大小
        file_size = f.stat().st_size
        
        if file_size > size_limit_bytes:
            try:
                # 打印被删除的文件名和大小
                size_mb = file_size / (1024 * 1024)
                print(f"删除文件: {f.name} ({size_mb:.2f} MB)")
                
                # 执行删除操作
                f.unlink()
                
                deleted_count += 1
                total_size_freed += file_size
            except Exception as e:
                print(f"无法删除 {f.name}: {e}")

    print("-" * 30)
    print(f"清理完成！")
    print(f"总计删除文件数: {deleted_count}")
    print(f"释放空间: {total_size_freed / (1024 * 1024):.2f} MB")


def flatten_and_cleanup(root_path):
    root = Path(root_path)
    
    # 1. 遍历所有子文件夹
    # sorted(reverse=True) 确保先处理最深层的子文件夹
    subdirs = sorted([d for d in root.rglob('*') if d.is_dir()], reverse=True)
    
    print(f"开始处理目录: {root.absolute()}")
    
    for folder in subdirs:
        # 查找该文件夹下直接存在的 .dcm 文件
        dcm_files = list(folder.glob("*.dcm")) + list(folder.glob("*.DCM"))
        
        if dcm_files:
            # 逻辑：如果有 DICOM 文件
            for dcm_path in dcm_files:
                # 定义新名字：父文件夹名 + 后缀
                new_name = f"{folder.name}.dcm"
                # 定义新路径：移动到根目录下（或者你指定的某个统一目录）
                target_path = root / new_name
                
                # 如果目标文件名已存在，避免覆盖（可选）
                if target_path.exists():
                    target_path = root / f"{folder.name}_{dcm_path.name}"

                try:
                    print(f"移动并重命名: {dcm_path.name} -> {new_name}")
                    shutil.move(str(dcm_path), str(target_path))
                except Exception as e:
                    print(f"处理文件 {dcm_path} 失败: {e}")
        
        # 2. 检查并删除空文件夹
        # 此时 DICOM 已移走，如果文件夹内没有其他重要文件，则删除
        try:
            # 再次检查文件夹是否为空（忽略隐藏文件如 .DS_Store 等可以加入额外逻辑）
            if not any(folder.iterdir()):
                folder.rmdir()
                print(f"已删除空文件夹: {folder.name}")
            else:
                # 如果文件夹非空（可能还有其他格式文件），你想强制删除吗？
                # 如果想强制删除，取消下面这行的注释：
                # shutil.rmtree(folder)
                # print(f"已强制删除非空文件夹: {folder.name}")
                pass
        except Exception as e:
            print(f"删除文件夹 {folder} 失败: {e}")


def keep_only_mask_dicoms(folder_path, dry_run=True):
    """
    folder_path: 目标文件夹
    dry_run: 为 True 时仅打印，False 时执行物理删除
    """
    folder = Path(folder_path)
    # 获取目录下所有 dcm 文件
    dcm_files = list(folder.rglob("*.dcm")) + list(folder.rglob("*.DCM"))
    
    keep_count = 0
    delete_count = 0
    
    print(f"正在清理文件夹: {folder.absolute()}")
    print("保留策略：Description 中必须包含 'mask' (不区分大小写)")

    for f in tqdm(dcm_files):
        try:
            # stop_before_pixels=True 极速读取元数据
            ds = pydicom.dcmread(f, stop_before_pixels=True)
            
            # 读取可能存在描述信息的多个 Tag
            series_desc = getattr(ds, 'SeriesDescription', '')
            study_desc = getattr(ds, 'StudyDescription', '')
            # 有时描述也会写在 ProtocolName 中
            protocol_name = getattr(ds, 'ProtocolName', '')
            
            combined_desc = (series_desc + study_desc + protocol_name).lower()
            
            # 判断逻辑：如果不包含 'mask'，则标记为删除
            if 'mask' not in combined_desc:
                if not dry_run:
                    f.unlink()
                delete_count += 1
            else:
                keep_count += 1
                
        except Exception as e:
            print(f"\n文件 {f.name} 读取失败，已跳过: {e}")

    print("-" * 30)
    if dry_run:
        print(f"[预演模式] 统计结果：")
        print(f"拟保留文件数 (含'mask'): {keep_count}")
        print(f"拟删除文件数 (不含'mask'): {delete_count}")
        print("确认无误后，请设置 dry_run=False 以执行删除。")
    else:
        print(f"清理完成！")
        print(f"保留文件数: {keep_count}")
        print(f"已物理删除文件数: {delete_count}")

def compare_folders(folder1_path, folder2_path):
    # 1. 获取两个文件夹下的所有文件名（不含路径）
    path1 = Path(folder1_path)
    path2 = Path(folder2_path)
    
    # 建议只提取文件名，忽略子目录本身
    files1 = {f.name for f in path1.glob('*') if f.is_file()}
    files2 = {f.name for f in path2.glob('*') if f.is_file()}
    
    # 2. 集合运算
    # 交集：两个文件夹都有的（重名）
    common = files1.intersection(files2)
    
    # 差集：文件夹1独有的
    only_in_1 = files1 - files2
    
    # 差集：文件夹2独有的
    only_in_2 = files2 - files1
    
    # 3. 打印结果
    print(f"文件夹 A: {path1.absolute()}")
    print(f"文件夹 B: {path2.absolute()}")
    print("-" * 30)
    print(f"文件夹 A 总计文件数: {len(files1)}")
    print(f"文件夹 B 总计文件数: {len(files2)}")
    print("-" * 30)
    print(f"【重名】的文件数量: {len(common)}")
    print(f"【不重名】的文件总数: {len(only_in_1) + len(only_in_2)}")
    print(f"   -> 仅在 A 中存在: {len(only_in_1)}")
    print(f"   -> 仅在 B 中存在: {len(only_in_2)}")

    # 如果你想查看具体哪些重名了，可以取消下面这行的注释
    # print(f"重名列表: {list(common)[:10]}...") 


def rename_dicoms_to_uuid(folder_path, dry_run=True):
    """
    folder_path: 目标文件夹
    dry_run: True 时只打印不改名，False 时正式重命名
    """
    folder = Path(folder_path)
    # 获取目录下所有 .dcm 文件
    dcm_files = list(folder.rglob("*.dcm")) + list(folder.rglob("*.DCM"))
    
    print(f"找到 {len(dcm_files)} 个 DICOM 文件。")
    if dry_run:
        print("--- 当前处于【预览模式】，不会修改文件 ---")

    count = 0
    for f in tqdm(dcm_files):
        # 1. 生成一个新的 UUID (使用 uuid4，即完全随机字符串)
        new_name = f"{uuid.uuid4()}.dcm"
        new_path = f.parent / new_name
        
        try:
            if not dry_run:
                # 2. 执行重命名（移动文件）
                f.rename(new_path)
            else:
                # 预览前 5 个重命名示例
                if count < 5:
                    print(f"示例: {f.name} -> {new_name}")
            
            count += 1
        except Exception as e:
            print(f"重命名 {f.name} 失败: {e}")

    if dry_run:
        print(f"\n预览完成。确认无误后请设置 dry_run=False。")
    else:
        print(f"\n重命名完成！共处理 {count} 个文件。")



# 使用示例
# gp, lp = process_dicom_pyramid("your_image.dcm")
if __name__ == '__main__':
    target_folder = 'CBIS-DDSM/full'
    # 先预览，确认没问题后再改为 False
    rename_dicoms_to_uuid(target_folder, dry_run=False)


