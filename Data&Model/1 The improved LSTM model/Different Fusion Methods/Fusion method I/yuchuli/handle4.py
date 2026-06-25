import os
import pandas as pd

def transpose_excel_files(folder_path, output_folder):
    """
    遍历文件夹下的表格文件，转置行列并保存新的文件。

    参数：
    - folder_path: 包含待处理表格文件的文件夹路径。
    - output_folder: 保存转置后文件的文件夹路径。
    """
    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)

    # 遍历文件夹中的所有表格文件
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.xlsx') or file_name.endswith('.xls'):  # 检查是否为 Excel 文件
            file_path = os.path.join(folder_path, file_name)
            print(f"正在处理文件：{file_name}")

            try:
                # 读取文件
                df = pd.read_excel(file_path)

                # 检查是否为空文件
                if df.empty:
                    print(f"警告: 文件 {file_name} 是空的，跳过处理。")
                    continue

                # 转置行列
                transposed_df = df.transpose()

                # 将原始列名设为转置后表格的第一行
                transposed_df.columns = transposed_df.iloc[0]
                transposed_df = transposed_df[1:]  # 移除第一行作为列名

                # 保存转置后的文件
                output_file_path = os.path.join(output_folder, f"transposed_{file_name}")
                transposed_df.to_excel(output_file_path, index=False)
                print(f"已保存转置后的文件：{output_file_path}")

            except Exception as e:
                print(f"错误: 无法处理文件 {file_name}。错误信息: {e}")
                continue

if __name__ == "__main__":
    # 输入文件夹路径
    folder_path = r"E:\加密货币\数据2024\数据更新\索引"  # 替换为您的输入文件夹路径

    # 输出文件夹路径
    output_folder = r"E:\加密货币\数据2024\数据更新\索引"  # 替换为您的输出文件夹路径

    # 调用函数进行处理
    transpose_excel_files(folder_path, output_folder)
