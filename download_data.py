import os
import time
from datasets import load_dataset
from huggingface_hub import snapshot_download
import hf_transfer
from requests import ConnectionError
import aria2p
import subprocess
import socket
from tcia_utils import nbia
import kaggle
import os

from physionet.api import PhysioNetClient

from kaggle.api.kaggle_api_extended import KaggleApi




def download_physionet_data(project_slug, destination):
    # 1. 配置你的 PhysioNet 账号（确保你已经在网页端签了 VinDr-Mammo 的协议）
    username = 'heubm'
    password = '12o34o56o'

    client = PhysioNetClient(username=username, password=password)

    # 2. 指定数据集信息
    # project_slug = 'vindr-mammo'
    version = '1.0.0'
    # destination = './vindr_mammo_data'

    if not os.path.exists(destination):
        os.makedirs(destination)

    # 3. 下载整个项目（注意：190GB，建议先下个 CSV 试试）
    # 如果你想先测试，可以只下载 metadata.csv
    print("正在尝试下载 metadata.csv...")
    client.projects.download_file(project_slug, version, 'metadata.csv', destination)

    # 如果想下整个数据集（请确保网速和空间）：
    # client.projects.download_all(project_slug, version, destination)


# def download_kaggle_data(dataset_slug, download_path):
#     # """
#     # dataset_slug: 数据集的后缀，例如 'awsaf49/cbis-ddsm-breast-cancer-image-dataset'
#     # download_path: 你想存放数据的本地路径
#     # """
#     # if not os.path.exists(download_path):
#     #     os.makedirs(download_path)
#     #     print(f"创建目录: {download_path}")

#     # # 设置环境变量（或者你也可以手动移动 kaggle.json）
#     # # os.environ['KAGGLE_CONFIG_DIR'] = "你的 kaggle.json 所在目录"
#     # print(f"正在开始下载数据集: {dataset_slug} ...")
    
#     # # 下载并自动解压
#     # # kaggle.api.dataset_download_files(dataset_slug, path=download_path, unzip=True)
#     # # 修改这一行
#     # kaggle.api.dataset_download_files(dataset_slug, path=download_path, unzip=True, quiet=False)
    
#     # print(f"下载并解压完成！存放在: {download_path}")
#     api = KaggleApi()
#     api.authenticate()

#     api.dataset_download_files(
#         dataset_slug,
#         path=download_path,
#         unzip=True
#     )


def download_tcia_data(data_name, dst_folder):
    print(nbia.getToken(user= "heubme", pw= "12o34o56o"))
    # data = nbia.getSeries(collection=data_name, modality='MG')
    # data = nbia.getSeries(modality="CT", api_url="nlst")
    # data = nbia.downloadSeries("manifest-NLST_allCT.tcia", input_type="manifest")
    # # 1. 获取该项目下所有的 MG 序列信息
    data = nbia.getSeries(collection=data_name, modality='MG')
    
    # 2. 精准过滤：只保留描述中包含 "full mammogram" 的序列
    # data = [s for s in data if "full mammogram" in s['SeriesDescription'].lower()]
    # data = [s for s in data if "ROI mask" in s['SeriesDescription'].lower()]
    print(len(data))
    idx = 0
    while idx < len(data):
        try:
            nbia.downloadSeries(data, number=idx)
            idx = idx + 1
        except:
            # time.sleep(1)
            idx = idx + 1
    # os.rename('tciaDownload', dst_folder)


if __name__ == '__main__':
    data_name = 'CDD-CESM'
    dst_folder = data_name
    download_tcia_data(data_name, dst_folder)

    # 2. 指定数据集信息
    # project_slug = 'vindr-mammo'

    # destination = './vindr_mammo_data'
    # download_physionet_data(project_slug, destination)