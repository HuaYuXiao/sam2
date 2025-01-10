import numpy as np
import torch
import cv2
import os
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

data_dir = r"infrared_data/"
data = []
for ff, name in enumerate(os.listdir(data_dir + "images/")):
    image_path = data_dir + "images/" + name
    annotation_path = data_dir + "annotations/" + name.replace("__img", "__label") 
    data.append({"image": image_path, "annotation": annotation_path})

# 输出结果，检查数据是否正确加载
# print(f"找到 {len(data)} 对图像和标注文件。")
# for item in data[:5]:  # 打印前5个数据项
#     print(item)

def read_batch(data):  # 从数据集中读取随机图像及其标注
    # 选择图像
    ent = data[np.random.randint(len(data))]  # 选择随机条目
    Img = cv2.imread(ent["image"])[..., ::-1]  # 读取图像
    ann_map = cv2.imread(ent["annotation"])  # 读取标注

    # 调整图像大小
    r = np.min([1024 / Img.shape[1], 1024 / Img.shape[0]])  # 缩放因子
    Img = cv2.resize(Img, (int(Img.shape[1] * r), int(Img.shape[0] * r)))
    ann_map = cv2.resize(ann_map, (int(ann_map.shape[1] * r), int(ann_map.shape[0] * r)), interpolation=cv2.INTER_NEAREST)

    # 合并容器和材料标注
    mat_map = ann_map[:, :, 0]  # 材料标注地图
    ves_map = ann_map[:, :, 2]  # 容器标注地图
    mat_map[mat_map == 0] = ves_map[mat_map == 0] * (mat_map.max() + 1)  # 合并地图

    # 获取二进制掩码和点
    inds = np.unique(mat_map)[1:]  # 加载所有索引
    points = []
    masks = [] 
    for ind in inds:
        mask = (mat_map == ind).astype(np.uint8)  # 制作二进制掩码
        masks.append(mask)
        coords = np.argwhere(mask > 0)  # 获取掩码中的所有坐标
        yx = np.array(coords[np.random.randint(len(coords))])  # 选择随机点/坐标
        points.append([[yx[1], yx[0]]])

    return Img, np.array(masks), np.array(points), np.ones([len(masks), 1])

sam2_checkpoint="./checkpoints/sam2.1_hiera_large.pt"# 模型权重路径
model_cfg="./configs/sam2.1/sam2.1_hiera_l.yaml"# 模型配置
sam2_model=build_sam2(model_cfg, sam2_checkpoint, device="cuda") # 加载模型
predictor=SAM2ImagePredictor(sam2_model) # 加载网络

