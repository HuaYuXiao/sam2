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
print(f"找到 {len(data)} 对图像和标注文件。")
for item in data[:5]:  # 打印前5个数据项
    print(item)

def read_batch(data):  # 从数据集（LabPics）中读取随机图像及其标注
    # 选择图像
    ent = data[np.random.randint(len(data))]  # 选择随机条目
    Img = cv2.imread(ent["image"], cv2.IMREAD_GRAYSCALE)  # 读取图像并转换为 RGB
    ann_map = cv2.imread(ent["annotation"], cv2.IMREAD_GRAYSCALE)  # 读取标注为灰度图

    # 调整图像大小
    r = np.min([1024 / Img.shape[1], 1024 / Img.shape[0]])  # 缩放因子
    Img = cv2.resize(Img, (int(Img.shape[1] * r), int(Img.shape[0] * r)))
    ann_map = cv2.resize(ann_map, (int(ann_map.shape[1] * r), int(ann_map.shape[0] * r)), interpolation=cv2.INTER_NEAREST)

    # 获取二进制掩码和点
    inds = np.unique(ann_map)[1:]  # 加载所有索引，跳过背景（假设背景为0）
    points = []
    masks = []
    for ind in inds:
        mask = (ann_map == ind).astype(np.uint8)  # 制作二进制掩码
        masks.append(mask)
        coords = np.argwhere(mask > 0)  # 获取掩码中的所有坐标
        if len(coords) == 0:
            continue  # 跳过空掩码
        yx = coords[np.random.randint(len(coords))]  # 选择随机点/坐标
        points.append([[int(yx[1]), int(yx[0])]])  # 转换为 [x, y] 格式

    # 转换为numpy数组
    masks = np.array(masks)
    points = np.array(points)
    labels = np.ones([len(masks), 1])  # 假设所有掩码都有标签

    return Img, masks, points, labels

image, mask, input_point, input_label = read_batch(data)  # 加载数据批次
print(image.size())

# # 加载SAM2模型
# sam2_checkpoint = "./checkpoints/sam2.1_hiera_large.pt"  # 模型权重路径
# model_cfg = "./configs/sam2.1/sam2.1_hiera_l.yaml"  # 模型配置
# sam2_model = build_sam2(model_cfg, sam2_checkpoint, device="cuda")  # 加载模型
# predictor = SAM2ImagePredictor(sam2_model)  # 加载网络

# # 设置模型训练模式
# sam2_model.train()
# predictor.model.sam_mask_decoder.train(True)  # 启用掩码解码器的训练
# predictor.model.sam_prompt_encoder.train(True)  # 启用提示编码器的训练

# # 定义优化器和混合精度训练
# optimizer = torch.optim.AdamW(params=predictor.model.parameters(), lr=1e-5, weight_decay=4e-5)
# scaler = torch.cuda.amp.GradScaler()  # 设置混合精度

# # 定义训练轮数
# num_epochs = 10  # 可以根据需求调整

# # 训练主循环
# for epoch in range(num_epochs):
#     running_loss = 0.0
#     for itr in range(len(data)):
#         # 混合精度上下文
#         with torch.cuda.amp.autocast():
#             # 读取数据批次
#             image, mask, input_point, input_label = read_batch(data)  # 加载数据批次

#             if mask.shape[0] == 0:
#                 continue  # 忽略空批次

#             # 将图像转换为张量并移动到GPU
#             image_tensor = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to("cuda")  # [1, C, H, W]

#             # 处理掩码和点
#             masks_tensor = torch.tensor(mask, dtype=torch.float32).unsqueeze(0).to("cuda")  # [1, num_masks, H, W]
#             points_tensor = torch.tensor(input_point, dtype=torch.float32).to("cuda")  # [num_masks, 1, 2]
#             labels_tensor = torch.tensor(input_label, dtype=torch.float32).to("cuda")  # [num_masks, 1]

#             # 设置图像给SAM模型
#             predictor.set_image(image_tensor)

#             # 准备提示（points和labels）
#             mask_input, unnorm_coords, labels, unnorm_box = predictor._prep_prompts(
#                 points_tensor, labels_tensor, box=None, mask_logits=None, normalize_coords=True
#             )
#             sparse_embeddings, dense_embeddings = predictor.model.sam_prompt_encoder(
#                 points=(unnorm_coords, labels), boxes=None, masks=None
#             )

#             # 预测掩码
#             batched_mode = unnorm_coords.shape[0] > 1  # multi mask prediction
#             high_res_features = [feat_level[-1].unsqueeze(0) for feat_level in predictor._features["high_res_feats"]]
#             low_res_masks, prd_scores, _, _ = predictor.model.sam_mask_decoder(
#                 image_embeddings=predictor._features["image_embed"][-1].unsqueeze(0),
#                 image_pe=predictor.model.sam_prompt_encoder.get_dense_pe(),
#                 sparse_prompt_embeddings=sparse_embeddings,
#                 dense_prompt_embeddings=dense_embeddings,
#                 multimask_output=True,
#                 repeat_image=batched_mode,
#                 high_res_features=high_res_features,
#             )
#             prd_masks = predictor._transforms.postprocess_masks(low_res_masks, predictor._orig_hw[-1])  # Upscale the masks to the original image resolution

#             # 分割损失
#             prd_mask = torch.sigmoid(prd_masks[:, 0])  # 将logit图转换为概率图
#             gt_mask = masks_tensor[:, 0]  # 假设每个掩码对应一个GT掩码

#             # 计算交叉熵损失
#             seg_loss = (
#                 (-gt_mask * torch.log(prd_mask + 1e-5) - (1 - gt_mask) * torch.log((1 - prd_mask) + 1e-5))
#                 .mean()
#             )

#             # 分数损失（可选）
#             inter = (gt_mask * (prd_mask > 0.5)).sum(dim=(1, 2))
#             iou = inter / (
#                 gt_mask.sum(dim=(1, 2)) + (prd_mask > 0.5).sum(dim=(1, 2)) - inter
#             )
#             score_loss = torch.abs(prd_scores[:, 0] - iou).mean()

#             # 合并损失
#             loss = seg_loss + score_loss * 0.05  # 混合损失

#         # 反向传播和优化
#         optimizer.zero_grad()
#         scaler.scale(loss).backward()
#         scaler.step(optimizer)
#         scaler.update()

#         running_loss += loss.item()

#         # 每1000步保存一次模型
#         if itr % 1000 == 0:
#             torch.save(predictor.model.state_dict(), "model.torch")
#             print(f"保存模型到 model.torch at step {itr}")

#         # 打印损失信息
#         if itr % 100 == 0:
#             mean_iou = iou.mean().item()
#             print(f"Epoch [{epoch+1}/{num_epochs}], Step [{itr+1}/{len(data)}], Loss: {loss.item():.4f}, IOU: {mean_iou:.4f}")

#     epoch_loss = running_loss / len(data)
#     print(f"Epoch [{epoch+1}/{num_epochs}] 完成，平均损失: {epoch_loss:.4f}")

#     # 每个epoch结束后保存模型
#     torch.save(predictor.model.state_dict(), f"./checkpoints/model_epoch_{epoch+1}.torch")

# print("训练完成！")
