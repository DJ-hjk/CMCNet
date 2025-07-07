import torch
import numpy as np
from torch.utils.data import DataLoader
import os
from utils.cracktree_dataloaderkeshi import Datases_loader as dataloader
import matplotlib.pyplot as plt
#from src.Net_newEEM_SE_cnn_ffm import Net_final
#from src.Net_OnlyCNN import Net_final
#from src.Net_OnlyMamba import Net_final
#from src.Net_noEEM import Net_fina8
#from src.Net_noFFM import Net_final
from src.Net_final import Net_final

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batchsz = 1


model = Net_final().to(device)
#model.eval()

savedir = r'path/CMCNet_DeepCrack_epoch100.pth'
"""
imgdir = r'path1/CQBCD_final/test_img'
labdir = r'path1/CQBCD_final/test_mask'

imgdir = r'Cracktree_deepcrack/test_img'
labdir = r'Cracktree_deepcrack/test_mask'

imgdir = r'DeepCrack1/test_img'
labdir = r'DeepCrack1/test_lab'
"""
imgdir = r'DeepCrack1/test_img'
labdir = r'DeepCrack1/test_lab'

imgsz = 512  # 图像大小
resultsdir = r'CMCNet_DeepCrack_epoch100'

# 加载数据集
dataset = dataloader(imgdir, labdir, imgsz, imgsz)
testsets = DataLoader(dataset, batch_size=batchsz, shuffle=False)

def calculate_metrics(pred, lab):
    pred_binary = pred > 0.5
    lab_binary = lab > 0.5 

    crack_intersection = np.sum(np.logical_and(pred_binary, lab_binary))
    crack_union = np.sum(np.logical_or(pred_binary, lab_binary))
    crack_iou = crack_intersection / (crack_union + 1e-6)

    background_intersection = np.sum(np.logical_and(~pred_binary, ~lab_binary))
    background_union = np.sum(np.logical_or(~pred_binary, ~lab_binary))
    background_iou = background_intersection / (background_union + 1e-6)

    true_positive = np.sum(np.logical_and(pred_binary, lab_binary))
    false_positive = np.sum(np.logical_and(pred_binary, ~lab_binary))
    false_negative = np.sum(np.logical_and(~pred_binary, lab_binary))

    precision = true_positive / (true_positive + false_positive + 1e-6)
    recall = true_positive / (true_positive + false_negative + 1e-6)

    if precision + recall > 0:
        f1_score = 2 * (precision * recall) / (precision + recall)
    else:
        f1_score = 0

    total_pixels = pred_binary.size
    correct_predictions = np.sum(pred_binary == lab_binary)
    accuracy = correct_predictions / total_pixels

    dice = (2 * crack_intersection) / (np.sum(pred_binary) + np.sum(lab_binary) + 1e-6)

    return crack_iou, background_iou, precision, recall, f1_score, accuracy, dice

def save_image_as_png(image, file_path):
    if image.ndim == 3:
        image = image.squeeze(0)
    plt.imsave(file_path, image, cmap='gray')

def save_mask_as_png(mask, file_path):
    if mask.ndim == 3:
        mask = mask.squeeze(0)
    binary_mask = (mask > 0.5 ).astype(np.uint8) * 255
    plt.imsave(file_path, binary_mask, cmap='gray')

def test():
    model.load_state_dict(torch.load(savedir, map_location=device), strict=False)
    if not os.path.exists(resultsdir):
        os.makedirs(resultsdir)

    total_crack_intersection = 0
    total_crack_union = 0
    total_background_intersection = 0
    total_background_union = 0
    total_true_positive = 0
    total_false_positive = 0
    total_false_negative = 0
    total_correct_predictions = 0
    total_pixels = 0

    for idx, samples in enumerate(testsets):
        img, lab = samples['image'], samples['mask']
        img, lab = img.to(device), lab.to(device)

        #pred, Lout = model(img, lab)
        pred = model(img)
        pred_np = pred.detach().cpu().numpy()
        lab_np = lab.detach().cpu().numpy()

        #save_image_as_png(img.detach().cpu().numpy()[0, 0], os.path.join(resultsdir, f'image{idx + 1}.png'))
        save_mask_as_png(lab_np[0, 0], os.path.join(resultsdir, f'label{idx + 1}.png'))
        save_mask_as_png(pred_np[0, 0], os.path.join(resultsdir, f'pred{idx + 1}.png'))

        crack_intersection = np.sum(np.logical_and(pred_np[0, 0] > 0.5 , lab_np[0, 0] > 0.5 ))
        crack_union = np.sum(np.logical_or(pred_np[0, 0] > 0.5 , lab_np[0, 0] > 0.5 ))
        background_intersection = np.sum(np.logical_and(pred_np[0, 0] <=0.5 , lab_np[0, 0] <= 0.5 ))
        background_union = np.sum(np.logical_or(pred_np[0, 0] <= 0.5 , lab_np[0, 0] <= 0.5 ))
        true_positive = np.sum(np.logical_and(pred_np[0, 0] > 0.5 , lab_np[0, 0] > 0.5 ))
        false_positive = np.sum(np.logical_and(pred_np[0, 0] > 0.5 , lab_np[0, 0] <= 0.5 ))
        false_negative = np.sum(np.logical_and(pred_np[0, 0] <= 0.5 , lab_np[0, 0] >0.5 ))
        correct_predictions = np.sum(pred_np[0, 0] == lab_np[0, 0])
        total_pixels += pred_np[0, 0].size

        total_crack_intersection += crack_intersection
        total_crack_union += crack_union
        total_background_intersection += background_intersection
        total_background_union += background_union
        total_true_positive += true_positive
        total_false_positive += false_positive
        total_false_negative += false_negative
        total_correct_predictions += correct_predictions

    print(f'total_correct_predictions: {total_correct_predictions:.4f}')

    global_crack_iou = total_crack_intersection / (total_crack_union + 1e-6)
    global_background_iou = total_background_intersection / (total_background_union + 1e-6)
    global_precision = total_true_positive / (total_true_positive + total_false_positive + 1e-6)
    global_recall = total_true_positive / (total_true_positive + total_false_negative + 1e-6)
    global_f1 = 2 * (global_precision * global_recall) / (global_precision + global_recall + 1e-6)
    global_accuracy = total_correct_predictions / total_pixels
    global_miou = (global_crack_iou + global_background_iou) / 2

    print(f'Global Precision: {global_precision:.4f}')
    print(f'Global Recall: {global_recall:.4f}')
    print(f'Global F1 Score: {global_f1:.4f}')
    print(f'Global Accuracy: {global_accuracy:.4f}')
    print(f'Global Mean IOU (mIoU): {global_miou:.4f}')

if __name__ == '__main__':
    test()
