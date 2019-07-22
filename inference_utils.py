import cv2
import torch
import numpy as np
from PIL import Image
from network.textnet import TextNet
from util.detection import TextDetector
from util.augmentation import BaseTransform
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

def load_image(path):
    image = Image.open(path)
    image = np.array(image)
    return image


def load_detector_and_transforms(model_path, input_img_size, tr_thresh=0.4, tcl_thresh=0.5, device='cuda'):
    textsnake_model = TextNet()
    textsnake_model.load_state_dict(torch.load(model_path)['model'])
    textsnake_model.to(device)
    detector = TextDetector(textsnake_model, tr_thresh, tcl_thresh)

    test_transforms = transforms.Compose([
        BaseTransform(size=input_img_size, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        lambda x: np.expand_dims(x[0].transpose(2, 0, 1), 0),
        torch.tensor
    ])

    return detector, test_transforms


def visualize_detection(image, tr, tcl, contours, tr_thresh=0.4, tcl_thresh=0.5, figsize=(12,8)):
    image_show = image.copy()
    image_show = np.ascontiguousarray(image_show[:, :, ::-1])
    image_show = cv2.polylines(image_show, contours, True, (0, 0, 255), 3)

    tr = (tr > tr_thresh).astype(np.uint8)
    tcl = (tcl > tcl_thresh).astype(np.uint8)

    tr = cv2.cvtColor(tr * 255, cv2.COLOR_GRAY2BGR)
    tcl = cv2.cvtColor(tcl * 255, cv2.COLOR_GRAY2BGR)
    image_show = np.concatenate([image_show, tr, tcl], axis=1)
    image_show = cv2.cvtColor(image_show, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=figsize)
    plt.imshow(image_show)
    plt.show()


def predict_single_image(detector, transformed_img, device='cuda'):
    with torch.no_grad():
        tcl_contours, det_result = detector.detect(transformed_img.to(device));
        tr_pred, tcl_pred = det_result['tr'], det_result['tcl']
    return tcl_contours, tr_pred[1], tcl_pred[1]
