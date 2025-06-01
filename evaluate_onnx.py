import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import onnxruntime as ort
from torchvision import transforms

def load_image(path):
    image = Image.open(path).convert("RGB").resize((320, 320))
    transform = transforms.ToTensor()
    tensor = transform(image).numpy()  # (3, 320, 320)
    return np.expand_dims(tensor, axis=0).astype(np.float32)  # (1, 3, 320, 320)

def load_mask(path):
    mask = Image.open(path).convert("L").resize((320, 320))
    return (np.array(mask) > 127).astype(np.uint8)

def iou(pred, gt):
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return intersection / union if union else 1.0

def dice(pred, gt):
    intersection = np.logical_and(pred, gt).sum()
    return 2 * intersection / (pred.sum() + gt.sum()) if (pred.sum() + gt.sum()) else 1.0

def evaluate(model_path, image_dir, mask_dir):
    session = ort.InferenceSession(model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg') or f.endswith('.png')]

    all_iou = []
    all_dice = []

    for fname in tqdm(image_files):
        img_path = os.path.join(image_dir, fname)
        mask_path = os.path.join(mask_dir, os.path.splitext(fname)[0] + '.png')

        if not os.path.exists(mask_path):
            print(f"[!] No mask for {fname}, skipping.")
            continue

        img = load_image(img_path)
        gt = load_mask(mask_path)

        pred = session.run([output_name], {input_name: img})[0]
        pred = pred.squeeze()
        pred = (pred > 0.5).astype(np.uint8)

        all_iou.append(iou(pred, gt))
        all_dice.append(dice(pred, gt))

    print(f"✅ Mean IoU:  {np.mean(all_iou):.4f}")
    print(f"✅ Mean Dice: {np.mean(all_dice):.4f}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--images', required=True)
    parser.add_argument('--masks', required=True)
    args = parser.parse_args()

    evaluate(args.model, args.images, args.masks)
