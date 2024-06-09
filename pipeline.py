import cv2
import numpy as np
import os
from PIL import Image, ImageOps
from load_data import test_dataloader

def resize_and_pad_image(image, target_size):
    """
    Resize and pad the image to the target size while maintaining aspect ratio.

    Args:
        image (PIL.Image): The input image to be resized and padded.
        target_size (tuple): The target size (width, height).

    Returns:
        PIL.Image: The resized and padded image.
    """
    img_ratio = image.width / image.height
    target_ratio = target_size[0] / target_size[1]

    if target_ratio > img_ratio:
        scale_factor = target_size[1] / image.height
        new_size = (int(image.width * scale_factor), target_size[1])
    else:
        scale_factor = target_size[0] / image.width
        new_size = (target_size[0], int(image.height * scale_factor))
    image = image.resize(new_size, Image.Resampling.LANCZOS)

    pad_width = (target_size[0] - image.width) // 2
    pad_height = (target_size[1] - image.height) // 2

    padded_image = ImageOps.expand(image, border=(pad_width, pad_height), fill="black")

    return padded_image

def read_and_crop(folder_path, files):
    """
    Read images, apply thresholding, and crop regions of interest.

    Args:
        folder_path (str): The path to the folder containing the images.
        files (list): List of image filenames to be processed.

    Returns:
        str: The path to the folder containing cropped images.
    """
    for file in files:
        path = os.path.join(folder_path, file)
        new_save_path = os.path.join(r'D:\classification\test-processed', file[:-4])
        save_cropped_path = os.path.join(new_save_path, 'cropped_images')
        os.makedirs(new_save_path, exist_ok=True)
        os.makedirs(save_cropped_path, exist_ok=True)

        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        original_image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        print(image.shape)

        thresh = 80
        _, thresholded = cv2.threshold(image, thresh, 255, cv2.THRESH_BINARY)

        mask_name = os.path.join(new_save_path, f'Thresholded_{thresh}.png')
        cv2.imwrite(mask_name, thresholded)

        binary_image = thresholded
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        area_threshold = 500
        image_with_boxes = original_image.copy()

        for i, contour in enumerate(contours):
            x, y, w, h = cv2.boundingRect(contour)
            margin = 10
            x1 = max(x - margin, 0)
            y1 = max(y - margin, 0)
            x2 = min(x + w + margin, original_image.shape[1])
            y2 = min(y + h + margin, original_image.shape[0])

            area = w * h
            if area > area_threshold:
                cv2.rectangle(image_with_boxes, (x, y), (x + w, y + h), (0, 0, 255), 5)

                cropped_image = original_image[y1:y2, x1:x2]
                cropped_name = os.path.join(save_cropped_path, f'Cropped_Image_{i}.png')
                cv2.imwrite(cropped_name, cropped_image)

        boximage_name = os.path.join(new_save_path, f'Image_with_Bounding_Boxes_{thresh}.png')
        cv2.imwrite(boximage_name, image_with_boxes)

        return save_cropped_path

def reshape_and_save(cropped_path, target_folder_path):
    """
    Resize and save the cropped images to the target folder.

    Args:
        cropped_path (str): The path to the folder containing cropped images.
        target_folder_path (str): The path to the target folder for resized images.
    """
    for file in os.listdir(cropped_path):
        if file.endswith('.png'):
            with Image.open(os.path.join(cropped_path, file)) as img:
                resized_img = resize_and_pad_image(img, target_size)
                resized_img.save(os.path.join(target_folder_path, 'resized_' + file))

def predict_classification(target_folder_path):
    """
    Load the model and predict the classification of images in the target folder.

    Args:
        target_folder_path (str): The path to the folder containing images for prediction.
    """
    import torch
    import torch.nn.functional as F
    from models.densenet import densenet169
    import cfg

    model = densenet169(num_classes=cfg.NUM_CLASSES)
    if torch.cuda.is_available():
        model.cuda()

    trained_model = cfg.TRAINED_MODEL
    state_dict = torch.load(trained_model)

    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)

    print('Finished loading model!')
    model.eval()

    num_predicted_ones = 0
    for batch_images, paths in test_dataloader:
        print(paths)
        with torch.no_grad():
            if torch.cuda.is_available():
                batch_images = batch_images.cuda()
            out = model(batch_images)
            out1 = torch.sigmoid(out)
            prediction = torch.max(out1, 1)[1]
            num_predicted_ones += (prediction == 1).sum().item()

    print('Number of samples predicted as 1: {}'.format(num_predicted_ones))

# Main script execution
folder_path = r'D:\classification\test'
files = os.listdir(folder_path)
cropped_path = read_and_crop(folder_path, files)

target_folder_path = cropped_path.replace('cropped_images', 'resized_images')
os.makedirs(target_folder_path, exist_ok=True)
target_size = (96, 96)
reshape_and_save(cropped_path, target_folder_path)
predict_classification(target_folder_path)
