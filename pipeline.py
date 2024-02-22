import cv2
import numpy as np
import os
import os
from PIL import Image,ImageOps
import numpy as np
from load_data import test_dataloader


def resize_and_pad_image(image, target_size):
    # 等比例缩放图像
    img_ratio = image.width / image.height
    target_ratio = target_size[0] / target_size[1]

    if target_ratio > img_ratio:
        scale_factor = target_size[1] / image.height
        new_size = (int(image.width * scale_factor), target_size[1])
    else:
        scale_factor = target_size[0] / image.width
        new_size = (target_size[0], int(image.height * scale_factor))
    image = image.resize(new_size, Image.Resampling.LANCZOS)

    # 计算填充量
    pad_width = (target_size[0] - image.width) // 2
    pad_height = (target_size[1] - image.height) // 2

    # 添加填充
    padded_image = ImageOps.expand(image, border=(pad_width, pad_height), fill="black")

    return padded_image


def read_and_crop(floder_path,files):
    for file in files:
        path = os.path.join(floder_path, file)
        new_save_path = os.path.join(r'D:\classification\test-processed', file[:-4])
        save_Cropped_path = os.path.join(new_save_path, 'cropped_images')
        os.makedirs(new_save_path, exist_ok=True)
        os.makedirs(save_Cropped_path, exist_ok=True)

        # 读取暗场图像
        # read the .tiff
        # image = cv2.imread('notebooks/images/dog.jpg')
        # path = r"./circ0011.tif"
        # arr = cv.imread(path,1)                          #(2960, 1976, 3)   备注：4波段的影像在opencv的读取方式中，显示为前三个波段，而且读取顺序为BGR
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        original_image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        print(image.shape)
        thresh = 80
        # 应用阈值分割
        _, thresholded = cv2.threshold(image, thresh, 255, cv2.THRESH_BINARY)

        # 显示分割结果
        mask_name = os.path.join(new_save_path, f'Thresholded_{thresh}.png')
        cv2.imwrite(mask_name, thresholded)

        # 裁剪
        binary_image = thresholded

        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        area_threshold = 500

        image_with_boxes = original_image.copy()

        for i, contour in enumerate(contours):
            # 计算边界框
            x, y, w, h = cv2.boundingRect(contour)

            margin = 10  # 边距大小
            x1 = max(x - margin, 0)
            y1 = max(y - margin, 0)
            x2 = min(x + w + margin, original_image.shape[1])
            y2 = min(y + h + margin, original_image.shape[0])

            area = w * h

            if area > area_threshold:
                cv2.rectangle(image_with_boxes, (x, y), (x + w, y + h), (0, 0, 255), 5)

                # 裁剪图像
                cropped_image = original_image[y1:y2, x1:x2]

                # 保存或处理裁剪后的图像
                cropped_name = os.path.join(save_Cropped_path, f'Cropped_Image_{i}.png')
                cv2.imwrite(cropped_name, cropped_image)

        boximage_name = os.path.join(new_save_path, f'Image with Bounding Boxes_{thresh}.png')
        cv2.imwrite(boximage_name, image_with_boxes)

        return save_Cropped_path
def reshape_and_save(cropped_path,target_folder_path):
    # 重塑并保存图像
    for file in os.listdir(cropped_path):
        if file.endswith('.png'):
            with Image.open(os.path.join(cropped_path, file)) as img:
                resized_img = resize_and_pad_image(img, target_size)
                # 保存重塑的图像，可以选择一个新的文件夹或覆盖原文件
                resized_img.save(os.path.join(target_folder_path, 'resized_' + file))

def create_the_target_path ():
    return target_folder_path
def predict_classification(target_folder_path):
    import torch.nn.functional as F
    import torch
    from models.densenet import densenet169
    import cfg

    #
    #
    # ##定义模型的框架
    model = densenet169(num_classes=cfg.NUM_CLASSES)
    # print(model)
    ##将模型放置在gpu上运行
    if torch.cuda.is_available():
        model.cuda()

    ###读取网络模型的键值对
    trained_model = cfg.TRAINED_MODEL
    state_dict = torch.load(trained_model)

    # create new OrderedDict that does not contain `module.`
    ##由于之前的模型是在多gpu上训练的，因而保存的模型参数，键前边有‘module’，需要去掉，和训练模型一样构建新的字典
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        head = k[:7]
        if head == 'module.':
            name = k[7:]  # remove `module.`
        else:

            name = k
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)

    print('Finished loading model!')

    ##进行模型测试时，eval（）会固定下BN与Dropout的参数
    model.eval()
    # eval_acc = 0.0

    # ...（前面的代码保持不变）

    # 用于统计预测为1的样本数量
    num_predicted_ones = 0

    for batch_images, paths in test_dataloader:
        print(paths)
        with torch.no_grad():
            if torch.cuda.is_available():
                batch_images = batch_images.cuda()

        out = model(batch_images)
        out1 = F.sigmoid(out)

        prediction = torch.max(out1, 1)[1]

        # 更新预测为1的样本数量
        num_predicted_ones += (prediction == 1).sum().item()

    # 打印准确度和预测为1的样本数量
    # print('Accuracy of the batch: {:.6f}'.format(eval_acc / len(val_datasets)))
    print('Number of samples predicted as 1: {}'.format(num_predicted_ones))

#输入要测试的图像数据（夹）
floder_path = r'D:\classification\test'
files = os.listdir(floder_path)
cropped_path = read_and_crop(floder_path, files)
#reshape后的path
target_folder_path = cropped_path.replace('cropped_images','resized_images')
os.makedirs(target_folder_path, exist_ok=True)
target_size = (96,96)
reshape_and_save(cropped_path,target_folder_path)
predict_classification(target_folder_path)


