import sys
import argparse
import numpy as np
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt

from keras.preprocessing import image
from keras.models import load_model
from keras.applications.inception_v3 import preprocess_input

# InceptionV3框架中固定的图像尺寸
target_size = (299, 299)

# 规定类别的名称
labels = ("background", "normal", "tumer")

# 模型文件路径
model_dir = "D:\\Inception-Camelyon17\\modules\\fine_tune_model.h5"

# 测试文件标签文本
test_label_dir = "F:\\ai_data\\camelyon17\\research_data\\test_label.txt"

# 测试文件的数量
test_file_number = 260

# 预测函数
def predict(model, img):
    """predict function
    
    Arguments:
        model {[keras model]} -- [model loaded from training]
        img {[image file]} -- [image data]
    """
    if img.size != target_size:
        img = img.resize(target_size)

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x)
    return preds[0]

def getFileNames(label_dir):
    # 获取测试图像标签文本
    label_file = open(label_dir, "r")
    label_text = label_file.readlines()
    file_name = []
    class_idx = []
    for idx in range(0, test_file_number):
        context = label_text[idx]
        r = context.split(" ")
        file_name.append(r[0])
        class_idx.append(r[1].split("\n")[0])
    return file_name, class_idx

def main():
    # 载入模型
    model = load_model(model_dir)

    # 获取图像文件名称以及对应标签
    file_names, classes_idx = getFileNames(test_label_dir)

    positive_num = 0
    negative_num = 0
    
    for idx in range(test_file_number):
        img = Image.open(file_names[idx])
        preds = predict(model, img)
        
        class_name = classes_idx[idx]
        img_file_name = file_names[idx]
        
        class_number = len(preds)
        pred_max = 0
        pred_pos = 0
        for i in range(class_number):
            if preds[i] > pred_max:
                pred_max = preds[i]
                pred_pos = i
        if int(pred_pos) != int(class_name):
            negative_num += 1
            print("Error : {0} : {1} ( {2} ) :: {3:3.2f}%".format(img_file_name, pred_pos, class_name, pred_max))
        else:
            positive_num += 1
        
    accuracy_positive = float(positive_num) / float(test_file_number) * 100.0
    accuracy_negative = float(negative_num) / float(test_file_number) * 100.0
    print("accuracy : {0:3.2f}[%]".format(accuracy_positive))



if __name__ == '__main__':
    main()