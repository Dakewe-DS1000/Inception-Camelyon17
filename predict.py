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
test_file_number = 4000

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

    positive_num = 0                    # 正确识别的图像数量
    negative_Normal2Tumer_num = 0       # 将正常的组织识别成肿瘤组织的图像数量
    negative_Tumer2Normal_num = 0       # 将肿瘤组织识别成正常组织的图像数量
    
    for idx in range(test_file_number):
    # 打开当前图像
        img = Image.open(file_names[idx])
    # 对当前图像进行识别，获得识别结果 perds
    # preds是一个数组，内含有所有类别的识别概率结果
        preds = predict(model, img)
        
    # 根据当前文件获取类别的真值 class_name
        class_name = classes_idx[idx]
    # 当前图像文件的文件名
        img_file_name = file_names[idx]
    # 识别结果的数组长度即为类别的数量
        class_number = len(preds)
    # 识别数组preds中最大概率值及其对应的位置
        pred_max = 0
        pred_pos = 0
    # 获得preds中最大概率 pred_max 及其对应的位置 pred_pos
        for i in range(class_number):
            if preds[i] > pred_max:
                pred_max = preds[i]
                pred_pos = i
    # preds数组中最大概率的位置就是识别的种类

    # 识别精度判别代码 - 通用代码
        # 当最大概率位置与类别真值不同时，表示识别出现失误
        #if int(pred_pos) != int(class_name):
            # negative增涨1步进的计数
            #negative_num += 1
            #print("Error : {0} : {1} ( {2} ) :: {3:3.2f}%".format(img_file_name, pred_pos, class_name, pred_max))
        # 当最大概率位置与类别真值一致时，表示识别正确
        #else:
            # positive增涨1步进的计数
            #positive_num += 1

    # 识别精度判别代码 - 数字病理诊断精度判别
        # 如果识别结果与真值不一致，同时，真值又是Tumer分类的，则认为是识别错误
        # 因为神经网络将Tumer识别成正常的或者背景了，所以应当判为识别错误
        if int(pred_pos) != int(class_name) and int(class_name)==2:
            negative_Tumer2Normal_num += 1
            print("Error : Tumer to Normal : {0} : {1} ({2}) :: {3:3.2f}%".format(img_file_name, pred_pos, class_name, pred_max))
            #str_error_fileName = "F://ai_data/camelyon17/research_data/error/[" + str(idx) + "]Tumer2Normal_Image_" + str(pred_pos) + "(" + str(class_name) + ").jpg"
            #img.save(str_error_fileName)
        # 其他的情况，则无关紧要，都判断为识别正确
        # 注意：这里，Normal的图像被识别成Tumer，也算是正确的
        elif int(pred_pos) == 2 and int(class_name) != 2:
            negative_Normal2Tumer_num += 1
            positive_num += 1
            print("Error : Normal to Tumer : {0} : {1} ({2}) :: {3:3.2f}%".format(img_file_name, pred_pos, class_name, pred_max))
            #str_error_fileName = "F://ai_data/camelyon17/research_data/error/[" + str(idx) + "]Normal2Tumer_Image_" + str(pred_pos) + "(" + str(class_name) + ").jpg"
            #img.save(str_error_fileName)
        else:
            positive_num += 1
    
    # 本程序中：
    # 正的正确率 定义 ： 正确识别的图像数量 / 测试集图像总数量
    # 负的正确率 定义 ： 错误识别的图像数量 / 测试集图像中数量
    accuracy_positive = float(positive_num) / float(test_file_number) * 100.0
    accuracy_negative_Tumer2Normal = float(negative_Tumer2Normal_num) / float(test_file_number) * 100.0
    accuracy_negative_Normal2Tumer = float(negative_Normal2Tumer_num) / float(test_file_number) * 100.0
    print("accuracy : {0:3.2f}[%]".format(accuracy_positive))
    print("negative tumer to normal : {0:3.2f}[%]".format(accuracy_negative_Tumer2Normal))
    print("negative normal to tumer : {0:3.2f}[%]".format(accuracy_negative_Normal2Tumer))



if __name__ == '__main__':
    main()