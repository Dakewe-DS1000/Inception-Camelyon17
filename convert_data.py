import numpy as np
import tensorflow as tf
import os
from shutil import copyfile, rmtree
#数据打包处理，将原始图像数据集转换成Keras要求的格式：
#每一个子文件夹代表一类，其中有该类所有的图像数据
'''
data
    customs folder
        classA
            image1
            image2
            ...
        classB
            image1
            image2
            ...
        classC
            ...
        ...

    research
        test
            0
                image1
                image2
                ...
            1
                image1
                image2
                ...
            2
                image1
                image2
                ...
        train
            0
                image1
                image2
                ...
            1
                image1
                image2
                ...
            2
                image1
                image2
                ...
        validation
            0
                image1
                image2
                ...
            1
                image1
                image2
                ...
            ...
'''
#图像原始数据文件夹
source_data_folder = "F://ai_data/camelyon17/train_data"
#新的文件夹
research_data_folder = "F://ai_data/camelyon17/research_data"
#类名文本
label_text_file = source_data_folder + "//labels.txt"

train_num = 210000      #用于训练的图像数目
val_num   = 719      #用于训练测试的图像数目
test_num  = 4000      #用于最终测试的图像数目

def convert_class_data():
    np.random.seed(0)    #使用统一的Seed，保证每次随机的结果都相同
    #打开已经生成的标签文件
    label_file = open(label_text_file)
    #按行读取标签文件中的文本信息
    labels = label_file.readlines()
    #随机打乱标签文本信息的顺序
    np.random.shuffle(labels)
    current_i = 0

    current_i = save_images(current_i=current_i, phase="train", d_size=train_num, labels=labels)
    current_i = save_images(current_i=current_i, phase="test", d_size=test_num, labels=labels)
    current_i = save_images(current_i=current_i, phase="validation", d_size=val_num, labels=labels)


def save_images(current_i, phase, d_size, labels):
    if phase == "train":        #选择存储训练集数据
        dst_folder = research_data_folder + "\\train\\"
    elif phase == "test":       #选择存储测试集数据
        dst_folder = research_data_folder + "\\test\\"
    elif phase == "validation": #选择存储训练测试集数据
        dst_folder = research_data_folder + "\\validation\\"
    else:
        print("phase error : {0}".format(phase))
        exit()
    #打开新的标签文本文件，准备录入不同数据集的标签信息，以作备用
    label_file = open(research_data_folder+"\\"+phase+"_label.txt", mode="w")
    for i in range(current_i, current_i+d_size):
        #获取被打乱顺序的标签
        item = labels[i]
        #根据空格分割文件名称和类别名称 
        r = item.split(" ")
        #获取文件名称
        img_source_path = r[0]
        #获取类别名称，注意需要把最后的换行符去掉
        img_class_name  = r[1].split("\n")[0]
        #创建新的路径，以拷贝图像文件
        img_dst_path = dst_folder + img_class_name + "\\" + os.path.basename(img_source_path)
        #如果新的路径不存在，则新建文件夹
        if not os.path.exists(os.path.dirname(img_dst_path)):
            os.makedirs(os.path.dirname(img_dst_path))
        #将文件拷贝到新的路径中
        copyfile(img_source_path, img_dst_path)
        print("{0} copied".format(img_dst_path))
        #顺手完成标签文本文件，以作备用
        label_text = img_dst_path + " " + img_class_name + "\n"
        #标签写入新的文本文件
        label_file.write(label_text)
        current_i = i
    label_file.close()
    return current_i    

def image_labeling():
    #数据目录
    directories = []
    #类别名称
    class_names = []
    #图像文件列表
    image_filenames = []

    #在数据根目录下寻找文件夹
    for filename in os.listdir(source_data_folder):
        #定位当前文件夹
        path = os.path.join(source_data_folder, filename)
        #如果路径为path的是文件夹
        if os.path.isdir(path):
            directories.append(path)        #录入数据目录
    
    #循环数据目录文件夹
    for i, directory in enumerate(directories):
        #在数据目录文件夹中遍历图像文件
        for filename in os.listdir(directory):
            path = os.path.join(directory, filename)
            #加入所有图像文件名
            image_filenames.append(path)
            #加入图像所对应的标签编号
            class_names.append(str(i))

    #打开标签文本文件，准备录入标签数据
    label_file = open(label_text_file, mode="w")
    for idx, item in enumerate(image_filenames):
        text = item + " " + class_names[idx] + "\n"
        print(text)
        label_file.write(text)
    label_file.close()

def main():
    print("Start to convert data")
    image_labeling()
    convert_class_data()


if __name__ == '__main__':
    main()
if __name__ == '__main__':
    main()