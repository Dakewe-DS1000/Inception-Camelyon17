from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.layers import GlobalAveragePooling2D, Dense
from keras.models import Model
from keras.utils.vis_utils import plot_model
from keras.optimizers import Adagrad
from keras.callbacks import TensorBoard

import os
import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt
# 仅使用GPU0 GeForce GTX 1080Ti
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# 迁移训练模型保存
save_file_name_TransferTraining = "D:\\Inception-Camelyon17\\modules\\transfer_learning_model.h5"
# 微调模型保存
save_file_name_fileTune = "D:\\Inception-Camelyon17\\modules\\fine_tune_model.h5"

# 训练数据集的路径
train_data_dir = "F:\\ai_data\\camelyon17\\research_data\\train"
# 训练测试数据集的路径
val_data_dir   = "F:\\ai_data\\camelyon17\\research_data\\validation"
# 测试数据集的路径
test_data_dir  = "F:\\ai_data\\camelyon17\\research_data\\test"
# 类别数目
class_number = 3
# 每一个epoch内的训练次数 default : 4000, 20000, 40000, 80000
# 官方建议的值 = 训练用图像总数量 / batch_size
# 本程序：训练用图像总数量=400000， batch_size=64, 所以，step_per_epoch = 6250
step_per_epoch = 6250
# epoch数目
epoch = 10
# 运行测试的间隔点
val_step = 1

# 绘图函数
def plot_training(history):
    acc      = history.history["acc"]
    val_acc  = history.history["val_acc"]
    loss     = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs   = range(len(acc))
    
    plt.plot(epochs, acc, "r.")
    plt.plot(epochs, val_acc, "r")
    plt.title("Training and validation accuracy")
    plt.figure()
    
    plt.plot(epochs, loss, "r.")
    plt.plot(epochs, val_loss, "r-")
    plt.title("Training and validation loss")
    plt.show()

# 转移训练设置函数
# Transfer Learning：将骨架模型的所有层都设置为不可训练
def setup_to_transfer_learning(model, base_model):
    for layer in base_model.layers:
        layer.trainable = False
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Fine Tune：将骨架模型中的前几层设置为不可训练，后面的所有Inception模块都设置为可训练
def setup_to_fine_tune(model, base_model):
    GAP_LAYER = class_number
    for layer in base_model.layers[:GAP_LAYER+1]:
        layer.trainable = False
    for layer in base_model.layers[GAP_LAYER+1:]:
        layer.trainable = True
    model.compile(optimizer=Adagrad(lr=0.0001), loss="categorical_crossentropy", metrics=["accuracy"])

def main():
    # 准备训练数据
    Gen_Train_Data = ImageDataGenerator(
        preprocessing_function=preprocess_input, # 将原始数据归一化到-1.0~+1.0
        rotation_range        = 45.0,              # 设置随机化的旋转随机
        width_shift_range     = 0.0,              # 设置随机化的平移随机
        height_shift_range    = 0.0,              # 设置随机化的平移垂直随机
        shear_range           = 0.0,              # 逆时针方向的剪切变换角度
        zoom_range            = 0.0,              # 图像缩放比例
        horizontal_flip       = True,             # 进行随机水平旋转
        vertical_flip         = True,             # 进行随机垂直旋转
    )

    # 准备训练测试数据
    Gen_Val_Data = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range        = 45.0,
        width_shift_range     = 0.0,
        height_shift_range    = 0.0,
        shear_range           = 0.0,
        zoom_range            = 0.0,
        horizontal_flip       = True,
        vertical_flip         = True
    )

    # 读取训练数据
    train_data_generator = Gen_Train_Data.flow_from_directory(directory   = train_data_dir,
                                                              target_size = (299, 299),
                                                              batch_size  = 64)
    # 读取训练测试数据
    val_data_generator = Gen_Val_Data.flow_from_directory(directory   = val_data_dir,
                                                          target_size = (299, 299),
                                                          batch_size  = 64)

    # 构建基础模型
    # weights 使用 imagenet 模型进行迁移学习
    # include_top False表示去掉原始的全连接层，输出一个8X8X2048的张量；True则表示使用原有的全连接层，1000个分类输出
    # 一般做迁移训练，去掉顶层，后面街上各种自定义的新层，这已经成为了训练任务的惯用套路
    # Inception基础骨架
    base_model = InceptionV3(weights='imagenet', include_top=False);

    # 输出层先用GlobalAveragePooling2D函数将8 * 8 * 2048的输出转换成1 * 2048的张量
    # 后面接了一个1024个节点的全连接层，最后是一个17个节点的输出层，用softmax激活函数
    # 增加新的输出层
    x = base_model.output
    # GlobalAveragePooling2D 将 MxNxC 的张量转换成 1xC 张量，C是通道数
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation="relu")(x)
    predictions = Dense(class_number, activation="softmax")(x)
    # 迁移学习模型骨架
    model = Model(inputs=base_model.input, outputs=predictions)

    # 输出迁移学习框架图
    plot_model(model, to_file="model.png",                       show_shapes = False, show_layer_names = False)
    plot_model(model, to_file="model_with_Shapes.png",           show_shapes = True,  show_layer_names = False)
    plot_model(model, to_file="model_with_LayerNames.png",       show_shapes = False, show_layer_names = True)
    plot_model(model, to_file="model_with_Shape&LayerNames.png", show_shapes = True,  show_layer_names = True)

    # 学习过程TensorBoard数据准备
    tb_log = TensorBoard(log_dir="tmp/log", batch_size=64, histogram_freq=1, write_graph=True, write_images=False, write_grads=True)

    # 设置并执行Transfer Learning迁移训练
    print("Start to Transfer Learning...")
    setup_to_transfer_learning(model, base_model)
    history_transferTraining = model.fit_generator(generator        = train_data_generator,  # 训练数据生成器
                                                   steps_per_epoch  = step_per_epoch,        # 一个epoch中的训练次数，这个训练次数可以理解为训练数据的个数，也即遍历学习
                                                   epochs           = epoch,                 # 迭代的轮数
                                                   validation_data  = val_data_generator,    # 训练测试数据生成器
                                                   validation_steps = val_step,                     # 指定训练测试数据集的生成器返回次数
                                                   class_weight     = "auto")
    model.save(save_file_name_TransferTraining)
    print("Saved transfer training model : {0}".format(save_file_name_TransferTraining))
    
    # 设置并执行Fine Tune迁移训练
    print("Start to Fine Tune...")
    setup_to_fine_tune(model, base_model)
    history_fineTune = model.fit_generator(generator        = train_data_generator,
                                           steps_per_epoch  = step_per_epoch,
                                           epochs           = epoch,
                                           validation_data  = val_data_generator,
                                           validation_steps = val_step,
                                           class_weight     = "auto")

    model.save(save_file_name_fileTune)
    print("Saved fine tune model : {0}".format(save_file_name_fileTune))

    plot_training(history_transferTraining)
    plot_training(history_fineTune)

    #print("Output transfer learning history...")
    # 输出迁移训练的精度的分布图
    #plt.plot(history_transferTraining.history["acc"], "o-", lable="accuracy")
    #plt.plot(history_transferTraining.history["val_acc"], "o-", label="val_acc")
    #plt.title("Transfer learning model accuracy")
    #plt.xlabel("epoch")
    #plt.ylabel("accuracy")
    #plt.legend(loc="lower right")
    #plt.show()
    # 输出迁移训练的损失的分布图
    #plt.plot(history_transferTraining.history["loss"], "o-", label="loss")
    #plt.plot(history_transferTraining.history["val_loss"], "o-", label="val_loss")
    #plt.title("Transfer learning model loss")
    #plt.xlabel("epoch")
    #plt.ylabel("loss")
    #plt.legend(loc="lower right")
    #plt.show()

    #print("Output fine tune history...")
    # 输出迁移训练微调的精度的分布图
    #plt.plot(history_ft.history["acc"], "o-", lable="accuracy")
    #plt.plot(history_ft.history["val_acc"], "o-", label="val_acc")
    #plt.title("Fine time model accuracy")
    #plt.xlabel("epoch")
    #plt.ylabel("accuracy")
    #plt.legend(loc="lower right")
    #plt.show()
    # 输出迁移训练微调的损失的分布图
    #plt.plot(history_ft.history["loss"], "o-", label="loss")
    #plt.plot(history_ft.history["val_loss"], "o-", label="val_loss")
    #plt.title("Fine tune model loss")
    #plt.xlabel("epoch")
    #plt.ylabel("loss")
    #plt.legend(loc="lower right")
    #plt.show()

if __name__ == '__main__':
    main()