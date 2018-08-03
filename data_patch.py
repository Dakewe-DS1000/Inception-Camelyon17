"""
Patch all whole slide data to source data for Inception

Index:
1- data_patch
2- convert_data.py
3- train_v3.py
4- predict.py
"""


import cv2
import numpy as np
from os import listdir
from xml.etree.ElementTree import parse
import openslide

file_path_slide       = "F:/ai_data/camelyon17/training/slides/"      # 原始WSI数据路径
file_path_annotations = "F:/ai_data/camelyon17/training/annotations/" # 标记XML数据路径

file_path_save_normal      = "F:/ai_data/camelyon17/training/patch/normal/"      # 保存正常Tile图像数据路径
file_path_save_background  = "F:/ai_data/camelyon17/training/patch/background/"  # 保存背景Tile图像数据路径
file_path_save_tumor       = "F:/ai_data/camelyon17/training/patch/tumor/"       # 保存肿瘤Tile图像数据路径
file_path_save_tumor_mask  = "F:/ai_data/camelyon17/training/patch/tumor_mask/"  # 保存肿瘤Mask图像数据路径
file_path_save_normal_mask = "F:/ai_data/camelyon17/training/patch/normal_mask/" # 保存正常Mask图像数据路径
file_path_save_tmp         = "F:/ai_data/camelyon17/training/patch/tmp/"         # 保存其他的图像数据路径

tile_width  = 288 # 切割Tile图像大小 width
tile_height = 288 # 切割Tile图像大小 height

down_sample_level = 4 # 下采样水平
down_sample_factor = 2 ** down_sample_level # 图像下采样因子
white_pixel_rate = 0.4 # 白色Mask在局部图像中的比例，大于这个比例的，则被视为病变图像

# 根据XML文件中的标记坐标，画出下采样之后的Mask图像
# contours       : XML文件中读取出来的标记点坐标
# wsi_downsample : 下采样之后的图像
# return         : Mask图像，大小与下采样之后的图像相同
def make_mask_tumor(wsi_downsample, contours):
    wsi_empty = np.zeros(wsi_downsample[:2])
    wsi_empty = wsi_empty.astype(np.uint8)
    dst_contours = []
    for idx1, contour in enumerate(contours):
        length = len(contour)
        if length==0: continue
        area = cv2.contourArea(contour)        
        dst_contours.append(contour)  

    cv2.drawContours(wsi_empty, dst_contours, -1, 255, -1)
    return wsi_empty

def make_mask_normal(slideImage_origin) :
    downsample_width = slideImage_origin.level_dimensions[down_sample_level][0]
    downsample_height = slideImage_origin.level_dimensions[down_sample_level][1]
    slideImage_downsample = get_slide_region(slideImage_origin, 0, 0, down_sample_level, downsample_width, downsample_height)

    slideImage_downsample_gray = cv2.cvtColor(slideImage_downsample, cv2.COLOR_BGR2GRAY)
    slideImage_downsample_blur = cv2.GaussianBlur(slideImage_downsample_gray, (5, 5), 0)
    _, slideImage_downsample_binary = cv2.threshold(slideImage_downsample_blur, 0, 255, cv2.THRESH_OTSU)

    return slideImage_downsample_binary

# 读取XML文件中的标记坐标
# file_path_xml : XML文件路径及文件名
# factor        : 下采样因子
# return        : 所有标记点坐标，一个标记区域为一组
# 注意长度坐标数组长度为零的情况
def find_contours_of_xml(file_path_xml, factor):
    list_blob = []
    tree = parse(file_path_xml)

    for parent in tree.getiterator():
        for index_1, child_1 in enumerate(parent):
            for index_2, child_2 in enumerate(child_1):
                for index_3, child_3 in enumerate(child_2):
                    list_point = []
                    for index_4, child_4 in enumerate(child_3):
                        p_x = float(child_4.attrib['X'])
                        p_y = float(child_4.attrib['Y'])
                        p_x = p_x / factor
                        p_y = p_y / factor
                        list_point.append([p_x, p_y])
                    if len(list_point) >= 0:
                        list_blob.append(list_point)

    contours = []

    for list_point in list_blob:
        list_point_int = [[[int(round(point[0])), int(round(point[1]))]] \
                            for point in list_point]
        contour = np.array(list_point_int, dtype=np.int32)
        contours.append(contour)
    return contours

# 判断XML文件与Slide图像数据文件是否一致
# slide_file_name    : WSI图像路径及文件名
# file_name_list_xml : XML标记文件名列表
# return             : 遍历XML标记文件名列表，如果Slide与XML文件名一致则返回True，如果不一致则返回Fasle
def is_tumor_slide(slide_file_name, file_name_list_xml):
    slide_file_name = slide_file_name.split('.')[0]
    
    for i, file_name_xml in enumerate(file_name_list_xml):
        file_name_xml = file_name_xml.split('.')[0]
        if slide_file_name == file_name_xml:
            return True 

    return False

# 获得与Slide图像数据文件一致的XML文件名
# slide_file_name    : WSI图像路径及文件名
# file_name_list_xml : XML标记文件名列表
# return             : 遍历XML标记文件名列表，如果Slide与XML文件名一致则返回XML文件名，
#                      如果不一致则返回字符串“no_xml_file”
def get_tumor_xml_fileName(slide_file_name, file_name_list_xml):
    slide_file_name = slide_file_name.split('.')[0]
    
    for i, file_name_xml in enumerate(file_name_list_xml):
        file_name_xml = file_name_xml.split('.')[0]
        if slide_file_name == file_name_xml:
            return file_name_xml + ".xml"

    return "no_xml_file"

# 获取slide和xml的文件名列表
# slide_file_path : slide图像数据文件路径
# xml_file_path   : xml标记数据文件路径
# return          : slide图像数据文件路径中所有的文件名列表
#                   xml标记数据文件路径中所有的文件名列表
def get_file_name_list(slide_file_path, xml_file_path):
    list_file_name_slide = [f for f in listdir(slide_file_path)]
    list_file_name_xml   = [f for f in listdir(xml_file_path)]
    return list_file_name_slide, list_file_name_xml

# 获取slide图像的尺寸
# slide : wsi图像，用OpenSlide打开
# return : wsi图像的宽和高
def get_slide_image_size(slide):
    width = slide.dimensions[0]
    height = slide.dimensions[1]
    return width, height

# 获取Slide图像中的局部图像数据到OpenCV图像中
# slide : OpenSlide读取的Slide图像数据
# x     : 局部图像的位置坐标，横坐标
# y     : 局部图像的位置坐标，纵坐标
# level : Slide图像的level
# width : 局部图像的大小，宽度
# height: 局部图像的大小，高度
# return: OpenCV图像数据 RGB-3-Channel
def get_slide_region(slide, x, y, level, width, height):
    return cv2.cvtColor(np.array(slide.read_region((x, y), 
                                 level, 
                                 (width, height))), 
                        cv2.COLOR_RGBA2BGR)

def extract_all_slide_image(slide_file_name_list, save_file_path):
    for i, file_name_slide in enumerate(slide_file_name_list):
        slideImage = openslide.OpenSlide(file_path_slide + file_name_slide);
        origin_width, origin_height = get_slide_image_size(slideImage)

        level_width = slideImage.level_dimensions[down_sample_level][0]
        level_height = slideImage.level_dimensions[down_sample_level][1]

        cvSlideImage = get_slide_region(slideImage, 0, 0, down_sample_level, level_width, level_height)

        save_file_name = save_file_path + file_name_slide.split(".")[0] + ".jpg"
        cv2.imwrite(save_file_name, cvSlideImage)

        print("Slide Down Sample Saved to ==> {0}".format(save_file_name))

def patch_normal_data(slide_file_name_list, xml_file_name_list):
    
    for i, file_name_slide in enumerate(slide_file_name_list):
        
        if is_tumor_slide(file_name_slide, xml_file_name_list) == False:
            print("Processing Normal data : {0} / {1} ==>{2}".format(i, len(slide_file_name_list), file_name_slide))

            slideImage_origin = openslide.OpenSlide(file_path_slide + file_name_slide)
            origin_width = slideImage_origin.dimensions[0]
            origin_height = slideImage_origin.dimensions[1]

            level_width = slideImage_origin.level_dimensions[down_sample_level][0]
            level_height = slideImage_origin.level_dimensions[down_sample_level][1]
            slideImage_downsample = get_slide_region(slideImage_origin, 0, 0, down_sample_level, level_width, level_height)
            slideImage_downsample_mask = make_mask_normal(slideImage_origin)

            save_file_name = file_path_save_tmp + file_name_slide.split(".")[0] + "_normal_mask.jpg"
            cv2.imwrite(save_file_name, slideImage_downsample_mask)
            save_file_name = file_path_save_tmp + file_name_slide.split(".")[0] + "_normal.jpg"
            cv2.imwrite(save_file_name, slideImage_downsample)
            print("save mask file : {0}".format(save_file_name))

            step_x = int(tile_width / down_sample_factor / 2)
            step_y = int(tile_height /  down_sample_factor / 2)
            for x in range(0, level_width - step_x * 2, step_x) : 
                for y in range(0, level_height - step_y * 2, step_y) :
                    mask_pixel = slideImage_downsample_mask[y, x]

                    if mask_pixel == 255 :
                        origin_x = x * down_sample_factor
                        origin_y = y * down_sample_factor
# 下采样Mask图像区域提取的开始坐标
                        level_x_start = x
                        level_y_start = y
# 下采样Mask图像区域提取的终止坐标
                        level_x_end = x + step_x * 2
                        level_y_end = y + step_y * 2

                        tileImage_downsample_mask = slideImage_downsample_mask[level_y_start:level_y_end, level_x_start:level_x_end]
                        tileImage_mask = cv2.resize(tileImage_downsample_mask, (tile_width, tile_height), interpolation=cv2.INTER_CUBIC)
                        _, tileImage_mask = cv2.threshold(tileImage_mask, 0, 255, cv2.THRESH_OTSU)

                        tileImage_origin = get_slide_region(slideImage_origin, origin_x, origin_y, 0, tile_width, tile_height)

                        pixel_mean_val = 0
                        red_mean_val   = 0
                        green_mean_val = 0
                        blue_mean_val  = 0
                        for i in range(0, tile_height, 1) :
                            for j in range(0, tile_width, 1):
                                pixel_mean_val = pixel_mean_val + tileImage_mask[i, j]
                                red_mean_val   = red_mean_val   + tileImage_origin[i, j, 2]
                                green_mean_val = green_mean_val + tileImage_origin[i, j, 1]
                                blue_mean_val  = green_mean_val + tileImage_origin[i, j, 0]

                        pixel_mean_val = float(pixel_mean_val) / float(tile_height * tile_width)
                        red_mean_val   = float(red_mean_val)   / float(tile_height * tile_width)
                        green_mean_val = float(green_mean_val) / float(tile_height * tile_width)
                        blue_mean_val  = float(blue_mean_val)  / float(tile_height * tile_width)

                        print("{0}, {1} :: {2}, {3}, {4}, {5}".format(origin_x, origin_y, pixel_mean_val, red_mean_val, green_mean_val, blue_mean_val))
                        if  pixel_mean_val < white_pixel_rate : 
                        #    print("Black Background")
                            continue
                        elif red_mean_val < 80  and green_mean_val < 80  and blue_mean_val < 80 :
                            continue
                        if   red_mean_val > 230 and green_mean_val > 230 and blue_mean_val > 230 :
                        #    print("Light Background")
                            continue        

                        save_file_name = file_path_save_normal + file_name_slide.split(".")[0] + "_" + str(origin_x) + "_" + str(origin_y) + "_" + str(int(red_mean_val)) + "-" + str(int(green_mean_val)) + "-" + str(int(blue_mean_val)) + ".jpg"
                        cv2.imwrite(save_file_name, tileImage_origin)
                        save_file_name = file_path_save_normal_mask + file_name_slide.split(".")[0] + "_" + str(origin_x) + "_" + str(origin_y) + "_mask.jpg"
                        cv2.imwrite(save_file_name, tileImage_mask)                    

            

# 获取tumor的数据Patch
# 根据XML文件的Mask标记，获取所有tumor的局部图像，局部图像大小 tile_width x tile_height
# slide_file_name_list : WSI图像文件名列表
# xml_file_name_list   : Mask标记的XML文件名列表
def patch_tumor_data(slide_file_name_list, xml_file_name_list):
# 开始遍历WSI图像数据
    for i, file_name_slide in enumerate(slide_file_name_list):
        
# 如果是肿瘤病变的图像数据
        if is_tumor_slide(file_name_slide, xml_file_name_list) == True:
            print("Processing : {0} / {1}".format(i, len(slide_file_name_list)))
# 读取一幅WSI图像文件
            slideImage = openslide.OpenSlide(file_path_slide + file_name_slide)
            origin_width = slideImage.dimensions[0]
            origin_height = slideImage.dimensions[1]

# 原始图像下采样
            level_width, level_height = slideImage.level_dimensions[down_sample_level]
            slideImage_downsample = slideImage.read_region((0, 0), down_sample_level, (level_width, level_height))
            cvSlideImage_downsample = cv2.cvtColor(np.array(slideImage_downsample), cv2.COLOR_RGBA2BGR)

# 获取XML标记信息，绘制下采样Mask图像
            file_name_xml = get_tumor_xml_fileName(file_name_slide, list_file_name_xml) 
            contours = find_contours_of_xml(file_path_annotations + file_name_xml, down_sample_factor)
            maskImage_downsample = make_mask_tumor(cvSlideImage_downsample.shape[0:2], contours)

# 在Mask图像中搜索白色Mask区域
            step_x = int(tile_width/down_sample_factor/2)
            step_y = int(tile_height/down_sample_factor/2)
            for x in range(0, level_width - step_x * 2, step_x):
                for y in range(0, level_height - step_y * 2, step_y):
# 下采样Mask图像区域提取的开始坐标
                    level_x_start = x
                    level_y_start = y
# 下采样Mask图像区域提取的终止坐标
                    level_x_end = x + step_x * 2
                    level_y_end = y + step_y * 2
# 原始WSI图像区域提取坐标
                    origin_x = x * down_sample_factor
                    origin_y = y * down_sample_factor
# 获取当前下采样Mask图像像素数值，白色则为病变标记
                    pixel = maskImage_downsample[y, x]

                    if pixel == 255 :
                        #print("{0}, {1}".format(origin_x, origin_y))
# 根据提取坐标，提取病变区域的WSI图像
                        tileImage_origin = slideImage.read_region((origin_x, origin_y), 0, (tile_width, tile_height))
# 数据从OpenSlide转换为可存储和操作的OpenCV图像数据
                        cvTileImage_origin = cv2.cvtColor(np.array(tileImage_origin), cv2.COLOR_RGBA2BGR)
# 根据下采样提取坐标，提取病变区域的Mask图像
                        tileImage_mask = maskImage_downsample[level_y_start:level_y_end, level_x_start:level_x_end]
# 将Mask图像重新变换为原始尺寸 tile_width x tile_height
                        resizedTileImage_mask = cv2.resize(tileImage_mask, (tile_width, tile_height), interpolation=cv2.INTER_CUBIC)
                        _, resizedTileImage_mask = cv2.threshold(resizedTileImage_mask, 0, 255, cv2.THRESH_OTSU)

# Mask中的白色区域小于一定比例，不作为病变进行识别
                        pxCounter = 0
                        for _y in range(0, tile_height, 1):
                            for _x in range(0, tile_width, 1):
                                if resizedTileImage_mask[_y, _x] == 255:
                                    pxCounter = pxCounter + 1
                        pxRate = float(pxCounter) / float(tile_height * tile_width)
                        if pxRate < white_pixel_rate: continue
                        else : print("White Rate : {0}".format(pxRate))
# 保存病变局域图像以及Mask局域图像
                        save_file_name = file_path_save_tumor + file_name_slide.split(".")[0] + "_" + str(origin_x) + "_" + str(origin_y) + ".jpg"
                        cv2.imwrite(save_file_name, cvTileImage_origin)                    
                        save_file_name = file_path_save_tumor_mask + file_name_slide.split(".")[0] + "_" + str(origin_x) + "_" + str(origin_y) + "_mask.jpg"
                        cv2.imwrite(save_file_name, resizedTileImage_mask)
                    
                    else:
                        tileImage_origin = slideImage.read_region((origin_x, origin_y), 0, (tile_width, tile_height))
                        cvTileImage_origin = cv2.cvtColor(np.array(tileImage_origin), cv2.COLOR_BGRA2BGR)
                        tileImage_mask = maskImage_downsample[level_y_start:level_y_end, level_x_start:level_x_end]
                        resizedTileImage_mask = cv2.resize(tileImage_mask, (tile_width, tile_height), interpolation=cv2.INTER_CUBIC)
                        _, resizedTileImage_mask = cv2.threshold(resizedTileImage_mask, 0, 255, cv2.THRESH_OTSU)
                        
                        pxCounter = 0
                        for _y in range(0, tile_height, 1):
                            for _x in range(0, tile_width, 1):
                                if resizedTileImage_mask[_y, _x] == 255:
                                    pxCounter = pxCounter + 1
                        pxRate = float(pxCounter) / float(tile_height * tile_width)
                        if pxRate > white_pixel_rate: continue
                        #else : print("White Rate : {0}".format(pxRate))
                        
                        red_counter = 0
                        green_counter = 0
                        blue_counter = 0

                        for _y in range(0, tile_height, 1):
                            for _x in range(0, tile_width, 1):
                                blue_counter = blue_counter + cvTileImage_origin[_y, _x, 0]
                                green_counter = green_counter + cvTileImage_origin[_y, _x, 1]
                                red_counter = red_counter + cvTileImage_origin[_y, _x, 2]
                        blue_counter = float(blue_counter) / float(tile_height * tile_width)
                        green_counter = float(green_counter) / float(tile_height * tile_width)
                        red_counter = float(red_counter) / float(tile_height * tile_width)

                        if y % 1000 == 0 :
                            print("x = {0}, y = {1}".format(x, y))                      

                        if blue_counter < 100 and green_counter < 100 and red_counter < 100 :
                            print("Black BackGround : {0} ==> {1}, {2}".format(file_name_slide.split(".")[0], origin_x, origin_y))
                            continue
                        elif blue_counter < 120 and green_counter > 120 and red_counter > 120 :
                            print("{0}, {1}, {2}".format(red_counter, green_counter, blue_counter))
                            save_file_name = file_path_save_normal + file_name_slide.split(".")[0] + "_" + str(origin_x) + "_" + str(origin_y) + ".jpg"
                            cv2.imwrite(save_file_name, cvTileImage_origin)                    
                            #save_file_name = file_path_save_normal_mask + file_name_slide.split(".")[0] + "_" + str(origin_x) + "_" + str(origin_y) + "_mask.jpg"
                            #cv2.imwrite(save_file_name, resizedTileImage_mask)
                        else :
                            save_file_name = file_path_save_background + file_name_slide.split(".")[0] + "_" + str(origin_x) + "_" + str(origin_y) + "_background.jpg"
                            cv2.imwrite(save_file_name, cvTileImage_origin)

                        
# 暂存下采样图像以及下采样Mask图像
            #saveFileName = "F:/ai_data/camelyon17/training/patch/tmp/" + str(i) + "_mask.jpg"
            #cv2.imwrite(saveFileName, maskImage_downsample)
            #saveFileName = "F:/ai_data/camelyon17/training/patch/tmp/" + str(i) + "_origin.jpg"
            #cv2.imwrite(saveFileName, cvSlideImage_downsample)

if __name__ == "__main__":
# 从数据库中读取WSI图像和XML标记的所有文件名
    list_file_name_slide, list_file_name_xml = get_file_name_list(file_path_slide, file_path_annotations)
    
    #extract_all_slide_image(list_file_name_slide, file_path_save_tmp)
    #patch_tumor_data(list_file_name_slide, list_file_name_xml)
    patch_normal_data(list_file_name_slide, list_file_name_xml)