#do not use programs in this py file

#Delete 2018-07-30
#def patch_normal_data(slide_file_name_list, xml_file_name_list):
step_x = int(tile_width/2)
            step_y = int(tile_height/2)
            for x in range(0, origin_width - tile_width, step_x):
                for y in range(0, origin_height - tile_height, step_y):
                    
                    tileImage = get_slide_region(slideImage, x, y, 0, tile_width, tile_height)

                    red_mean_val   = 0
                    green_mean_val = 0
                    blue_mean_val  = 0
                    for i in range(0, tile_height, 1):
                        for j in range(0, tile_width, 1):
                            red_mean_val   = red_mean_val   + tileImage[i, j, 2]
                            green_mean_val = green_mean_val + tileImage[i, j, 1]
                            blue_mean_val  = blue_mean_val  + tileImage[i, j, 0]
                    red_mean_val   = float(red_mean_val)   / float(tile_width * tile_height)
                    green_mean_val = float(green_mean_val) / float(tile_width * tile_height)
                    blue_mean_val  = float(blue_mean_val)  / float(tile_width * tile_height)

                    if red_mean_val < 100 and green_mean_val < 100 and blue_mean_val < 100 : 
                        # this is the BLACK BACKGROUND !
                        print("black background : {0} ==> {1}, {2} :: ({3}, {4}, {5})".format(file_name_slide, x, y, red_mean_val, green_mean_val, blue_mean_val))
                        save_file_name = file_path_save_background + file_name_slide.split(".")[0] + "_" + str(x) + "_" + str(y) + "_black_ground.jpg"
                        #cv2.imwrite(save_file_name, tileImage)

                    elif red_mean_val > 120 and green_mean_val > 120 and blue_mean_val < 120 :
                        # this is the TISSUE !
                        save_file_name = file_path_save_normal + file_name_slide.split(".")[0] + "_" + str(x) + "_" + str(y) + "_normal.jpg"
                        cv2.imwrite(save_file_name, tileImage)
                        print("Save Normal Tissue Block Image ==> {0} :: ({1}, {2}, {3}".format(save_file_name, red_mean_val, green_mean_val, blue_mean_val))

                    else :
                        # this is the others as BACKGROUND !
                        print("other background : {0} ==> {1}, {2} :: ({3}, {4}, {5}".format(file_name_slide, x, y, red_mean_val, green_mean_val, blue_mean_val))
                        save_file_name = file_path_save_background + file_name_slide.split(".")[0] + "_" + str(x) + "_" + str(y) + "_ground.jpg"
                        #cv2.imwrite(save_file_name, tileImage)