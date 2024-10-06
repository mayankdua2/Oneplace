import os
import re
import sys
import json
import datetime
import numpy as np
from PIL import Image
import rembg
from rembg import remove
import cv2
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from webcolors import rgb_to_name
from . import CLIP as clip
import torch
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import cosine_distances
import matplotlib.patches as patches
import matplotlib.colors as mcolors
from matplotlib.patches import Circle
from mypy.sbv_lib import utils as sbv_utils
SBV_HOME = sbv_utils.SBV_HOME

def replace_black_background_with_green(image, color4Trans=(0, 255, 0)):
    print("image.mode = " + image.mode)
    print("image.size = ", image.size)
    width, height = image.size
    new_image = Image.new("RGB", (width, height))
    for x in range(width):
        for y in range(height):
            pixel = image.getpixel((x, y))
            if pixel != (0, 0, 0, 0):
                new_image.putpixel((x, y), pixel)
    return new_image

def remove_background_from_image(image_path):
    input_image = Image.open(image_path)
    output_image = remove(input_image)
    return output_image


def analyze_accept_segment(image, my_segment_List, prompts, parent_directory):
    new_image = image.copy()
    for np_segment, prompt in zip(my_segment_List, prompts):
        new_image_segment = image.copy()
        for i in range(np_segment.shape[0]):
            for j in range(np_segment.shape[1]):
                if np_segment[i][j] >= 0.3:
                   new_image_segment.putpixel((j, i), (0, 255, 0))
                   new_image.putpixel((j, i), (0, 255, 0))
        fn = (prompt.replace(" ", "-")).replace(",", "-")
        extracted_image_path = os.path.join(parent_directory, fn + ".jpg")
        new_image_segment.save(extracted_image_path)
        print(
            f"The {prompt} is completed and the extracted_path is :-",
            extracted_image_path,
        )
    accept_prompt_image_path = os.path.join(parent_directory, "accept_segment.jpg")
    # new_image_segment.save(accept_prompt_image_path)
    new_image.save(accept_prompt_image_path)
    return new_image

def inverted_image_generator(final_image, original_image, image_directory):
    new_image = original_image.copy()
    width, height = new_image.size
    for i in range(width):
        for j in range(height):
            if final_image.getpixel((i, j)) != (0,255,0):
               new_image.putpixel((i, j), (0, 255, 0))
    extracted_image_path = os.path.join(image_directory, "inverted_image.jpg")
    new_image.save(extracted_image_path)          
    return new_image

def analyze_reject_segments(accept_segment_img, image, my_segment_List, prompts, target_area, image_directory):
    final_image = accept_segment_img.copy()
    for np_segment, prompt in zip(my_segment_List, prompts):
        new_image_segment = image.copy()
        for i in range(np_segment.shape[0]):
            for j in range(np_segment.shape[1]):
                if np_segment[i][j] >= 0.3:
                    final_image.putpixel((j, i), image.getpixel((j, i)))
                    new_image_segment.putpixel((j, i), (0, 255, 0))
        fn = (prompt.replace(" ", "-")).replace(",", "-")
        extracted_image_path = os.path.join(image_directory, fn + ".jpg")
        new_image_segment.save(extracted_image_path)
    extracted_image_path = os.path.join(image_directory, target_area+"_segment.jpg")
    final_image.save(extracted_image_path)
    return final_image

def calculate_pixels(image, parent_directory):
    new_image = image.copy()
    pixel_count, total_pixel_count = 0, 0
    width, height = new_image.size
    print("the width is  :-",width, " the height : ", height)
    min_x, max_x, min_y, max_y = 10000, 0, 10000, 0
    total_pixel_count = width * height
    for i in range(width):
        for j in range(height):
            if image.getpixel((i, j)) == (0,255,0):
                    pixel_count += 1
                    min_x = min(min_x, i)
                    max_x = max(max_x, i)
                    min_y = min(min_y, j)
                    max_y = max(max_y, j)
    margin_l, margin_r, margin_t, margin_b = min_x, width - max_x, min_y, height - max_y
    print("left_x: ", min_x, "right_x: ", height - max_x)
    print("top_y: ", min_y, "bottom_y: ",width - max_y)
    return pixel_count, total_pixel_count, margin_l, margin_r, margin_t, margin_b

def extract_pixel_count(image_path_filename, target_area):
    removed_new_image = remove_background_from_image(image_path_filename)
    new_image = replace_black_background_with_green(removed_new_image)
    proj_folder = sbv_utils.get_proj_folder()
    json_file_path = os.path.join(proj_folder, "data", "misc", "image_quality.json")
    tgt_area_obj = sbv_utils.read_json_file(json_file_path)[target_area]
    my_reject_prompts, my_accept_prompts = [], []
    if "reject_prompts" in tgt_area_obj.keys():
        my_reject_prompts = sbv_utils.read_json_file(json_file_path)[target_area][
            "reject_prompts"
        ]
    if "accept_prompts" in tgt_area_obj.keys():
        my_accept_prompts = sbv_utils.read_json_file(json_file_path)[target_area][
            "accept_prompts"
        ]

    if len(my_accept_prompts) == 0:
        return

    filename_with_jpg = os.path.basename(image_path_filename)
    filename = os.path.splitext(filename_with_jpg)[0]
    product_directory = os.path.dirname(image_path_filename)
    image_directory = os.path.join(product_directory, filename)

    margin_l, margin_r, margin_t, margin_b = 0,0,0,0

    if not os.path.exists(image_directory):
        os.makedirs(image_directory)
    np_accept_segment_List, accpet_prompts = clip.segment_image(new_image, my_accept_prompts)
    accept_prompt_image =  analyze_accept_segment(new_image, np_accept_segment_List, accpet_prompts, image_directory)

    if len(my_reject_prompts)>0:
        np_reject_segment_List, reject_prompts = clip.segment_image(new_image, my_reject_prompts)
        final_image =  analyze_reject_segments(accept_prompt_image, new_image, np_reject_segment_List, reject_prompts, target_area, image_directory)
    else:
        final_image = accept_prompt_image
        
    inverted_image = inverted_image_generator(final_image, new_image, image_directory)

    target_pixel_count, total_pixel_count, margin_l, margin_r, margin_t, margin_b = calculate_pixels(final_image, image_directory)
    print("target_pixel_count: ", target_pixel_count)
    print("total_pixel_count: ", total_pixel_count)
    percentage = 100 * target_pixel_count / total_pixel_count
    print("% Target: ", percentage)
    return target_pixel_count, total_pixel_count, margin_l, margin_r, margin_t, margin_b, percentage


if __name__ == "__main__":
    print("len(sys.argv) = ", len(sys.argv))
    if len(sys.argv) < 3:
        image_pfn = "/content/drive/MyDrive/workspace/assets/new-dress-tags/abstract-leaf-print-halter-top/AbstractLeafPrintHalterTop1.jpg"
        image_pfn = "/home/puneet/workspace/assets/new-dress-tags/fablestreet/data/bow-leather-ballerina-heels-brown/bow_leather_ballerina_heels_z1_1_5685a2de-b05b-4a45-a5b2-6ccf74f4ba7a.jpg"
        target_area = "footwear"
    else:
        image_pfn = sys.argv[sys.argv.index("--image_path") + 1]
        target_area = sys.argv[sys.argv.index("--target_area") + 1]
    print("=====================================================")
    print(datetime.datetime.now())
    print("image_path :-", image_pfn)
    print("target_area :-", target_area)
    print("=====================================================")
    # if SBV_HOME.startswith("/content"):
    #    mount_google_drive()
    extract_pixel_count(image_pfn, target_area)
