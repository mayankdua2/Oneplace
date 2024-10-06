import os
import json
import csv
from scipy.spatial import distance

SBV_HOME = os.environ.get("SBV_HOME", "/content/drive/MyDrive/workspace/")

def check_file(pfn):
    if not os.path.exists(pfn):
        return False
    file_size = os.path.getsize(pfn)
    return file_size > 0


def eulidean_dist(v1, v2):
    ed = distance.euclidean(v1, v2)
    rd_round = round(ed, 4)
    return rd_round

def check_path_exists(path, tag_line):
    if  not os.path.exists(path):
        print(f"The {tag_line} does not exist in product path :- {path}")
        return False
    return True

def get_proj_folder():
    template_folder = os.path.join(SBV_HOME, "Fashion-LLM")
    return template_folder

def get_color_folder():
    template_folder = os.path.join(SBV_HOME, "Fashion-LLM", "data", "colors")
    return template_folder

def get_intent_folder(store_name):
    data_folder = os.path.join(SBV_HOME, "StyleBoards", "client", "src")
    return data_folder
    
def get_stats_folder(store_name):
    stats_folder = os.path.join(SBV_HOME, "assets", "new-dress-tags", store_name, "stats")
    return stats_folder

def get_store_folder(store_name):
    data_folder = os.path.join(SBV_HOME, "assets", "new-dress-tags", store_name)
    return data_folder

def get_lookboards_folder():
    template_folder = os.path.join(SBV_HOME, "assets", "look-boards")
    return template_folder


def get_templ_folder():
    template_folder = os.path.join(SBV_HOME, "Fashion-LLM", "data", "tag-templates")
    return template_folder

def get_intent1_folder():
    template_folder = os.path.join(SBV_HOME, "Fashion-LLM", "data", "intent")
    return template_folder

def get_misc_folder():
    template_folder = os.path.join(SBV_HOME, "Fashion-LLM", "data", "misc")
    return template_folder

def get_parent_directory_and_current_directory(image_path_filename):
    filename_with_jpg = os.path.basename(image_path_filename)
    filename = os.path.splitext(filename_with_jpg)[0]
    my_current_dir = os.path.dirname(image_path_filename)
    my_parent_dir = os.path.join(my_current_dir, filename)
    return my_parent_dir, my_current_dir

def get_data_folder(store_name):
    data_folder = os.path.join(SBV_HOME, "assets", "new-dress-tags", store_name, "data")
    return data_folder


def create_csv_file(file_path):
    with open(file_path, 'w', newline='') as csv_file:
         csv_writer = csv.writer(csv_file)

def create_file(file_path):
    try:
        with open(file_path, 'w') as file:
            pass
    except Exception as e:
        print(f"An error occurred while creating the file: {e}")

def write_to_json_file(output_json_file, json_data):
    # print("output_json_file", output_json_file)
    dest_folder = os.path.dirname(output_json_file)
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
    with open(output_json_file, "w") as output_file:
        json.dump(json_data, output_file, indent=4)
        

def print_readable_context(json_data):
    chat_history_list = []
    key = ""
    for message in json_data['chat_history']:
        message_type = message.type
        if message_type == "human":
            key  = "HumanMessage" 
        if message_type == "ai":
            key =  "AIMessage"      
        chat_history_list.append({key: message.__dict__})
    json_data['chat_history'] = chat_history_list
    formatted_data = json.dumps(json_data, indent=2)
    print(formatted_data)     

def write_to_text_file(output_text_file, text_data):
    with open(output_text_file, 'w') as file:
         file.write(text_data + '\n')

def write_row_to_csv(output_csv_file, row_data):
    with open(output_csv_file, 'a', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(row_data)

def read_json_file(file_path):
    # print("read_json_file() - file_path", file_path)
    my_json_data = {}
    if not check_file(file_path):
        return my_json_data
    with open(file_path, "r", encoding="utf-8") as json_file:
        my_json_data = json.load(json_file)
    return my_json_data
