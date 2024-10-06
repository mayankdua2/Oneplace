import os
import json
import argparse

def read_json_file(file_path):
    my_json_data = {}
    if not check_file(file_path):
        return my_json_data
    with open(file_path, "r", encoding="utf-8") as json_file:
        my_json_data = json.load(json_file)
    return my_json_data

def check_file(pfn):
    if not os.path.exists(pfn):
        return False
    file_size = os.path.getsize(pfn)
    return file_size > 0

def get_json(store_name):
    directory_name = "C:\\Users\\IUD\\OneDrive\\Documents\\StylebyVida\\Dalleex.py"
    primary_json_path = os.path.join(directory_name, "storename", store_name, "assets", "css.json")
    alternative_json_path = os.path.join(directory_name, "storename", "default-css.json")
    
    json_data = read_json_file(primary_json_path)
    if not json_data:
        json_data = read_json_file(alternative_json_path)
    
    print(json_data)

if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Get JSON data for a specific store name.')
    parser.add_argument('--store_name', required=True, help='The name of the store.')

    # Parse the arguments
    args = parser.parse_args()

    # Call the get_json function with the provided store name
    get_json(args.store_name)
