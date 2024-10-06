import os
import sys
import json
import random
# import pinecone
from pinecone import Pinecone, PodSpec, ServerlessSpec
import torch
import datetime
from torch import nn
from torch.nn.functional import normalize
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from transformers import AutoTokenizer, AutoModel
from mypy.sbv_lib import utils as sbv_utils

SBV_HOME = sbv_utils.SBV_HOME

model_ckpt = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
model = AutoModel.from_pretrained(model_ckpt)


def initialize_pinecone(my_index_name, pinecone_instance):
    vector_dim = 384
    # environment = "gcp-starter"

    cloud = os.getenv('PINECONE_CLOUD', 'aws')
    region = os.getenv('PINECONE_REGION', 'us-east-1')
    spec = ServerlessSpec(cloud=cloud, region=region)

    if my_index_name not in pinecone_instance.list_indexes().names():
        print("Creating new Index ...")
        # pinecone_instance.create_index(
        #     name=my_index_name, dimension=vector_dim, metric="cosine", spec = PodSpec(environment=environment)
        # )
        pinecone_instance.create_index(
            name=my_index_name, dimension=vector_dim, metric="cosine", spec = spec
        )
    else:
        print("Index already exists...")
    my_index = pinecone_instance.Index(my_index_name)
    # index_desc = my_index.describe_index_stats()
    # print("index_desc - ", index_desc)
    return my_index


# def save_max_image_id(store_name, max_image_id):
#     index_stats_pfn = sbv_utils.get_intent_folder(store_name) + "/index_stats.json"
#     with open(index_stats_pfn, "w") as f:
#         json.dump({"max_image_id": max_image_id}, f)


# def read_max_image_id(store_name):
#     index_stats_pfn = sbv_utils.get_store_folder(store_name) + "/index_stats.json"
#     index_stats = sbv_utils.read_json_file(index_stats_pfn)
#     return index_stats["max_image_id"]


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


def get_single_text_embedding(text):
    encoded_input = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        model_output = model(**encoded_input)
    sentence_embeddings = mean_pooling(model_output, encoded_input["attention_mask"])

    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
    return sentence_embeddings


def ingest_image_and_data(embedding, file_name, folder_name, text, index, my_index):
    final_metadata = []
    final_metadata.append(
        {"image-folder-name": folder_name, "image-name": file_name, "text": text}
    )
    image_IDs = str(index)
    # Create the single list of dictionary format to insert
    data_to_upsert = list(zip(image_IDs, embedding.tolist(), final_metadata))
    # Upload the final data
    my_index.upsert(vectors=data_to_upsert)


def retrive_text_for_text_query(text_query, number_of_top_images, my_index):
    query_embedding = get_single_text_embedding(text_query).tolist()
    retrieved_dict = my_index.query(
        query_embedding, top_k=number_of_top_images, include_metadata=True
    )
    return retrieved_dict


def read_product_description(descriptive_file_path):
    try:
        with open(descriptive_file_path, "r") as file:
            file_content = file.read()
        return file_content
    except Exception as e:
        print(e)
        return ""


def store_text_line_by_line(descriptive_file_path):
    lines = []
    with open(descriptive_file_path, "r") as file:
        file_content = file.read()
    text_lines = file_content.split("\n")
    for line in text_lines:
        lines.append(line)
    return lines


def upsert_index(my_index, count, image_IDs, embedding_list, final_metadata):
    print("\nStarting upsert for next batch of", str(len(image_IDs)), "at count =", count)
    my_tuples = zip(image_IDs, embedding_list, final_metadata)
    data_to_upsert = [{'id': my_id, 'values': my_embedding.tolist()[0], 'metadata':my_metadata} for my_id, my_embedding, my_metadata in my_tuples]
    my_index.upsert(vectors=data_to_upsert)
    print("\nCompleted upsert for next batch of", str(len(image_IDs)), "at count =", count)


def remove_data_for_store(store_name, my_index):
    count = 0
    store_id = sbv_utils.get_store_id(store_name)
    print("store_id :- ", store_id, end="\n")
    max_image_id = read_max_image_id(store_name)
    ids_to_delete = []
    rows_deleted = 0
    for count in range(0, max_image_id, 1):
        my_image_id = store_id * 10000 + count
        ids_to_delete.append(str(my_image_id))
        if (count % 100) == 0 and count > 0:
            my_index.delete(ids=ids_to_delete)
            rows_deleted += len(ids_to_delete)
            ids_to_delete = []
    if len(ids_to_delete) > 0:
        my_index.delete(ids=ids_to_delete)
        rows_deleted += len(ids_to_delete)
    print("\n rows_deleted", rows_deleted)
    print("\n max_image_id", max_image_id)

def inserting_embeddings_for_all_text(store_name, my_index):
    count = 0
    store_id = sbv_utils.get_store_id(store_name)
    print("store_id :- ", store_id, end="\n")                  
    data_folder = sbv_utils.get_intent1_folder() 
    image_IDs, embedding_list, final_metadata = [], [], []
    prod_desc_list_not_found = 0
    for next_product in os.listdir(data_folder):
        if  next_product.endswith(".json"):
            next_product_folder = os.path.join(data_folder, next_product)
            json_content = sbv_utils.read_json_file(next_product_folder)
            for parent_key, parent_value in json_content.items():
                for key, value in parent_value.items():
                    product_desc_list = value  
                    if product_desc_list is None or product_desc_list == "":
                        prod_desc_list_not_found += 1
                        print("\nproduct_desc_list empty, Skipping :- ", next_product, end="\n")
                        continue
                    for product_desc in product_desc_list:
                        print("parent_key", parent_key, "key", key) 
                        print("Processing the product :- ", product_desc, count, "                           ", end="\r")
                        embedding = get_single_text_embedding(product_desc)
                        my_image_id = store_id * 10000 + count
                        image_IDs.append(str(my_image_id))
                        embedding_list.append(embedding.to(dtype=torch.float32))
                        final_metadata.append({'intent-id': key, "parent_tag":  parent_key, "sid": store_id})
                        count += 1
                        if ((count % 100) == 0 and count > 0):
                           upsert_index(my_index, count, image_IDs, embedding_list, final_metadata)
                           image_IDs, embedding_list, final_metadata = [], [], []
    if len(image_IDs) > 0:
        upsert_index(my_index, count, image_IDs, embedding_list, final_metadata)
    # save_max_image_id(store_name, count)
    print("\n prod_desc_list_not_found", prod_desc_list_not_found)
    print("\n products inserted", count)



def delete_index(my_index_name, pc):
    pc.delete_index(my_index_name)
    print(f"Pincone {my_index_name} index deleted .")


def main(store_name, my_index):
    inserting_embeddings_for_all_text(store_name, my_index)


if __name__ == "__main__":
    print("len(sys.argv) = ", len(sys.argv))
    if len(sys.argv) < 5:
        store_name = "intent"
        action = "c"
    else:
        store_name = sys.argv[sys.argv.index("--store_name") + 1]
        action = sys.argv[sys.argv.index("--action") + 1]
    print("=====================================================")
    print(datetime.datetime.now())
    print("store_name=", store_name)
    print("action=", action)
    print("=====================================================")
    # my_api_key = "dbdb2f60-8d1c-4043-a78d-0d47a84fe18b"  # suyashsingh176@gmail.com
    #  my_api_key = "01fe140d-8534-4f62-9e2b-00c934a771b7"  # a.puneet.dr@gmail.com
    #  my_api_key = "1606562b-fd81-4681-b683-177b6682dd3f"  # apuneetindia@gmail.com
    # my_api_key = "ccbd00be-401e-4beb-8873-42d8c29280be"  #suyashsingh767@gmail.com
    # my_api_key = "d41bb6ea-605f-43d4-a997-0dcd4fcedd98"  # suyashsingh968@gmail.com
    my_api_key = "763c0ba7-6807-457d-a793-4f675bb17241"  # mayankdua0369@gmail.com
    if "SBV_PINECONE_API_KEY" in os.environ:
        my_api_key = os.environ.get("SBV_PINECONE_API_KEY")
        print("PINECONE_API_KEY found in env: ", my_api_key)
    print("my_api_key", my_api_key)
    os.environ.get("SBV_HOME")
    pc = Pinecone(api_key=my_api_key)
    SBV_HOME = os.environ.get("SBV_HOME")
    my_index_name = "clip-image-search"
    my_index = initialize_pinecone(my_index_name, pc)
    if action == "d":  # Delete Index
        delete_index(my_index_name, pc)
    elif action == "a":  # Add data for a Store
        main(store_name, my_index)
    elif action == "c":  # Clean Data for a Store
        remove_data_for_store(store_name, my_index)
    else:
        print("Unknown action - ", action)
