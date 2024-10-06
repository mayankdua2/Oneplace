import ollama
from ollama import generate
from PIL import Image
from io import BytesIO

# Function to process a single image and generate a description
def process_single_image(image_path):
    with Image.open(image_path) as img:
        with BytesIO() as buffer:
            img.save(buffer, format='PNG')
            image_bytes = buffer.getvalue()

    # Generate a description of the image
    full_response = ''
    for response in generate(
            model='llava',
            prompt="""Describe the Garment Very shortly,To the point and in simple language. Description should only focus on the Garment's Sleeve length , how sleeves are Used,To which body part bottom hem of top is reaching and Dress design(Should Only focus on Garments not on models wearing tha)""",
            images=[image_bytes],
            stream=True):
        full_response += response['response']

    return full_response

# Example usage
image_path = '/content/Black-Chase-Shorts-CAVA-athleisure-1686832555830.webp'
description = process_single_image(image_path)
print("Description:", description)
