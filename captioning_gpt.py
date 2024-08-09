import base64
import requests
import openai
import os
import pickle
import numpy as np
import csv
import pandas as pd
import glob
import argparse

# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

parser = argparse.ArgumentParser(description="Process API key and CSV file path.")
parser.add_argument('--api_key', type=str, required=True, help="Your OpenAI API Key.")
parser.add_argument('--csv_file', type=str, default='./caption.csv', help="Path to the output CSV file.")
args = parser.parse_args()

api_key = args.api_key
csv_file = args.csv_file

output_csv = open(csv_file, 'a')
writer = csv.writer(output_csv)

uid = pickle.load(open('filtered_uids1.pkl','rb'))
wrong_or_none_files = []
captions = {}
for kk, u in enumerate(uid):
    image_paths = []
    # insert your image_path
    # in DiffuRank, we send the top-6 views after ranking
    for i in range(6):
        image_paths.append(os.path.join(u, '%d.png'%i))

    base64_images = []
    try:
        for i in image_paths:
            base64_images.append(encode_image(i))
    except:
        wrong_or_none_files.append(u)
        continue
    
    headers = {
      "Content-Type": "application/json",
      "Authorization": f"Bearer {api_key}"
    }
    
    payload = {
      # GPT-4o-mini is much cheaper than GPT-4o, 
      # DiffuRank used GPT-4o
      # "model": "gpt-4o-2024-05-13",
      "model": "gpt-4o-mini-2024-07-18",
      "messages": [
        {
          "role": "user",
          "content": [
            {
              "type": "text",
              "text": "Renderings show different angles of the same set of 3D objects. Concisely describe 3D object (distinct features, objects, structures, material, color, etc) as a caption, not mentioning angles and image related words"
            },
          ]
        }
      ],
      "max_tokens": 300
    }
    for i in range(len(base64_images)):
        payload['messages'][0]['content'].append( {
       "type": "image_url",
       "image_url": {
         "url": f"data:image/jpeg;base64,{base64_images[i]}"
       }
     }
    )
    
    try:
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    except:
        continue
    try:
        r = response.json()
    except:
        continue
    captions[u] = r
    try:
        cur_caption = r['choices'][0]['message']['content']
    except:
        continue
    writer.writerow([u, cur_caption])
    print(kk, u, cur_caption)
    if (kk)% 100 == 0:
        output_csv.flush()
        os.fsync(output_csv.fileno())
output_csv.close()