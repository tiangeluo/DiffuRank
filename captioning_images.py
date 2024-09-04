import torch
from PIL import Image
from lavis.models import load_model_and_preprocess
import glob
import pickle as pkl
from tqdm import tqdm
import os
import argparse
import random
import pickle
import time
from PIL import Image
import numpy as np
from IPython import embed

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent_dir", type = str, default='./example_material')
    parser.add_argument("--model_type", type = str, default='pretrain_flant5xxl', choices=['pretrain_flant5xxl', 'pretrain_flant5xl'])
    parser.add_argument("--use_qa", action="store_true")
    return parser.parse_args()

def main():
    args = parse_args()

    # setup device to use
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

    all_output = {}

    name = 'blip2_t5'
    model_type = args.model_type

    infolder = glob.glob(os.path.join(args.parent_dir, 'Cap3D_imgs', '*'))
    random.shuffle(infolder)
    
    model, vis_processors, _ = load_model_and_preprocess(name=name, model_type=model_type, is_eval=True, device=device)
    ct = 0
        
    count = 0
    for folder in tqdm(infolder):
        if not os.path.exists(folder):
            continue
        if os.path.exists(os.path.join(folder,'caption.pkl')):
            continue
        captions = {}
        for j in range(28):
            filename = os.path.join(folder, '%05d.png'%j)
            try:
                raw_image = Image.open(filename).convert("RGB")
            except:
                print("file not work skipping", filename)
                continue

            image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)

            if args.use_qa:
                prompt = "Question: what object is in this image? Answer:"
                object = model.generate({"image": image, "prompt": prompt})[0]
                full_prompt = "Question: what is the structure and geometry of this %s?" % object
                x = model.generate({"image": image, "prompt": full_prompt}, use_nucleus_sampling=True, num_captions=5)
            else:
                try:
                    x = model.generate({"image": image}, use_nucleus_sampling=True, num_captions=5)
                except:
                    continue

            captions[j] = [z for z in x]

        count += 1
        print(count, folder)
        with open(os.path.join(folder,'caption.pkl'),'wb') as f:
            pickle.dump(captions, f)
            
if __name__ == "__main__":
    main()
