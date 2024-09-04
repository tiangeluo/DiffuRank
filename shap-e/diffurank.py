# ==============================================================================
# Copyright (c) 2024 Tiange Luo, tiange.cs@gmail.com
# Last modified: September 04, 2024
#
# This code is licensed under the MIT License.
# ==============================================================================

import torch
import torch.optim as optim

from shap_e.diffusion.sample import sample_latents
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.models.download import load_model, load_config
from shap_e.models.configs import model_from_config
from shap_e.util.notebooks import create_pan_cameras, decode_latent_images, gif_widget

import os
import argparse
import glob
import pickle
import pandas as pd
import csv
import time
import random
import numpy as np
from datetime import datetime
import math
from PIL import Image

from IPython import embed

def train(rank, args):
    if args.gpus > 1:
        setup_ddp(rank, args)

    niter = 5
    batch_size = 8
    save_name = args.save_name

    torch.manual_seed(rank+10+int(datetime.now().timestamp()))

    ## set resume_flag to load finetuned shapE model, otherwise will load OpenAI shapE model weights
    resume_flag = True if args.resume_name != 'none' else False
    if resume_flag:
        model_list = glob.glob('./model_ckpts/%s*.pth'%save_name)
        idx_rank = []
        for l in model_list:
            idx_rank.append(int(l.split('/')[-1].split('_')[-2][5:]) * 41000 + int(l.split('/')[-1].split('_')[-1].split('.')[0]))
        newest = np.argmax(np.array(idx_rank))
        args.resume_name = model_list[newest].split('/')[-1].split('.')[0]

    start_epoch = 0 if not resume_flag else int(args.resume_name.split('_')[-2][5:])
    start_iter = 0 if not resume_flag else int(args.resume_name.split('_')[-1].split('.')[0])
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if resume_flag:
        print('reload from ./model_ckpts/%s.pth'%args.resume_name)
        checkpoint = torch.load('./model_ckpts/%s.pth'%args.resume_name, map_location=device)

    if not resume_flag:
        model = load_model('text300M', device=device)
    else:
        model = model_from_config(load_config('text300M'), device=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    if args.gpus > 1:
        model = DistributedDataParallel(
                model, device_ids=[rank], find_unused_parameters=False
        )
    
    diffusion = diffusion_from_config(load_config('diffusion'))

    paths = glob.glob(args.image_dir+'/*')
    random.shuffle(paths)
    for index, path in enumerate(paths):
        index_file = os.path.join(path, 'diffurank_scores.pkl')
        if os.path.exists(index_file):
            continue

        caption_file = os.path.join(path, 'caption.pkl')
        if not os.path.exists(caption_file):
            print('caption not generated:', path)
            continue
        try:
            captions = pickle.load(open(caption_file,'rb'))
        except:
            print('caption file wrong:', path)
            continue
        prompt = []
        try:
            for i in range(28):
                prompt += captions[i]
        except:
            print('caption file not completed:', path)
            continue

        x_start = torch.load(os.path.join(args.latent_dir, path.split('/')[-1]+'.pt')).cuda()
        print('DiffuRank:', index, path.split('/')[-1], len(paths))

        batch_size = 140
        model_kwargs=dict(texts=prompt)
        x_start = x_start.repeat(batch_size,1)
        x0 =x_start.detach()
        view_loss = []
        for _ in range(5):
            with torch.no_grad():
                t = []
                for _ in range(5):
                    # 800~900 is the hyperparameter we heuristically set
                    # its principle should be 0~1000, i.e., the range of the time steps
                    t_cur = torch.randint(850, 851, size=(int(batch_size/5),), device=device)
                    random_integer = np.random.randint(-50, 51)
                    t_cur += random_integer
                    t.append(t_cur.unsqueeze(0))
                t = torch.cat(t,0).transpose(0,1).reshape(-1)

                loss = diffusion.diffurank_scores(model, x_start, t, model_kwargs=model_kwargs, times = 5)
            view_loss.append(loss['mse'].reshape(1, -1, 5))

        view_loss = torch.cat(view_loss)
        pickle.dump(torch.mean(torch.mean(view_loss,-1),0).cpu().numpy(), open(index_file, 'wb'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    model_group = parser.add_argument_group('Model settings')
    model_group.add_argument('--gpus', type = int, default = 1, help = 'how many gpu we use')
    model_group.add_argument('--resume_name', type = str, default = 'none', help = 'port for parallel')
    model_group.add_argument('--save_name', type = str, default = 'none', help = 'port for parallel')
    model_group.add_argument('--image_dir', type = str, default='../example_material/Cap3D_imgs')
    model_group.add_argument('--latent_dir', type = str, default='../example_material/extracted_shapE_latent')

    args = parser.parse_args()

    if args.gpus == 1:
        train(args.gpus, args)
    else:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = args.port
        args.world_size = args.gpus
        mp.spawn(train, nprocs=args.gpus, args=(args,))


