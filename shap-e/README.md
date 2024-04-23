## Overall
- Install with `pip install -e .`. Please make sure you download the shap-E code from this repo as there are modifications compared to the original repo. Additionally, you need to install [Pytorch3D](https://github.com/facebookresearch/pytorch3d) to render images via `stf` mode. You can skip installing Pytorch3D to generate meshes, while it is needed to calculate final numbers.

- Download finetuned checkpoint from https:https://huggingface.co/datasets/tiange/Cap3D/tree/main/misc/our_finetuned_models, and move it to `model_ckpts`.
```
wget https://huggingface.co/datasets/tiange/Cap3D/resolve/main/misc/our_finetuned_models/shapE_finetuned_with_825kdata.pth
mkdir model_ckpts
mv shapE_finetuned_with_825kdata.pth model_ckpts
```

## Extract latents
We provide the code necessary to extract the shape latent code, should you need to apply it to your own 3D objects. Objaverse shapE latent codes are provided on our [dataset page](https://huggingface.co/datasets/tiange/Cap3D/tree/main/misc/ShapELatentCode_zips). 

Please run `python extract_latent.py` and the results will be saved at `./extracted_shapE_latent`. You can look at the example files to see how to apply it to your own data.

## Rendering (cleaning code)
```
wget https://huggingface.co/datasets/tiange/Cap3D/resolve/main/blender.zip
unzip blender.zip

./blender-3.4.1-linux-x64/blender -b -P render_script_shapE.py -- --save_dir './rendering_output' --parent_dir './shapE_inference/Cap3D_test1_meshes'
```


## Citation

if you use shap-E model/data, please cite:
```
@article{jun2023shap,
  title={Shap-e: Generating conditional 3d implicit functions},
  author={Jun, Heewoo and Nichol, Alex},
  journal={arXiv preprint arXiv:2305.02463},
  year={2023}
}
```

If you find our code or data useful, please consider citing:
```
@article{luo2024view,
      title={View Selection for 3D Captioning via Diffusion Ranking},
      author={Luo, Tiange and Johnson, Justin and Lee, Honglak},
      journal={arXiv preprint arXiv:2404.07984},
      year={2024}
}

@article{luo2023scalable,
      title={Scalable 3D Captioning with Pretrained Models},
      author={Luo, Tiange and Rockwell, Chris and Lee, Honglak and Johnson, Justin},
      journal={arXiv preprint arXiv:2306.07279},
      year={2023}
}
```
