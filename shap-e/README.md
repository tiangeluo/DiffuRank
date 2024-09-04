# Overall
- Install with `pip install -e .`. Please make sure you download the shap-E code from this repo as there are modifications compared to the original repo. 

## Extract latents
We provide the code necessary to extract the shape latent code, should you need to apply it to your own 3D objects. Objaverse shapE latent codes are provided on our [dataset page](https://huggingface.co/datasets/tiange/Cap3D/tree/main/misc/ShapELatentCode_zips). 

Please run `python extract_latent.py` and the results will be saved at `./extracted_shapE_latent`. You can look at the example files to see how to apply it to your own data. We provided shapE latent codes for example objects.

## Perform DiffuRank
Please run `python diffu_rank.py` to perform DiffuRank on the input 3D objects. It will use both the shapE latent code and the caption associated with the rendered images.


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
