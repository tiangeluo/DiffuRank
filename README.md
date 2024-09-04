# [View Selection for 3D Captioning via Diffusion Ranking](http://arxiv.org/abs/2404.07984)

<a href="https://cap3d-um.github.io/"><img src="https://img.shields.io/static/v1?label=Project&message=Website&color=red" height=20.5></a>
<a href="https://arxiv.org/abs/2404.07984"><img src="https://img.shields.io/badge/arXiv-2404.07984-b31b1b.svg" height=20.5></a>
<a href="https://arxiv.org/abs/2306.07279"><img src="https://img.shields.io/badge/arXiv-2306.07279-b31b1b.svg" height=20.5></a>


[Tiange Luo](https://tiangeluo.github.io/), [Justin Johnson†](https://web.eecs.umich.edu/~justincj) [Honglak Lee†](https://web.eecs.umich.edu/~honglak/) (†Equal Advising)

Data download available at [Hugging Face](https://huggingface.co/datasets/tiange/Cap3D), including `1,002,422` 3D-caption pairs covering the whole [Objaverse](https://arxiv.org/abs/2212.08051) and subset of [Objaverse-XL](https://arxiv.org/abs/2307.05663) datasets. We also the associated objects' point clouds and rendered images (with camera, depth, and MatAlpha information).

## Overall Logistics
To identify key views of a 3D object for downstream tasks like captioning in DiffuRank, we follow a multi-step process:

- We render the object from 28 distinct views using two different rendering settings, producing 2D images for each view.
- We generate captions for all 28 images, resulting in 5 captions per image.
- We extract ShapE latent codes, which serve as the 3D descriptor for the object.
- Using DiffuRank, we calculate the alignment between the generated captions and the ShapE latent codes to assess their compatibility.
- Finally, the images whose captions show the highest alignment with the 3D descriptor are selected as the key views.

### Rendering
Please first download our Blender via the below commands. You can use your own Blender, while may need to pip install several packages.
```
wget https://huggingface.co/datasets/tiange/Cap3D/resolve/main/misc/blender.zip
unzip blender.zip
```

Please run the below command to render objects into `.png` images saved at `{parent_dir}/Cap3D_imgs/{uid}/{00000~00028}.png`
```
# --object_path_pkl: point to a pickle file which store the object path
# --parent_dir: the directory store the rendered images and their associated camera matrix

# Rendered images will be stored at partent_dir/Cap3D_imgs/
# 8 views will be rendered for each object; camera placed horizontally around the object's default orientation
./blender-3.4.1-linux-x64/blender -b -P render_script_type1.py -- --object_path_pkl './example_material/example_object_path.pkl' --parent_dir './example_material'

# Rendered images will be stored at partent_dir/Cap3D_imgs/
# 20 views will be rendered for each object; camera placed randomly
./blender-3.4.1-linux-x64/blender -b -P render_script_type2.py -- --object_path_pkl './example_material/example_object_path.pkl' --parent_dir './example_material'
```

### Captioning
We currently use BLIP2 to generate captions for rendered images. There are a lot of other new captioning model that can be used for this task.

```bash
pip install salesforce-lavis

# this program will scan all the folders inside partent_dir/Cap3D_imgs/
python caption_blip2.py --parent_dir ./example_material
```

### Extract ShapE Latent Codes
Please go to [shap-e](https://github.com/tiangeluo/DiffuRank/tree/main/shap-e) folder to extract ShapE latent codes.

### DiffuRank process
Please go to [shap-e](https://github.com/tiangeluo/DiffuRank/tree/main/shap-e) folder to perform DiffuRank.




## Citation
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
