# [View Selection for 3D Captioning via Diffusion Ranking](http://arxiv.org/abs/2404.07984)

<a href="https://arxiv.org/abs/2404.07984"><img src="https://img.shields.io/badge/arXiv-2404.07984-b31b1b.svg" height=20.5></a>

[Tiange Luo](https://tiangeluo.github.io/), [Justin Johnson†](https://web.eecs.umich.edu/~justincj) [Honglak Lee†](https://web.eecs.umich.edu/~honglak/) (†Equal Advising)

## Rendering
Please first download our Blender via the below commands. You can use your own Blender, while may need to pip install several packages.
```
wget https://huggingface.co/datasets/tiange/Cap3D/resolve/main/misc/blender.zip
unzip blender.zip
```

Please run the below command to render objects into `.png` images saved at `{parent_dir}/Cap3D_imgs/{uid}/{0~7}.png`
```
# --object_path_pkl: point to a pickle file which store the object path
# --parent_dir: the directory store the rendered images and their associated camera matrix
# Rendered images will be stored at partent_dir/Cap3D_imgs/

./blender-3.4.1-linux-x64/blender -b -P render_script_type1.py -- --object_path_pkl './example_material/example_object_path.pkl' --parent_dir './example_material'
```
