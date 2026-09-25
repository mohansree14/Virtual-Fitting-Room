# Virtual Fitting Room — 3D try-on from your own photos

Try a garment on a person from a single front photo and get a 3D, textured result back.

![Try-on pipeline](/assets/teaser.gif)

## Built on M3D-VTON

The core model in this repo is **M3D-VTON** (Zhao et al., ICCV 2021). The code in `models/`, `data/`, `util/`, `train.py`, `test.py` and `rgbd2pcd.py` is theirs and is included unchanged, except where listed below. All credit for the network design and the MPV3D dataset goes to the original authors:

- Paper: [M3D-VTON: A Monocular-to-3D Virtual Try-On Network](https://arxiv.org/abs/2108.05126)
- Original repo: [fyviezhao/M3D-VTON](https://github.com/fyviezhao/M3D-VTON)

## What I built on top

The original repo works well on its benchmark dataset. To run it on **my own photos**, you need a pose file, a human parsing map, a cloth mask, a palm mask and image gradients, and you have to produce them with separate external tools. I wrote a lightweight preprocessing pipeline to generate all of these, and got the model running on a machine without a GPU.

| File | What it does |
|---|---|
| `pose.py` | Extracts 18 body keypoints with OpenCV's DNN module and a COCO OpenPose Caffe model, saved in the OpenPose JSON format M3D-VTON expects. No full OpenPose build needed. |
| `human.py` | Resizes person photos to 320×512, builds the palm mask (HSV skin segmentation, largest contour) and Sobel X/Y gradient maps, then runs `pose.py`. All of a person photo's preprocessing in one command. |
| `segment.py` | Human segmentation map using torchvision's pretrained DeepLabV3-ResNet101. |
| `cloth.py` | Resizes garment images and builds the cloth mask by thresholding. |
| `visualize.py` | Automates the manual MeshLab steps (normal estimation and screened Poisson remeshing) with `pymeshlab` and `trimesh`. |
| `options/base_options.py` | CPU fallback: defaults to CPU and only uses CUDA when it's available. |

### Pipeline

```
person photo ─┬─ pose.py ────────► keypoints JSON
              ├─ segment.py ─────► parsing map
              └─ human.py ───────► palm mask, Sobel maps
garment photo ── cloth.py ───────► cloth mask
                        │
                        ▼
      M3D-VTON: MTM (warp) → DRM (depth) → TFM (texture)
                        │
                        ▼
      rgbd2pcd.py → point cloud → visualize.py → 3D mesh
```

### Known limitations

- DeepLabV3 produces a person/background mask, not the fine-grained body-part labels M3D-VTON was trained on, so results on custom photos are rougher than on MPV3D.
- The depth step in `human.py` is a grayscale placeholder, not a real depth estimate.

## Running it

Requirements: `python >= 3.8`, `pytorch == 1.6.0`, `torchvision == 0.7.0`, `opencv-python`, plus `trimesh` and `pymeshlab` for remeshing. Download the [pretrained models](https://figshare.com/s/fad809619d2f9ac666fc) and the OpenPose COCO weights (`pose_iter_440000.caffemodel`).

1. Preprocess your photos. Name the person photo like the dataset (`<id>=person_whole_front.png`, garment as `.jpg`) so the loader finds every file. Each script writes into the matching subfolder of `--dataroot`:
   ```sh
   python cloth.py   example/cloth/shirt.jpg --dataroot example
   python human.py   example/image/me=person_whole_front.png --dataroot example   # resize, palm mask, Sobel, pose, depth
   python segment.py example/image/me=person_whole_front.png --dataroot example
   ```
   `pose.py` can also be run on its own. It looks for `openpose_pose_coco.prototxt` and `pose_iter_440000.caffemodel` in the repo root, or you can pass `--prototxt` and `--caffemodel`.
2. Run the three modules in order:
   ```sh
   python test.py --model MTM --name MTM --dataroot example --datalist test_pairs --results_dir results
   python test.py --model DRM --name DRM --dataroot example --datalist test_pairs --results_dir results
   python test.py --model TFM --name TFM --dataroot example --datalist test_pairs --results_dir results
   ```
3. Build the 3D result: `python rgbd2pcd.py`, then `python visualize.py results/aligned/pcd/test_pairs/<name>.ply`.

For training on the MPV3D dataset, see the [original repo](https://github.com/fyviezhao/M3D-VTON#training-on-mpv3d-dataset).

## License

The M3D-VTON code and the MPV3D dataset are restricted to **non-commercial research and educational use**. This project follows the same terms.

## Citation

```
@InProceedings{M3D-VTON,
    author    = {Zhao, Fuwei and Xie, Zhenyu and Kampffmeyer, Michael and Dong, Haoye and Han, Songfang and Zheng, Tianxiang and Zhang, Tao and Liang, Xiaodan},
    title     = {M3D-VTON: A Monocular-to-3D Virtual Try-On Network},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2021},
    pages     = {13239-13249}
}
```
