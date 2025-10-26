## Checkpoints

* Here is the folder u could download the ckpts for continual training & inference.
* After download them, put the ckpts here and modify the config as:

### Inference

```python
#oasis/config/eval_config.yaml - weights here for inference
amp: False
weights: null
output_dir: null # defaults to run_dir; specify this to override
```

### Training

For training, we take advantages of cutie image-pretrained ckpts, could get it here [image-pretrained](https://github.com/hkchengrex/Cutie/releases/download/v1.0). Special thanks to them. Some ckpts can also get from us as in [Link](https://drive.google.com/drive/folders/1nUEDMaa8KXgjWHz5-w4GlDAHfCCeEs8T?usp=sharing)

```python
# oasis/config/train_config.yaml
# weights here for training. Note that checkpoints contain more, e.g., optimiers, as for continual training

weights: null
checkpoint: null
```

### Data

We format the project as belows following cutie. Detailed download instructions can be checked here: [Instructions](https://github.com/hkchengrex/Cutie/blob/main/docs/TRAINING.md)

```
├── OASIS
├── DAVIS
│   └── 2017
│       ├── test-dev
│       │   ├── Annotations
│       │   └── ...
│       └── trainval
│           ├── Annotations
│           └── ...
├── BURST
│   ├── frames
│   ├── val
│   │   ├── all_classes.json
│   │   └── first_frame_annotations.json
│   ├── train
│   │   └── train.json
│   └── train-vos
│       ├── JEPGImages
│       └── Annotations
├── static
│   ├── BIG_small
│   └── ...
└── YouTube
│   ├── all_frames
│   │   └── valid_all_frames
│   ├── train
│   └── valid
├── OVIS-VOS-train
│   ├── JPEGImages
│   └── Annotations
└── MOSE
    ├── JPEGImages
    └── Annotations
```
