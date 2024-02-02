# SimVOS

The codes for ICCV 2023 paper 'Scalable Video Object Segmentation with Simplified Framework'

## :sunny: Highlights

#### * Our Goal: providing a simple and scalable VOS baseline to explore the effect of self-supervised pre-training.

#### * Our SimVOS only relies on video sequence for one-stage training and achieves favorable performance on DAVIS and YouTube datasets.

#### * Our project is built upon the [_STCN_](https://github.com/hkchengrex/STCN) library. Thanks for their contribution.

## Install the environment
We use the Anaconda to create the Python environment, which mainly follows the installation in [_STCN_](https://github.com/hkchengrex/STCN). The cuda environment we use for result reproduction is python3.6-cuda11.0-cudnn8.1.
One installation packages can be found in `environment.yaml`.


## Data preparation
We follow the same data preparation steps used in [_STCN_](https://github.com/hkchengrex/STCN). Download both DAVIS and YouTube-19 datasets:
```bash
├── DAVIS
│   ├── 2016
│   │   ├── Annotations
│   │   └── ...
│   └── 2017
│       ├── test-dev
│       │   ├── Annotations
│       │   └── ...
│       └── trainval
│           ├── Annotations
│           └── ...
├── YouTube
│   ├── all_frames
│   │   └── valid_all_frames
│   ├── train
│   ├── train_480p
│   └── valid
```

### Pre-trained model download
Please download the pre-trained weights (e.g., MAE: ViT-Base or ViT-Large) and put them in `./pretrained_models` folder.

### Training command
To train a SimVOS model (ViT-Base with MAE Init.) w/ token refinement (e.g., the default seeting with 384/384 foreground/background prototypes and `layer_index=4` for prptotype generation):
```
python -m torch.distributed.launch --master_port 9842 --nproc_per_node=8 train_simvos.py --id retrain_s03 --stage 3
```
If you want to train a SimVOS-B model w/o token refinement:
```
python -m torch.distributed.launch --master_port 9842 --nproc_per_node=2 train_simvos.py --id retrain_s03 --stage 3 --layer_index 0 --use_token_learner False
```
or SimVOS-L model:
```
python -m torch.distributed.launch --master_port 9842 --nproc_per_node=2 train_simvos.py --id retrain_s03 --stage 3 --layer_index 0 --use_token_learner False --backbone_type vit_large
```

### Evaluation command
Download the SimVOS models [SimVOS-BS(384/384-layer_index=4-vitbase)](https://drive.google.com/file/d/1v1FdDc5oFFUOBZ_Oc2yhPxDZpbHpTYsY/view?usp=drive_link), [SimVOS-B(vitbase)](https://drive.google.com/file/d/1v1FdDc5oFFUOBZ_Oc2yhPxDZpbHpTYsY/view?usp=drive_link](https://drive.google.com/file/d/1uSobYg2JQzpR-Lwb81YsUoyjEr1jaTwJ/view?usp=drive_link)), and [SimVOS-L(vitlarge)](https://drive.google.com/file/d/1bh2FyaoRlTdupvCHRiJc9O9vnRhSkcE8/view?usp=drive_link). Put the models in the `test_checkpoints` folder. After taht, run the evaluation w/ the following commands. All evaluations are done in the 480p resolution.
```
#SimVOS-BS
python submit_eval_davis_ours_all.py --model_path ./test_checkpoints --davis_path ./Data/DAVIS/2017 --output ./results --split val --layer_index 4 --use_token_learner --backbone_type #SimVOS-B
python submit_eval_davis_ours_all.py --model_path ./test_checkpoints --davis_path ./Data/DAVIS/2017 --output ./results --split val --layer_index 0 --backbone_type vit_base
SimVOS-L
python submit_eval_davis_ours_all.py --model_path ./test_checkpoints --davis_path ./Data/DAVIS/2017 --output ./results --split val --layer_index 0 --backbone_type vit_large
```

After running the above evaluation, you could get the qualitative results saved in the root project directory. You could use the offline evaluation toolikit (https://github.com/davisvideochallenge/davis2017-evaluation) to get the validation performance on DAVIS-16/17. For `test-dev` on DAVIS-17, using the online evaluation server instead.

------

If you find our work useful in your research, please consider citing:

```
@inproceedings{wu2023,
  title={Scalable Video Object Segmentation with Simplified Framework},
  author={Qiangqiang Wu and Tianyu Yang and Wei Wu and Antoni B. Chan},
  booktitle={ICCV},
  year={2023}
}
```
