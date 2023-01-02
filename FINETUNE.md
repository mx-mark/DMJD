## Fine-tuning Pre-trained DMJD for Classification

### Fine-tuning

Get our pre-trained checkpoints from [here](https://github.com/mx-mark/DMJD#evaluation).

To fine-tune our pre-trained ConViT-Base with **single-node training**, run the following on 1 node with 8 GPUs:
```shell
cd /PATH/TO/ROOT/DIR
CURDIR='/PATH/TO/ROOT/DIR'
export PYTHONPATH="$PYTHONPATH:$CURDIR"
DATA_PATH='/PATH/TO/IMAGENET'
GPUS=8

OUTPUT_DIR='/PATH/TO/OUTPUT/DIR'
MODEL_PATH="/PATH/TO/PRETRAIN/WEIGHTS"
OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=${GPUS} main_finetune.py \
    --batch_size 128 \
    --model convit_base_patch16 \
    --finetune ${MODEL_PATH} \
    --epochs 100 \
    --blr 5e-4 --layer_decay 0.65 --drop_path 0.1 \
    --weight_decay 0.05 --reprob 0.25 --mixup 0.8 --cutmix 1.0 \
    --dist_eval --data_path ${DATA_PATH} --output_dir ${OUTPUT_DIR}
```

To train ConViT-Large, set `--model convit_large_patch16`, `--layer_decay 0.75`, and `--drop_path 0.2`. It is sufficient to train 50 epochs `--epochs 50`.

### Linear Probing

Run the following on 1 nodes with 8 GPUs each:
```shell
cd /PATH/TO/ROOT/DIR
CURDIR='/PATH/TO/ROOT/DIR'
export PYTHONPATH="$PYTHONPATH:$CURDIR"
DATA_PATH='/PATH/TO/IMAGENET'
GPUS=8

OUTPUT_DIR='/PATH/TO/OUTPUT/DIR'
MODEL_PATH="/PATH/TO/PRETRAIN/WEIGHTS"
OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=${GPUS} main_linprobe.py \
    --batch_size 512 \
    --model convit_base_patch16 \
    --global_pool \
    --finetune ${MODEL_PATH} \
    --epochs 90 \
    --blr 0.1 --weight_decay 0.0 \
    --dist_eval --data_path ${DATA_PATH} --output_dir ${OUTPUT_DIR}
```

To train ConViT-Large, set `--model convit_large_patch16` and `--blr 0.05`. It is sufficient to train 50 epochs `--epochs 50`.
