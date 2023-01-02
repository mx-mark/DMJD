## Pre-training DMJD

To pre-train ConViT-Base with **single-node distributed training**, run the following on 1 nodes with 8 GPUs each:
```shell
cd /PATH/TO/ROOT/DIR
CURDIR='/PATH/TO/ROOT/DIR'
export PYTHONPATH="$PYTHONPATH:$CURDIR"
DATA_PATH='/PATH/TO/IMAGENET'
GPUS=8

OUTPUT_DIR='/PATH/TO/OUTPUT/DIR'
OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=${GPUS} main_pretrain.py \
    --batch_size 64 \
    --model dmjd_convit_base_patch16_dec8_hog \
    --mask_ratio 0.6 \
    --mask_type block \
    --pred_type hog \
    --mim_loss_type l2 \
    --vis_loss_type smooth_l1 \
    --epochs 400 \
    --warmup_epochs 20 \
    --blr 6e-4 --weight_decay 0.05 \
    --data_path ${DATA_PATH} \
    --output_dir ${OUTPUT_DIR}
```
- Here the effective batch size is 64 (unique `batch_size` per gpu) * 2 (the number of masked views) * 8 (gpus per node) = 1024. If memory or # gpus is limited, use `--accum_iter` to maintain the effective batch size, which is unique `batch_size` (per gpu) * 2 (the number of masked views) * 8 (gpus per node) * `accum_iter`.
- `blr` is the base learning rate. The actual `lr` is computed by the adaptive learning rate scaling rule: `lr` = `blr` * unique batch size * $\frac{m_{pred}}{m_{corr}}$ / 256.
- Here we use the block-wise masking proposed by [SimMIM](https://arxiv.org/abs/2111.09886).
- Here we use HOG as the laerning target for better representation learning.
- Training time is ~101h in 8× Nvidia A100 (40GB) (400 epochs).

To train ConViT-Large, set `--model dmjd_convit_large_patch16_dec8_hog` and `--clip_grad 3.0` for stable training.
