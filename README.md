<div align="center">
<h3>Disjoint Masking with Joint Distillation for Efficient Masked Image Modeling</h3>

Xin Ma<sup>1</sup>, Chang Liu<sup>2</sup>, Chunyu Xie<sup>3</sup>, Long Ye<sup>1</sup>, Yafeng Deng<sup>3</sup>, Xiangyang Ji<sup>2</sup>,

<sup>1</sup> Communication University of China, <sup>2</sup> Tsinghua University, <sup>3</sup> 360 AI Research.

</div>

This repo is the official implementation of [Disjoint Masking with Joint Distillation for Efficient Masked Image Modeling](). It currently concludes codes and Pre-trained checkpoints.

<p align="center">
  <img src="https://user-images.githubusercontent.com/94091472/210162854-da4afe07-4304-4e43-af55-45092270b479.png" width="1500">
</p>

## Introduction
This work aims to alleviate the training inefficiency in masked image modeling. We believe the insufficient utilization of training signals should be responsible. To alleviate this issue, DMJD imposes a masking regulation to generate multiple complementary views facilitating more invisible tokens of each image to be reconstructed in the invisible reconstruction branch and further devise a dual-branch joint distillation architecture with an additional visible distillation branch to take full use of the input signals with superior targets. Extensive experiments and visualizations prove that with increased prediction rates, visible distillation, and superior targets can accelerate the training convergence yet not sacrificing the model generalization ability.

The contributions are summarized as follows: 
* We propose a conceptually simple yet learning-efficient MIM training scheme, termed disjoint masking with joint distillation (DMJD), which targets increasing the utilization of per image at each training loop. 
* We devise a multi-view generation strategy, i.e., disjoint masking (DM), to increase the prediction rate while keeping the corruption rate for efficient MIM and introduce the adaptive learning rate scale rule for better model generalization with augmented training batches.
* We develop a dual-branch architecture for joint distillation (JD), effectively pursuing representation learning on both visible and invisible regions with superior targets. 
* We conduct sufficient evaluations justifying our DMJD can significantly accelerate model convergence and achieve outstanding performances on standard benchmark. Take an example, for linear probing classification on [MaskFeat](https://arxiv.org/abs/2112.09133) and [ConvMAE](https://arxiv.org/abs/2205.03892) baselines, DMJD achieves performance gains of 3.4% and 5.8% with 1.8× and 3× acceleration.

## Getting Started
### Setup
- Installation and preparation can follow the [DeiT repo](https://github.com/facebookresearch/deit). Note that this repo is based on [timm==0.4.12](https://github.com/rwightman/pytorch-image-models).

- Using DMJD with Docker.

**Step 1.** We provide a [Dockerfile](./files/Dockerfile) to build an image. Ensure that your [docker version>=19.03](https://docs.docker.com/engine/install/).
```bash
# build an image with PyTorch 1.11, CUDA 11.3, and mmsegmentation
# If you prefer other versions, just modified the Dockerfile
docker build -t env:dmjd .
```
**Step 2.** Run it with
```bash
docker run --gpus all --shm-size=8g -itd -v {DATA_DIR}:/path/to/data -v {CODE_DIR}:/path/to/dmjd env:dmjd
```

### Pre-training
The pre-training instruction is in [PRETRAIN.md](PRETRAIN.md).

### Evaluation

 - The pre-trained checkpoints and its corresponding results on classification task.
<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom"></th>
<th valign="bottom">ConViT-Base</th>
<th valign="bottom">ConViT-Large</th>
<!-- TABLE BODY -->
<tr><td align="left">pre-trained checkpoint</td>
<td align="center"><a href="https://drive.google.com/file/d/13kVEGFlZcRdSdt8-4hQ1dwOezSx0DLVl/view?usp=share_link">download</a></td>
<td align="center"><a href="https://drive.google.com/file/d/1HdxvfWy8NlfhOJVBC5fLHOtD0_YhbI-j/view?usp=share_link">download</a></td>
</tr>
</tbody></table>

 - Main results on ImageNet-1K.

| Method | Backbone | ETE | Gh. | Learning Target | FT acc@1(%) | LIN acc@1(%) |
| :---: | :---: | :---: | :--- | :---: | :--- | :--- | 
| MaskFeat | ViT-B | 1600 | 240 | HOG | 84.0 | 68.5 | 
|  +DMJD | ViT-B | 1600 | 132 (1.8×) | HOG | 84.1 (+0.1) | 71.9 (+3.4) | 
| ConvMAE | ConViT-B | 1600 | 300 | RGB | 85.0 | 70.9 | 
|  +DMJD | ConViT-B | 800 | 101 (3×) | HOG | 85.2 (+0.2) | 76.7 (+5.8) |
| ConvMAE | ConViT-L | 800 | 480 | RGB | 86.2 | - | 
|  +DMJD | ConViT-L | 800 | 267 (1.8×) | HOG | 86.3 (+0.1) | 79.7 | 

 - The fine-tuning and linear probing instruction is in [FINETUNE.md](FINETUNE.md).

## Acknowledgement
This repo is built on top of [DeiT](https://github.com/facebookresearch/deit),  [MAE](https://github.com/facebookresearch/mae) and [ConvMAE](https://github.com/Alpha-VL/ConvMAE). The semantic segmentation parts is based on [MMSegmentation](https://github.com/open-mmlab/mmsegmentation). Thanks for their wonderful work.

## License
DMJD is released under the [MIT License](https://github.com/mx-mark/DMJD/blob/main/LICENSE).

## Citation

```bash
```

