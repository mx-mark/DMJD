# Copyright (c) 2022 Alpha-VL
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# MAE: https://github.com/facebookresearch/mae
# ConvMAE: https://github.com/Alpha-VL/ConvMAE
# RelPosViT: timm/models/vision_transformer_relpos.py
# Projector: ibot/models/head.py, simsiam/builder.py
# --------------------------------------------------------

from functools import partial
import pdb
import torch
import torch.nn as nn

from vision_transformer import PatchEmbed, Block, CBlock

from util.pos_embed import get_2d_sincos_pos_embed


def expand_patches_indice(ids_shuffle, sub_w, H, W):
    # convert to the 2d coordinate in sub scale (49 patches)
    ids_x, ids_y = torch.mul(torch.div(ids_shuffle, sub_w, rounding_mode='floor'),2), torch.mul(torch.fmod(ids_shuffle, sub_w),2)
    ids_x1, ids_y1 = torch.add(ids_x, 1), torch.add(ids_y, 1)
    # project back to the 1d coordinate in original scale (196 patches)
    ids_xy = torch.add(ids_y, ids_x, alpha=W)
    ids_xy1 = torch.add(ids_y1, ids_x, alpha=W)
    ids_x1y = torch.add(ids_y, ids_x1, alpha=W)
    ids_x1y1 = torch.add(ids_y1, ids_x1, alpha=W)
    return ids_xy, ids_xy1, ids_x1y, ids_x1y1


class DMJDConViT(nn.Module):
    """ DMJD with ConViT backbone
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=1024, depth=24, 
                 num_heads=16, decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, decoder_pred_dim=768):
        super().__init__()
        # --------------------------------------------------------------------------
        # DMJD encoder specifics
        # --------------------------------------------------------------------------
        self.patch_embed1 = PatchEmbed(
                img_size=img_size[0], patch_size=patch_size[0], in_chans=in_chans, embed_dim=embed_dim[0])
        self.patch_embed2 = PatchEmbed(
                img_size=img_size[1], patch_size=patch_size[1], in_chans=embed_dim[0], embed_dim=embed_dim[1])
        self.patch_embed3 = PatchEmbed(
                img_size=img_size[2], patch_size=patch_size[2], in_chans=embed_dim[1], embed_dim=embed_dim[2])

        self.patch_embed4 = nn.Linear(embed_dim[2], embed_dim[2])
        self.stage1_output_decode = nn.Conv2d(embed_dim[0], embed_dim[2], 4, stride=4)
        self.stage2_output_decode = nn.Conv2d(embed_dim[1], embed_dim[2], 2, stride=2)

        num_patches = self.patch_embed3.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim[2]), requires_grad=False)
        self.blocks1 = nn.ModuleList([
            CBlock(
                dim=embed_dim[0], num_heads=num_heads, mlp_ratio=mlp_ratio[0],  qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth[0])])
        self.blocks2 = nn.ModuleList([
            CBlock(
                dim=embed_dim[1], num_heads=num_heads, mlp_ratio=mlp_ratio[1],  qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth[1])])
        self.blocks3 = nn.ModuleList([
            Block(
                dim=embed_dim[2], num_heads=num_heads, mlp_ratio=mlp_ratio[2],  qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth[2])])
        self.norm = norm_layer(embed_dim[-1])
        self.embed_dim = embed_dim
        # --------------------------------------------------------------------------
        # DMJD decoder specifics
        # --------------------------------------------------------------------------
        # Masked prediction branch
        self.decoder_embed = nn.Linear(embed_dim[-1], decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding
        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio[0], qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, decoder_pred_dim, bias=True) # decoder to target features

        # Visible distillation branch
        self.proj_embed = nn.Sequential(
            nn.Linear(embed_dim[-1], decoder_embed_dim),
            norm_layer(decoder_embed_dim),
            nn.GELU(),
            nn.Linear(decoder_embed_dim, decoder_embed_dim),
            norm_layer(decoder_embed_dim),
            nn.GELU(),
            nn.Linear(decoder_embed_dim, decoder_embed_dim),
            norm_layer(decoder_embed_dim, elementwise_affine=False),
        )
        self.predictor = nn.Sequential(
            # nn.Linear(decoder_embed_dim, decoder_embed_dim),
            # norm_layer(decoder_embed_dim),
            # nn.GELU(),
            nn.Linear(decoder_embed_dim, decoder_pred_dim),
        )
        # --------------------------------------------------------------------------

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed3.num_patches**.5), cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed3.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed3.num_patches**.5), cls_token=False)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))
        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)
    
    def random_masking(self, x, mask_ratio, pred_ratio=1.0):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N = x.shape[0]
        L = self.patch_embed3.num_patches
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)
        
        """
        ### ---------------------------
        ###         disjoint mask
        ### --------------------------- 
        """
        disjoint_ratio = min(pred_ratio - mask_ratio, mask_ratio)
        num_mask = L - len_keep
        num_extra = int(disjoint_ratio * L)

        # sampling disjoint mask 
        ids_joint, ids_disj = ids_shuffle[:, len_keep:], ids_keep    
        shuffle_disj_ids = torch.randperm(ids_disj.shape[1], device=ids_disj.device)
        shuffle_joint_ids = torch.randperm(ids_joint.shape[1], device=ids_joint.device)
        mask_ids1, vis_ids1 = shuffle_disj_ids[:num_extra], shuffle_disj_ids[num_extra:] 
        mask_ids2, vis_ids2 = shuffle_joint_ids[:(num_mask-num_extra)], shuffle_joint_ids[(num_mask-num_extra):]
        
        # concat the indice with a specific order [visible indice, masked indice]
        ids_shuffle = torch.cat([ids_disj[:, vis_ids1], ids_joint[:, vis_ids2], ids_disj[:, mask_ids1], ids_joint[:, mask_ids2]], dim=1)
        ids_restore_disj = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset of disjoint mask
        ids_keep_disj = ids_shuffle[:, :len_keep]

        # generate the binary mask: 0 is keep, 1 is remove
        mask_disj = torch.ones([N, L], device=x.device)
        mask_disj[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask_disj = torch.gather(mask_disj, dim=1, index=ids_restore_disj)
        
        # build batch
        ids_keep = torch.cat([ids_keep, ids_keep_disj], dim=0)
        mask = torch.cat([mask, mask_disj], dim=0)
        ids_restore = torch.cat([ids_restore, ids_restore_disj], dim=0)
        return ids_keep, mask, ids_restore
    
    def block_masking(self, x, mask_ratio, pred_ratio=1.0):
        """
        Perform consecutive-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N = x.shape[0]
        L = self.patch_embed3.num_patches
        sub_L = L//4
        H = W = L ** 0.5
        len_keep = int(sub_L * (1 - mask_ratio))
        
        noise = torch.rand(N, sub_L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle_ori = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        # expand mask to consecutive patches
        ids_xy, ids_xy1, ids_x1y, ids_x1y1 = expand_patches_indice(ids_shuffle_ori, sub_L**0.5, H, W)
        ids_shuffle = torch.cat([
            ids_xy[:, :len_keep], ids_xy1[:, :len_keep], ids_x1y[:, :len_keep], ids_x1y1[:, :len_keep],
            ids_xy[:, len_keep:], ids_xy1[:, len_keep:], ids_x1y[:, len_keep:], ids_x1y1[:, len_keep:]],
            dim=1).to(torch.long)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        # keep the first subset
        ids_keep = ids_shuffle[:, :4*len_keep]

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :4*len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        """
        ### ---------------------------
        ###         disjoint mask
        ### --------------------------- 
        """       
        disjoint_ratio = min(pred_ratio - mask_ratio, mask_ratio)
        num_mask = sub_L - len_keep
        num_extra = int(disjoint_ratio * sub_L)

        # sampling disjoint mask on a subset
        ids_joint, ids_disj = ids_shuffle_ori[:, len_keep:], ids_shuffle_ori[:, :len_keep]    
        shuffle_disj_ids = torch.randperm(ids_disj.shape[1], device=ids_disj.device)
        shuffle_joint_ids = torch.randperm(ids_joint.shape[1], device=ids_joint.device)
        # though have the same index, but batch instances are not same
        mask_ids1, vis_ids1 = shuffle_disj_ids[:num_extra], shuffle_disj_ids[num_extra:] 
        mask_ids2, vis_ids2 = shuffle_joint_ids[:(num_mask-num_extra)], shuffle_joint_ids[(num_mask-num_extra):]
        # concat with a specific order [visible indice, masked indice] in a sub patches
        ids_shuffle = torch.cat([ids_disj[:, vis_ids1], ids_joint[:, vis_ids2], ids_disj[:, mask_ids1], ids_joint[:, mask_ids2]], dim=1)
        
        # expand mask to consecutive patches
        ids_xy, ids_xy1, ids_x1y, ids_x1y1 = expand_patches_indice(ids_shuffle, sub_L**0.5, H, W)
        # concat with a specific order [visible indice, masked indice] in the whole patches
        ids_shuffle = torch.cat([
            ids_xy[:, :len_keep], ids_xy1[:, :len_keep], ids_x1y[:, :len_keep], ids_x1y1[:, :len_keep],
            ids_xy[:, len_keep:], ids_xy1[:, len_keep:], ids_x1y[:, len_keep:], ids_x1y1[:, len_keep:]],
            dim=1).to(torch.long)
        ids_restore_disj = torch.argsort(ids_shuffle, dim=1)
        
        # keep the first subset of disjoint mask
        ids_keep_disj = ids_shuffle[:, :4*len_keep]

        # generate the binary mask: 0 is keep, 1 is remove
        mask_disj = torch.ones([N, L], device=x.device)
        mask_disj[:, :4*len_keep] = 0
        # unshuffle to get the binary mask
        mask_disj = torch.gather(mask_disj, dim=1, index=ids_restore_disj)
        
        # build batch
        ids_keep = torch.cat([ids_keep, ids_keep_disj], dim=0)
        mask = torch.cat([mask, mask_disj], dim=0)
        ids_restore = torch.cat([ids_restore, ids_restore_disj], dim=0)
        return ids_keep, mask, ids_restore

    def forward_encoder(self, x, mask_ratio, mask_type='block', pred_ratio=1.0):
        # prepare mask
        if mask_type == 'rand':
            ids_keep, mask, ids_restore = self.random_masking(x, mask_ratio, pred_ratio)
        else:
            ids_keep, mask, ids_restore = self.block_masking(x, mask_ratio, pred_ratio)
        mask_for_patch1 = mask.reshape(-1, 14, 14).unsqueeze(-1).repeat(1, 1, 1, 16).reshape(-1, 14, 14, 4, 4).permute(0, 1, 3, 2, 4).reshape(mask.shape[0], 56, 56).unsqueeze(1)
        mask_for_patch2 = mask.reshape(-1, 14, 14).unsqueeze(-1).repeat(1, 1, 1, 4).reshape(-1, 14, 14, 2, 2).permute(0, 1, 3, 2, 4).reshape(mask.shape[0], 28, 28).unsqueeze(1)
        
        # embed patches
        x = self.patch_embed1(x)
        # repeat twice for two masked views
        x = torch.cat([x, x], dim=0)
        for blk in self.blocks1:
            x = blk(x, 1 - mask_for_patch1)
        stage1_embed = self.stage1_output_decode(x).flatten(2).permute(0, 2, 1)
        x = self.patch_embed2(x)
        for blk in self.blocks2:
            x = blk(x, 1 - mask_for_patch2)
        stage2_embed = self.stage2_output_decode(x).flatten(2).permute(0, 2, 1)
        x = self.patch_embed3(x)
        x = x.flatten(2).permute(0, 2, 1)
        x = self.patch_embed4(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed
        # pick out visible tokens 
        x = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, x.shape[-1]))
        stage1_embed = torch.gather(stage1_embed, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, stage1_embed.shape[-1]))
        stage2_embed = torch.gather(stage2_embed, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, stage2_embed.shape[-1]))

        # apply Transformer blocks
        for blk in self.blocks3:
            x = blk(x)
        x = x + stage1_embed + stage2_embed
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):

        # branch 1-visible tokens distillation        
        x_vis = self.proj_embed(x)
        x_vis = self.predictor(x_vis)

        # branch 2-masked tokens reconstruction
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1]  - x.shape[1], 1)
        x_ = torch.cat([x, mask_tokens], dim=1)  # no cls token
        x = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        return x, x_vis

    def forward_loss(self, imgs, pred, target, mask, mim_loss_type='l2', vis_loss_type='smooth_l1', theta=1.0):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        # repeat target twice for two masked views
        target = torch.cat([target, target], dim=0)

        # loss on visible branch
        pred, pred_vis = pred
        target_vis = target[torch.abs(mask-1).to(torch.bool)].view(target.shape[0], -1, target.shape[-1])
        if vis_loss_type == 'smooth_l1':
            loss_vis = nn.functional.smooth_l1_loss(pred_vis, target_vis, reduction='none', beta=2.0)
        else:
            loss_vis = (pred_vis - target_vis) ** 2
        loss_vis = loss_vis.mean()

        # loss on masked branch
        if mim_loss_type == 'smooth_l1':
            loss_mim = nn.functional.smooth_l1_loss(pred, target, reduction='none', beta=2.0)
        else:
            loss_mim = (pred - target) ** 2
        loss_mim = loss_mim.mean(dim=-1)  # [N, L], mean loss per patch
        loss_mim = (loss_mim * mask).sum() / mask.sum()  # mean loss on removed patches

        return (loss_mim + theta * loss_vis)

    def forward(self, imgs, target, mask_ratio=0.75, mask_type='rand', mim_loss_type='l2', vis_loss_type='smooth_l1'):
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio, mask_type)
        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        loss = self.forward_loss(imgs, pred, target, mask, mim_loss_type, vis_loss_type)
        return loss, pred, mask


def dmjd_convit_base_patch16_dec512d2b_hog(**kwargs):
    model = DMJDConViT(
        img_size=[224, 56, 28], patch_size=[4, 2, 2], embed_dim=[256, 384, 768], depth=[2, 2, 11], num_heads=12,
        decoder_embed_dim=512, decoder_depth=2, decoder_num_heads=16,
        mlp_ratio=[8, 8, 4], norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def dmjd_convit_base_patch16_dec512d8b_hog(**kwargs):
    model = DMJDConViT(
        img_size=[224, 56, 28], patch_size=[4, 2, 2], embed_dim=[256, 384, 768], depth=[2, 2, 11], num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=[8, 8, 4], norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def dmjd_convit_large_patch16_dec512d8b_hog(**kwargs):
    model = DMJDConViT(
        img_size=[224, 56, 28], patch_size=[4, 2, 2], embed_dim=[384, 768, 1024], depth=[2, 2, 23], num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16, 
        mlp_ratio=[8, 8, 4], norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


# set recommended archs
dmjd_convit_base_patch16_dec2_hog = dmjd_convit_base_patch16_dec512d2b_hog
dmjd_convit_base_patch16_dec8_hog = dmjd_convit_base_patch16_dec512d8b_hog
dmjd_convit_large_patch16_dec8_hog = dmjd_convit_large_patch16_dec512d8b_hog
