# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial
from tkinter.messagebox import NO

import torch
import torch.nn as nn

import timm.models.vision_transformer
from timm.models.layers.helpers import *
from timm.models.vision_transformer import PatchEmbed
from model.network import Decoder
import torch.nn.functional as F
import time
import math
import torch.nn.functional as NF


def l2norm(inp, dim):
    norm = torch.linalg.norm(inp, dim=dim, keepdim=True) + 1e-6
    return inp/norm


def window_partition(x, window_size):
    """
    Partition into non-overlapping windows with padding if needed.
    Args:
        x (tensor): input tokens with [B, H, W, C].
        window_size (int): window size.

    Returns:
        windows: windows after partition with [B * num_windows, window_size, window_size, C].
        (Hp, Wp): padded height and width before partition
    """
    B, H, W, C = x.shape

    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w

    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows, (Hp, Wp)


def window_unpartition(windows, window_size, pad_hw, hw):
    """
    Window unpartition into original sequences and removing padding.
    Args:
        x (tensor): input tokens with [B * num_windows, window_size, window_size, C].
        window_size (int): window size.
        pad_hw (Tuple): padded height and width (Hp, Wp).
        hw (Tuple): original height and width (H, W) before padding.

    Returns:
        x: unpartitioned sequences with [B, H, W, C].
    """
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    x = windows.view(B, Hp // window_size, Wp // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)

    if Hp > H or Wp > W:
        x = x[:, :H, :W, :].contiguous()
    return x



# class TokenLearnerModuleV11(nn.Module):
#     """TokenLearner module Version 1.1, using slightly different conv. layers.
#     Instead of using 4 conv. layers with small channels to implement spatial
#     attention, this version uses 2 grouped conv. layers with more channels. It
#     also uses softmax instead of sigmoid. We confirmed that this version works
#     better when having limited training data, such as training with ImageNet1K
#     from scratch.
#     Attributes:
#       num_tokens: Number of tokens.
#       dropout_rate: Dropout rate.
#     """

#     def __init__(self, in_channels, num_tokens):
#         """Applies learnable tokenization to the 2D inputs.
#         Args:
#           inputs: Inputs of shape `[bs, h, w, c]`.
#         Returns:
#           Output of shape `[bs, n_token, c]`.
#         """
#         super(TokenLearnerModuleV11, self).__init__()
#         self.in_channels = in_channels
#         self.num_tokens = num_tokens
#         self.norm = nn.LayerNorm(self.in_channels)  # Operates on the last axis (c) of the input data.

#         # use a patch-to-cluster architecture
#         self.patch_to_cluster_atten = nn.Sequential(
#             nn.Conv2d(self.in_channels, self.in_channels // 4, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
#             nn.GELU()
#         )
#         self.patch_to_cluster_linear = nn.Linear(in_features=self.in_channels // 4, out_features=self.num_tokens)


#     # def forward(self, inputs, mask, pos, offset_input=None):
#     def forward(self, inputs, weights):
#         '''
#         inputs: bs, h, w, c
#         weights: bs, 2, h, w
#         '''

#         feature_shape = inputs.shape  # Shape:  [bs, h, w, c]
        
#         # use the appearance features for attention map generation
#         selected = inputs
#         selected = selected.permute(0, 3, 1, 2)  # Shape:  [bs, c, h, w]
#         selected = self.patch_to_cluster_atten(selected)  # Shape: [bs, c_dim, h, w].
#         selected = selected.permute(0, 2, 3, 1)  # Shape: [bs, h, w, c_dim].
#         selected = selected.contiguous().view(feature_shape[0], feature_shape[1] * feature_shape[2], -1)  # Shape: [bs, h*w, c_dim].
#         selected = self.patch_to_cluster_linear(selected)  # Shape: [bs, h*w, n_token].

#         selected = selected.permute(0, 2, 1)  # Shape: [bs, n_token, h*w].
#         # selected = F.softmax(selected, dim=-1)  # bs, n_token, hw

#         num_for_tokens  = self.num_tokens // 2
#         num_back_tokens = self.num_tokens // 2 

#         for_weights  = weights.flatten(2, 3)[:, 0].unsqueeze(1)  # bs, 1, hw
#         back_weights = weights.flatten(2, 3)[:, 1].unsqueeze(1) # bs, 1, hw

#         combined_weights = torch.cat((for_weights.repeat(1, num_for_tokens, 1), back_weights.repeat(1, num_back_tokens, 1)), dim=1) # bs, n_token, hw
#         selected[combined_weights == 0] = -float('inf')
#         selected = F.softmax(selected, dim=-1)  # bs, n_token, hw
#         selected = torch.nan_to_num(selected)   # replace nan to 0.0, especially for selector = [1, 0], i.e., only has one object, sec_obj is empty

#         # selected = selected * torch.cat((for_weights.repeat(1, num_for_tokens, 1), back_weights.repeat(1, num_back_tokens, 1)), dim=1) # bs, n_token, hw
#         # sum_weights = torch.sum(selected, dim=-1).unsqueeze(-1) # bs, n_token, 1
        
#         feat = inputs #bs, h, w, c
#         feat = feat.contiguous().view(feature_shape[0], feature_shape[1] * feature_shape[2], -1)  # Shape: [bs, h*w, c].

#         # Produced the attended inputs.
#         outputs = torch.einsum("...si,...id->...sd",  selected, feat)  # (B, n_token, c)
#         # outputs = outputs / (sum_weights + 1e-20)
#         outputs = self.norm(outputs)

#         return outputs



class TokenLearnerModuleV11_w_Mask(nn.Module):
    """TokenLearner module Version 1.1, using slightly different conv. layers.
    Instead of using 4 conv. layers with small channels to implement spatial
    attention, this version uses 2 grouped conv. layers with more channels. It
    also uses softmax instead of sigmoid. We confirmed that this version works
    better when having limited training data, such as training with ImageNet1K
    from scratch.
    Attributes:
      num_tokens: Number of tokens.
      dropout_rate: Dropout rate.
    """

    def __init__(self, in_channels, num_tokens):
        """Applies learnable tokenization to the 2D inputs.
        Args:
          inputs: Inputs of shape `[bs, h, w, c]`.
        Returns:
          Output of shape `[bs, n_token, c]`.
        """
        super(TokenLearnerModuleV11_w_Mask, self).__init__()
        self.in_channels = in_channels
        self.num_tokens = num_tokens
        self.norm = nn.LayerNorm(self.in_channels)  # Operates on the last axis (c) of the input data.
        self.input_norm = nn.LayerNorm(self.in_channels + 1)

        # Input: appearance features + 1-channel mask
        self.patch_to_cluster_atten = nn.Sequential(
            nn.Conv2d(self.in_channels + 1, self.in_channels // 4, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
            nn.GELU()
        )
        self.patch_to_cluster_linear = nn.Linear(in_features=self.in_channels // 4, out_features=self.num_tokens)


    def forward(self, inputs, weights):
        '''
        inputs: bs, h, w, c
        weights: bs, 1, h, w
        '''

        feature_shape = inputs.shape  # Shape:  [bs, h, w, c]
        
        # use the appearance features for attention map generation
        selected = torch.cat((inputs, weights.permute(0, 2, 3, 1)), dim=-1) # bs, h, w, c+1 
        selected = self.input_norm(selected) # need this norm here? shape: bs, h, w, c+1 
        selected = selected.permute(0, 3, 1, 2)  # Shape:  [bs, c+1, h, w]
        selected = self.patch_to_cluster_atten(selected)  # Shape: [bs, c_dim, h, w].
        selected = selected.permute(0, 2, 3, 1)  # Shape: [bs, h, w, c_dim].
        selected = selected.contiguous().view(feature_shape[0], feature_shape[1] * feature_shape[2], -1)  # Shape: [bs, h*w, c_dim].
        selected = self.patch_to_cluster_linear(selected)  # Shape: [bs, h*w, n_token//2].

        selected = selected.permute(0, 2, 1)  # Shape: [bs, n_token//2, h*w].
        # selected = F.softmax(selected, dim=-1)  # bs, n_token, hw

        num_for_tokens  = self.num_tokens

        for_weights  = weights.flatten(2, 3)  # bs, 1, hw

        # # mask map guided suppresion
        combined_weights = for_weights.repeat(1, num_for_tokens, 1) # bs, n_token//2, hw
        selected[combined_weights == 0] = -float('inf')

        selected = F.softmax(selected, dim=-1)  # bs, n_token, hw
        selected = torch.nan_to_num(selected)  # replace nan to 0.0, especially for selector = [1, 0], i.e., only has one object, sec_obj is empty

        # selected = selected * torch.cat((for_weights.repeat(1, num_for_tokens, 1), back_weights.repeat(1, num_back_tokens, 1)), dim=1) # bs, n_token, hw
        # sum_weights = torch.sum(selected, dim=-1).unsqueeze(-1) # bs, n_token, 1
    
        feat = inputs #bs, h, w, c
        feat = feat.contiguous().view(feature_shape[0], feature_shape[1] * feature_shape[2], -1)  # Shape: [bs, h*w, c].

        # Produced the attended inputs.
        outputs = torch.einsum("...si,...id->...sd",  selected, feat)  # (B, n_token//2, c)
        outputs = self.norm(outputs)

        return outputs, selected

# borrowed from https://github.com/ViTAE-Transformer/ViTDet/blob/main/mmdet/models/backbones/vitae.py
class Norm2d(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.ln = nn.LayerNorm(embed_dim, eps=1e-6)
    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.ln(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x


class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, global_pool=False, single_object=False, num_bases_foreground=None, num_bases_background=None, img_size=None, vit_dim=None, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)

        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)
            self.single_object = single_object

        self.vit_dim = vit_dim


        self.mask_patch_embed = PatchEmbed(
                img_size=img_size, patch_size=16, in_chans=1, embed_dim=vit_dim) # !!! to check whether it has grads
        
        # borrowed from https://github.com/ViTAE-Transformer/ViTDet/blob/main/mmdet/models/backbones/vitae.py
        self.fpn1 = nn.Sequential(  # 1/4
            nn.ConvTranspose2d(vit_dim, vit_dim, kernel_size=2, stride=2),
            Norm2d(vit_dim),
            nn.GELU(),
            nn.ConvTranspose2d(vit_dim, 256, kernel_size=2, stride=2),
        )

        self.fpn2 = nn.Sequential(   # 1/8
            nn.ConvTranspose2d(vit_dim, 512, kernel_size=2, stride=2),
        )

        self.stcn_decoder = Decoder(vit_dim=vit_dim)

        self.num_bases_foreground = num_bases_foreground
        self.num_bases_background = num_bases_background

        print('num_bases_foreground: %d' %(self.num_bases_foreground))
        print('num_bases_background: %d' %(self.num_bases_background))

        self.tlearner = TokenLearnerModuleV11_w_Mask(in_channels=vit_dim, num_tokens=self.num_bases_foreground)
        self.tlearner_back = TokenLearnerModuleV11_w_Mask(in_channels=vit_dim, num_tokens=self.num_bases_background)
        

    def aggregate(self, prob):
        new_prob = torch.cat([
            torch.prod(1-prob, dim=1, keepdim=True),  # get the background region
            prob
        ], 1).clamp(1e-7, 1-1e-7)
        logits = torch.log((new_prob /(1-new_prob))) # bs, 3, 384, 384
        return logits


    def forward(self, mode=None, **kwargs): #memory_frames=None, mask_frames=None,  query_frame=None, mode=None, selector=None):
        '''
        memory_frames: bs, T, 3, 384, 384
        mask_frames: bs, T, 3, 384, 384
        query_frame: bs, 3, 384, 384
        '''
        if mode == 'extract_feat_w_mask':
            frames = kwargs['frames']  #B, C=3, H, W
            mask_frames = kwargs['mask_frames'] #B, 1, H, W
            layer_index = kwargs['layer_index']
            use_window_partition = kwargs['use_window_partition']
            use_token_learner = kwargs['use_token_learner']
            # local attention for feature extraction
            x = self.patch_embed(frames)    # bs*T, (H//16 * W//16), 768
            B, _, H, W = frames.shape
            _, _, C = x.shape
            mask_tokens   = self.mask_patch_embed(mask_frames)
            x = x + self.pos_embed_new[:, 1:, :] + mask_tokens
            x = self.pos_drop(x)   #bs, (T+1)*hw, C

            bs = frames.shape[0]

            # if use_window_partition, perform the window_partition 
            if use_window_partition:
                # local-in-local attention
                H = frames.shape[-2] // 16
                W = frames.shape[-1] // 16
                dim = x.shape[-1]
                x = x.view(bs, H, W, dim)
                window_size = H // 2
                x, pad_hw = window_partition(x, window_size) # x: bs*N, window_size, window_size, c
                x = x.view(-1, window_size*window_size, dim)
            # token interaction in early layers of ViT
            for blk in self.blocks[0:layer_index]:
                x = blk(x)
            
            # if use window_partition
            if use_window_partition:
                # local-in-local attention recover
                x = window_unpartition(x, window_size, pad_hw, (H, W)) #x: bs, H, W, c
                if use_token_learner:
                    return x.permute(0, 3, 1, 2) #.view(B, C, int(H//16), int(W//16))
                else:
                    return x.view(bs, -1, dim) #(b, hw, c)
            else:
                if use_token_learner:
                    return x.view(bs, H//16, W//16, x.shape[-1]).permute(0, 3, 1, 2) # (b, c, h, w)
                else:
                    return x # (b, hw, c)
        elif mode == 'backbone_full':
            memory_frames = kwargs['memory_frames']
            mask_frames = kwargs['mask_frames']
            query_frame = kwargs['query_frame']
            B, T, C, H, W = memory_frames.shape
            
            memory_frames = memory_frames.flatten(0, 1)
            mask_frames = mask_frames.flatten(0, 1)
            memory_tokens = self.patch_embed(memory_frames)    # bs*T, (H//16 * W//16), 768
            mask_tokens   = self.mask_patch_embed(mask_frames) # bs*T, (H//16 * W//16), 768
            # add the target-aware positional encoding
            memory_tokens = memory_tokens + mask_tokens
            query_tokens = self.patch_embed(query_frame) # bs, (H//16 * W//16), 768
            
            if T > 1: # multiple memory frames
                memory_tokens = memory_tokens.view(B, T, -1, memory_tokens.size()[-1]).contiguous() #bs ,T, num, C
                # use all the memory frames
                memory_tokens = memory_tokens.flatten(1, 2) # bs ,total_num, C
                
            x = torch.cat((memory_tokens, query_tokens), dim=1)
            if T > 1:
                single_size = int((self.pos_embed_new[:, 1:, :].shape[1]))
                x = x + self.pos_embed_new[:, 1:(single_size+1), :].repeat(1, T+1, 1)
            else:
                x = x + self.pos_embed_new[:, 1:, :].repeat(1, 2, 1) # 2 frames
            x = self.pos_drop(x)
            for blk in self.blocks:
                x = blk(x)
            
            # maybe we need the norm(x), improves the results!
            x = self.norm(x)

            num_query_tokens = query_tokens.shape[1]
            updated_query_tokens = x[:, -num_query_tokens:, :]
            updated_query_tokens = updated_query_tokens.permute(0, 2, 1).contiguous().view(B, self.vit_dim, int(H//16), int(W//16))
            m16 =  updated_query_tokens             # bs, 768, 24, 24
            m8  =  self.fpn2(updated_query_tokens)  # bs, 512, 48, 48
            m4  =  self.fpn1(updated_query_tokens)  # bs, 256, 96, 96

            return m16, m8, m4
        elif mode == 'extract_feat_wo_mask':
            frames = kwargs['frames']
            layer_index = kwargs['layer_index']
            use_window_partition = kwargs['use_window_partition']
            x = self.patch_embed(frames)
            x = x + self.pos_embed_new[:, 1:, :]
            x = self.pos_drop(x)
            
            if use_window_partition:
                # local-in-local attention
                H = frames.shape[-2] // 16
                W = frames.shape[-1] // 16
                bs = frames.shape[0]
                dim = x.shape[-1]
                x = x.view(bs, H, W, dim)
                window_size = H // 2
                x, pad_hw = window_partition(x, window_size) # x: bs*N, window_size, window_size, c
                x = x.view(-1, window_size*window_size, dim)
            for blk in self.blocks[0:layer_index]:
                x = blk(x)
            
            # local-in-local attention recover
            if use_window_partition:
                x = window_unpartition(x, window_size, pad_hw, (H, W)) #x: bs, H, W, c
                x = x.view(bs, -1, dim)
                return x  # bs, hw, c
            else:
                return x
        elif mode == 'forward_together':
            template = kwargs['template']
            search = kwargs['search']
            layer_index = kwargs['layer_index']
            H = kwargs['H']
            W = kwargs['W']
            L = kwargs['L']
            x = torch.cat((template, search), dim=1)
            bs, _, _ = x.shape
            for blk in self.blocks[layer_index:]:
                x = blk(x)
            # do the normalization for the output
            x = self.norm(x)
            updated_query_tokens = x[:, (-L):, :]
            updated_query_tokens = updated_query_tokens.permute(0, 2, 1).contiguous().view(bs, self.vit_dim, int(H//16), int(W//16))
            m16 =  updated_query_tokens             # bs, 768, 24, 24
            m8  =  self.fpn2(updated_query_tokens)  # bs, 512, 48, 48
            m4  =  self.fpn1(updated_query_tokens)  # bs, 256, 96, 96
            return m16, m8, m4 #, att_list
        elif mode == 'extract_feat_in_later_layer':
            x = kwargs['x']
            H = kwargs['H']
            W = kwargs['W']
            L = kwargs['L']
            bs, _, _ = x.shape

            iden_embed = torch.cat((self.pos_iden.repeat(1, self.num_bases_foreground, 1), self.neg_iden.repeat(1, self.num_bases_background, 1)), dim=1).repeat(bs, 1, 1)
            x[:, 0:L, :] = x[:, 0:L, :] + iden_embed
            layer_index = kwargs['layer_index']
            for blk in self.blocks[layer_index:]:
                x = blk(x)

            updated_query_tokens = x[:, L:, :]
            updated_query_tokens = updated_query_tokens.permute(0, 2, 1).contiguous().view(bs, self.vit_dim, int(H//16), int(W//16))
            m16 =  updated_query_tokens             # bs, 768, 24, 24
            m8  =  self.fpn2(updated_query_tokens)  # bs, 512, 48, 48
            m4  =  self.fpn1(updated_query_tokens)  # bs, 256, 96, 96
            return m16, m8, m4 #, att_list
        elif mode == 'extract_feat_in_later_layer_test': # for inference
            x = kwargs['x']
            H = kwargs['H']
            W = kwargs['W']
            L = kwargs['L']
            # att_list = []
            bs, _, _ = x.shape

            iden_embed = torch.cat((self.pos_iden.repeat(1, self.num_bases_foreground, 1), self.neg_iden.repeat(1, self.num_bases_background, 1)), dim=1).repeat(bs, 1, 1)
            x[:, 0:L, :] = x[:, 0:L, :] + iden_embed
            x[:, L:(2*L), :] = x[:, L:(2*L), :] + iden_embed
            layer_index = kwargs['layer_index']
            for blk in self.blocks[layer_index:]:
                x, attn = blk(x)
                # att_list.append(attn)
            updated_query_tokens = x[:, (2*L):, :]
            updated_query_tokens = updated_query_tokens.permute(0, 2, 1).contiguous().view(bs, self.vit_dim, int(H//16), int(W//16))
            m16 =  updated_query_tokens             # bs, 768, 24, 24
            m8  =  self.fpn2(updated_query_tokens)  # bs, 512, 48, 48
            m4  =  self.fpn1(updated_query_tokens)  # bs, 256, 96, 96
            return m16, m8, m4 #, att_list
        elif mode == 'extract_feat_in_later_layers_w_memory_bank':
            x1 = kwargs['x1'] # first frame 1, 512, c
            x2 = kwargs['x2'] # pos mem bank 1, 2048, c
            x3 = kwargs['x3'] # neg mem bank 1, 2048, c
            x4 = kwargs['x4'] # context features 
            H = kwargs['H']
            W = kwargs['W']
            L = kwargs['L']
            att_list = []
            bs, num_pos_mem, _ = x2.shape
            bs, num_neg_mem, _ = x3.shape

            x2 = x2 + self.pos_iden.repeat(1, num_pos_mem, 1)
            x3 = x3 + self.neg_iden.repeat(1, num_neg_mem, 1)
            if x1 is not None:
                x1 = x1 + torch.cat((self.pos_iden.repeat(1, self.num_bases_foreground, 1), self.neg_iden.repeat(1, self.num_bases_background, 1)), dim=1)
                x = torch.cat((x1, x2, x3, x4), dim=1)
            else:
                x = torch.cat((x2, x3, x4), dim=1)
            layer_index = kwargs['layer_index']
            for blk in self.blocks[layer_index:]:
                x, attn = blk(x)
                if x1 is not None:
                    att_list.append(torch.max(torch.mean(attn,dim=1)[:,-(x4.shape[-2]):][:,:,x1.shape[1]:(x1.shape[1]+num_pos_mem+num_neg_mem)],dim=1).values)
                else:
                    att_list.append(torch.max(torch.mean(attn,dim=1)[:,-(x4.shape[-2]):][:,:,0:(num_pos_mem+num_neg_mem)],dim=1).values)
            updated_query_tokens = x[:, -(x4.shape[-2]):, :]
            updated_query_tokens = updated_query_tokens.permute(0, 2, 1).contiguous().view(bs, self.vit_dim, int(H//16), int(W//16))
            m16 =  updated_query_tokens             # bs, 768, 24, 24
            m8  =  self.fpn2(updated_query_tokens)  # bs, 512, 48, 48
            m4  =  self.fpn1(updated_query_tokens)  # bs, 256, 96, 96
            return m16, m8, m4, att_list
        elif mode == 'tokenlearner_w_masks':
            # qk16 = kwargs['qk16'] #bs, c, h, w
            # qk16 = qk16.permute(0, 2, 3, 1) #bs, h, w, c
            # mask = kwargs['mask'] #bs, 2, h, w
            # qk16 = qk16.permute(0, 2, 3, 1)

            qk16 = kwargs['qk16'] #bs, c, h, w
            mask = kwargs['mask'] #bs, 2, h, w
            # qk16 = torch.cat((qk16, mask), dim=1) #bs, c+2, h, w
            qk16 = qk16.permute(0, 2, 3, 1) #bs, h, w, c
            
            # rf_tokens = self.tlearner(qk16, mask)
            fore_tokens, fore_att = self.tlearner(qk16, mask[:, 0].unsqueeze(1))
            back_tokens, back_att = self.tlearner_back(qk16, mask[:, 1].unsqueeze(1)) 
            # fore_tokens = self.tlearner(qk16, mask[:, 0].unsqueeze(1)) 
            # back_tokens = self.tlearner_back(qk16, mask[:, 1].unsqueeze(1))
            rf_tokens = torch.cat((fore_tokens, back_tokens), dim=1) #bs, num_token, c
            return rf_tokens, fore_att, back_att, mask.shape[-2], mask.shape[-1]
        elif mode == 'segmentation':
            # print('decoder for segmentation')
            m16 = kwargs['m16']
            m8 = kwargs['m8']
            m4 = kwargs['m4']
            selector = kwargs['selector']

            # m16=m16, m8=m8, m4=m4, selector=selector
            if self.single_object:
                logits = self.decoder(m16, m8, m4)
                prob = torch.sigmoid(logits)
            else:
                
                #self.memory.readout(affinity, mv16[:,0], qv16): 4, 1024, 24, 24 
                # qf8: 4, 512, 48, 48; qf4: 4, 256, 96, 96;
                logits = torch.cat([
                    self.stcn_decoder(m16[:,0], m8[:,0], m4[:,0]), 
                    self.stcn_decoder(m16[:,1], m8[:,1], m4[:,1]), 
                ], 1)  # 4, 2, 384, 384

                prob = torch.sigmoid(logits) # 4, 2, 384, 384; 2: two targets
                prob = prob * selector.unsqueeze(2).unsqueeze(2) # 4, 2, 384, 384 

            logits = self.aggregate(prob)
            prob = F.softmax(logits, dim=1)[:, 1:]
            return logits, prob, F.softmax(logits, dim=1) # for memorize
        elif mode == 'segmentation_single_onject':
            m16 = kwargs['m16']
            m8 = kwargs['m8']
            m4 = kwargs['m4']
            logits = self.stcn_decoder(m16, m8, m4)
            return torch.sigmoid(logits)
    
    def forward_patch_embedding(self, x):
        return self.patch_embed(x)

    def forward_features_testing(self, x, z):
            B = x.shape[0]
            # cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
            # x = torch.cat((cls_tokens, x, z), dim=1)
            x = torch.cat((x, z), dim=1)
            x = x + self.pos_embed
            x = self.pos_drop(x)

            for blk in self.blocks:
                x = blk(x)

            # if self.global_pool:  # mae use the global_pool instead of cls token for classficaition
            #     x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            #     outcome = self.fc_norm(x)
            # else:
            #     x = self.norm(x)
            #     outcome = x[:, 0]

            search_tokens = z.shape[1]

            return x[:, -search_tokens:, :]


def vit_base_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_large_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_huge_patch14(**kwargs):
    model = VisionTransformer(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

    
