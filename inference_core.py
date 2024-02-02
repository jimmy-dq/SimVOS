from os import name
import torch

from model.eval_network import STCN
from model.aggregate import aggregate

from util.tensor_util import pad_divide_by
from torch.nn import functional as F
from matplotlib import pyplot as plt
import cv2
import numpy as np

import time
import os

from model import models_vit
import timm
assert timm.__version__ == "0.3.2" # version check


def show_cam_on_image(img: np.ndarray,
                      mask: np.ndarray,
                      use_rgb: bool = False,
                      colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """ This function overlays the cam mask on the image as an heatmap.
    By default the heatmap is in BGR format.
    :param img: The base image in RGB or BGR format.
    :param mask: The cam mask.
    :param use_rgb: Whether to use an RGB or BGR heatmap, this should be set to True if 'img' is in RGB format.
    :param colormap: The OpenCV colormap to be used.
    :returns: The default image with the cam overlay.
    """
    # print('max')
    # print(colormap.max())
    heatmap = cv2.applyColorMap(np.uint8(255 * (mask/np.max(mask))), colormap)
    img = img/np.max(img)
    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255

    if np.max(img) > 1:
        raise Exception(
            "The input image should np.float32 in the range [0, 1]")

    cam = heatmap + img
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)


def interpolate_pos_embed_2D(pos_embed, kh, kw):
    
        num_extra_tokens = 1
        model_pos_tokens = pos_embed[:, num_extra_tokens:, :] 
        model_token_size = int((model_pos_tokens.shape[1])**0.5)
        # pos_embed = net.pos_embed
        model_pos_tokens = pos_embed[:, num_extra_tokens:(model_token_size*model_token_size + 1), :] # bs, N, C
        extra_pos_tokens = pos_embed[:, :num_extra_tokens]

        embedding_size = extra_pos_tokens.shape[-1]

        if kh != model_token_size or kw != model_token_size: # do interpolation
            model_pos_tokens_temp = model_pos_tokens.reshape(-1, model_token_size, model_token_size, embedding_size).contiguous().permute(0, 3, 1, 2) # bs, c, h, w
            search_pos_tokens = torch.nn.functional.interpolate(
                model_pos_tokens_temp, size=(kh, kw), mode='bicubic', align_corners=False)
            search_pos_tokens = search_pos_tokens.permute(0, 2, 3, 1).contiguous().flatten(1, 2)
        else:
            search_pos_tokens = model_pos_tokens
        new_pos_embed = torch.cat((extra_pos_tokens, search_pos_tokens), dim=1)
        return new_pos_embed

class InferenceCore_ViT:
    def __init__(self, prop_net:models_vit, images, num_objects, pos_embed_new, video_name=None):
        self.prop_net = prop_net

        # True dimensions
        t = images.shape[1]
        h, w = images.shape[-2:]

        # Pad each side to multiple of 16
        images, self.pad = pad_divide_by(images, 16)
        # Padded dimensions
        nh, nw = images.shape[-2:]

        self.images = images
        self.device = 'cuda'

        self.k = num_objects

        # Background included, not always consistent (i.e. sum up to 1)
        self.prob = torch.zeros((self.k+1, t, 1, nh, nw), dtype=torch.float32, device=self.device)
        self.prob[0] = 1e-7   # for the background 

        self.t, self.h, self.w = t, h, w
        self.nh, self.nw = nh, nw
        self.kh = self.nh//16
        self.kw = self.nw//16

        pos_embed_new_temp  = interpolate_pos_embed_2D(pos_embed_new, self.kh, self.kw)
        self.prop_net.pos_embed_new = torch.nn.Parameter(pos_embed_new_temp)

        print('after interpolation:')
        print(self.prop_net.pos_embed_new.shape)

        # self.mem_bank = MemoryBank(k=self.k, top_k=top_k)
        self.video_name = video_name
        print('init inference_core')

    def visualize(self, att_weights, video_name, mode, frame_index):
        base_path = '/apdcephfs/share_1290939/qiangqwu/VOS/DAVIS/2017/trainval/JPEGImages/480p/'+video_name+'/{:05d}.jpg'.format(frame_index)
        image = cv2.imread(base_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # W, H = 320, 320
        # image = cv2.resize(image, (W, H))
        H, W, C = image.shape
        # here we create the canvas
        fig = plt.figure(constrained_layout=True, figsize=(25 * 0.7, 25 * 0.7))
        # and we add one plot per reference point
        grid_len = 8
        # gs = fig.add_gridspec(grid_len+1, grid_len)
        gs = fig.add_gridspec(1, 1)
        axs = []
        axs.append(fig.add_subplot(gs[0, 0]))
        # axs.append(fig.add_subplot(gs[0, 1]))
        # for i in range(1, grid_len):
        #     for j in range(grid_len):
        #         axs.append(fig.add_subplot(gs[i, j]))
        # axs[0].imshow(image)
        # axs[0].axis('off')
        temp_att = cv2.resize(np.mean(att_weights, axis=0), (W, H))
        template_attention_map = show_cam_on_image(image, temp_att, use_rgb=True)
        axs[0].imshow(template_attention_map, cmap='cividis', interpolation='nearest')
        axs[0].axis('off')

        # axs = axs[2:]
        # token_index = 0
        # for ax in axs:
        #     temp_att = cv2.resize(att_weights[token_index], (W, H))
        #     template_attention_map = show_cam_on_image(image, temp_att, use_rgb=True)
        #     ax.imshow(template_attention_map, cmap='cividis', interpolation='nearest')
        #     ax.axis('off')
        #     # show_cam_on_image(cv2.resize(start_frame_img_rgb, (320, 320)), cv2.resize(dec_attn_weights_template, (320, 320)), use_rgb=True)
        #     # ax.imshow(cv2.resize(z_patch, (320, 320)))
        #     # ax.axis('off')
        #     token_index += 1
        if not os.path.exists("./visualization/"+self.video_name):
            os.makedirs("./visualization/"+self.video_name)
        plt.savefig("./visualization/"+self.video_name+"/"+mode+'_{:05d}.png'.format(frame_index))

    def do_pass(self, target_initial_mask, idx, end_idx, layer_index, use_local_window, use_token_learner):
        '''
        target_initial_mask: N, 1, H, W
        '''
        # self.mem_bank.add_memory(key_k, key_v)
        print('layer index: %d' %(layer_index))
        closest_ti = end_idx

        # Note that we never reach closest_ti, just the frame before it
        this_range = range(idx+1, closest_ti)
        end = closest_ti - 1
        
        target_models = []

        if layer_index > 0:
            # # tokenlearner w/ weights
            f1_v = self.prop_net(frames=self.images[:,idx].repeat(self.k, 1, 1, 1), mode='extract_feat_w_mask', layer_index=layer_index, mask_frames=target_initial_mask, use_window_partition=use_local_window, use_token_learner=use_token_learner) # N, dim, h, w
            if use_token_learner: 
                bg_mask = torch.ones_like(target_initial_mask[0].unsqueeze(0))  - torch.sum(target_initial_mask.permute(1, 0, 2, 3), dim=1, keepdim=True)
                bg_mask[bg_mask != 1] = 0
                init_mask = torch.cat((bg_mask, target_initial_mask.permute(1, 0, 2, 3)), dim=1)     # B, N+1, H, W
                h, w = f1_v.shape[-2:]
                props = F.interpolate(init_mask, size=(h, w), mode='bilinear', align_corners=False)  # B, N+1, H, W
                hard_probs = torch.zeros_like(props)
                max_indices = torch.max(props, dim=1, keepdim=True)[1]
                hard_probs.scatter_(dim=1, index=max_indices, value=1.0)
                hard_probs = hard_probs[:, 1:]  # B, N, H, W; The binary segmentation mask
                masks = torch.stack([props[:, 1:] * hard_probs, (1 - props[:, 1:]) * (1 - hard_probs)], dim=2)  # B, N, 2, H, W;
            for i in range(self.k):
                if use_token_learner:
                    # for visualizaiton
                    # rf_tokens, for_weight, back_weight, h_mask, w_mask = self.prop_net(qk16=f1_v[i].unsqueeze(0), mode='tokenlearner_w_masks', mask = masks[:,i])
                    # target_models.append(rf_tokens)

                    # for_weight = for_weight.view(1, -1, h_mask, w_mask)
                    # back_weight = back_weight.view(1, -1, h_mask, w_mask)

                    # self.visualize(for_weight.squeeze(0).cpu().data.numpy(), self.video_name, 'foreground_'+str(i), idx)
                    # self.visualize(back_weight.squeeze(0).cpu().data.numpy(), self.video_name, 'background_'+str(i), idx)

                    target_models.append(self.prop_net(qk16=f1_v[i].unsqueeze(0), mode='tokenlearner_w_masks', mask = masks[:,i])[0]) # B, num_token, c
                    
                else:
                    # print(f1_v[i].shape)
                    target_models.append(f1_v[i].unsqueeze(0)) # 1, C, H, W

        for ti in this_range:
            mask_list = []
            # s1 = time.time()

            if layer_index > 0:
                m16_f_ti = self.prop_net(frames=self.images[:,ti], mode='extract_feat_wo_mask', layer_index=layer_index, use_window_partition=use_local_window)   # 1, hw, c
                _, L, _ = m16_f_ti.shape
                
                for obj_index in range(self.k): # for different objects. here we can also use previous frames
                    # tokenlearner w/ weights; updating
                    if ti == 1:  #target_models[obj_index]
                        m16_f2_v_index, m8_f2_v_index, m4_f2_v_index = self.prop_net(template=target_models[obj_index], search=m16_f_ti, mode='forward_together', layer_index=layer_index, H=self.images[:,ti].shape[-2], W=self.images[:,ti].shape[-1], L=L)
                        out_mask = self.prop_net(m16=m16_f2_v_index, m8 = m8_f2_v_index, m4 = m4_f2_v_index, mode='segmentation_single_onject')
                        mask_list.append(out_mask)

                    else:
                        m16_f2_v_index, m8_f2_v_index, m4_f2_v_index = self.prop_net(template=torch.cat((target_models[obj_index], target_models_latest[obj_index]), dim=1), search=m16_f_ti, mode='forward_together', layer_index=layer_index, H=self.images[:,ti].shape[-2], W=self.images[:,ti].shape[-1], L=L)
                        out_mask = self.prop_net(m16=m16_f2_v_index, m8 = m8_f2_v_index, m4 = m4_f2_v_index, mode='segmentation_single_onject')
                        mask_list.append(out_mask)
            else:
                for obj_index in range(self.k): # for different objects. here we can also use previous frames
                    target_mask = target_initial_mask[obj_index].unsqueeze(0).unsqueeze(0) # 1,1,1, H, W

                    if ti == 1:
                        m16_f1_v1, m8_f1_v1, m4_f1_v1 = self.prop_net(memory_frames=self.images[:,idx].unsqueeze(1), mask_frames=target_mask,  query_frame=self.images[:,ti], mode='backbone_full')
                        out_mask = self.prop_net(m16=m16_f1_v1, m8 = m8_f1_v1, m4 = m4_f1_v1, mode='segmentation_single_onject')
                        mask_list.append(out_mask)
                    else:
                        m16_f1_v1, m8_f1_v1, m4_f1_v1 = self.prop_net(memory_frames=torch.cat((self.images[:,idx].unsqueeze(1), self.images[:,ti-1].unsqueeze(1)), dim=1), mask_frames=torch.cat((target_mask, self.prob[:,ti-1][obj_index+1].unsqueeze(0).unsqueeze(0)), dim=1),  query_frame=self.images[:,ti], mode='backbone_full')
                        out_mask = self.prop_net(m16=m16_f1_v1, m8 = m8_f1_v1, m4 = m4_f1_v1, mode='segmentation_single_onject')
                        mask_list.append(out_mask)
                        # m16_f1_v1, m8_f1_v1, m4_f1_v1 = self.prop_net(memory_frames=torch.cat((self.images[:,idx].unsqueeze(1), self.images[:,ti-1].unsqueeze(1)), dim=1), mask_frames=torch.cat((target_mask, (torch.argmax(self.prob[:,ti-1], dim=0)==(obj_index+1)).float().unsqueeze(0).unsqueeze(0)), dim=1),  query_frame=self.images[:,ti], mode='backbone')
            
            out_mask = torch.stack(mask_list, dim=0).flatten(0, 1) #num, 1, 1, h, w
            
            out_mask = aggregate(out_mask, keep_bg=True)
            self.prob[:,ti] = out_mask  # N+1, 1, H, W


            if layer_index > 0:

                # tokenlearner w/ latest frame
                if ti < end:
                    f_ti_v = self.prop_net(frames=self.images[:,ti].repeat(self.k, 1, 1, 1), mode='extract_feat_w_mask', layer_index=layer_index, mask_frames=out_mask[1:], use_window_partition=use_local_window, use_token_learner=use_token_learner) # N, dim, h, w
                    target_models_latest = []

                    if use_token_learner:
                        bg_mask = torch.ones_like(target_initial_mask[0].unsqueeze(0))  - torch.sum(out_mask[1:].permute(1, 0, 2, 3), dim=1, keepdim=True)
                        bg_mask[bg_mask != 1] = 0
                        init_mask = torch.cat((bg_mask, out_mask[1:].permute(1, 0, 2, 3)), dim=1)     # B, N+1, H, W
                        h, w = f_ti_v.shape[-2:]
                        props = F.interpolate(init_mask, size=(h, w), mode='bilinear', align_corners=False)  # B, N+1, H, W
                        hard_probs = torch.zeros_like(props)
                        max_indices = torch.max(props, dim=1, keepdim=True)[1]
                        hard_probs.scatter_(dim=1, index=max_indices, value=1.0)
                        hard_probs = hard_probs[:, 1:]  # B, N, H, W; The binary segmentation mask
                        masks = torch.stack([props[:, 1:] * hard_probs, (1 - props[:, 1:]) * (1 - hard_probs)], dim=2)  # B, N, 2, H, W;
                    for i in range(self.k):
                        if use_token_learner:
                            # rf_tokens, for_weight, back_weight, h_mask, w_mask = self.prop_net(qk16=f_ti_v[i].unsqueeze(0), mode='tokenlearner_w_masks', mask = masks[:,i])
                            # target_models_latest.append(rf_tokens)

                            # for_weight = for_weight.view(1, -1, h_mask, w_mask)
                            # back_weight = back_weight.view(1, -1, h_mask, w_mask)

                            # self.visualize(for_weight.squeeze(0).cpu().data.numpy(), self.video_name, 'foreground_'+str(i), ti)
                            # self.visualize(back_weight.squeeze(0).cpu().data.numpy(), self.video_name, 'background_'+str(i), ti)

                            target_models_latest.append(self.prop_net(qk16=f_ti_v[i].unsqueeze(0), mode='tokenlearner_w_masks', mask = masks[:,i])[0]) # B, num_token, c
                        else:
                            target_models_latest.append(f_ti_v[i].unsqueeze(0)) #.permute(0, 2, 3, 1).view(1, -1, 768)

        return closest_ti

    def interact(self, mask, frame_idx, end_idx, layer_index, use_local_window, use_token_learner):
        mask, _ = pad_divide_by(mask.cuda(), 16) # 2, 1, 480, 912

        self.prob[:, frame_idx] = aggregate(mask, keep_bg=True) # the 1st frame

        # Propagate
        self.do_pass(mask, frame_idx, end_idx, layer_index, use_local_window, use_token_learner)
