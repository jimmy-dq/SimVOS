import torch

from inference_memory_bank import MemoryBank
from model.eval_network import STCN
from model.aggregate import aggregate

from util.tensor_util import pad_divide_by
from torch.nn import functional as F


class InferenceCore:
    def __init__(self, prop_net:STCN, images, num_objects, top_k=20, 
                    mem_every=5, include_last=False, req_frames=None):
        self.prop_net = prop_net
        self.mem_every = mem_every
        self.include_last = include_last

        # We HAVE to get the output for these frames
        # None if all frames are required
        self.req_frames = req_frames

        self.top_k = top_k

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
        self.prob[0] = 1e-7

        self.t, self.h, self.w = t, h, w
        self.nh, self.nw = nh, nw
        self.kh = self.nh//16
        self.kw = self.nw//16

        # list of objects with usable memory
        self.enabled_obj = []

        self.mem_banks = dict()

    def encode_key(self, idx):
        result = self.prop_net.encode_key(self.images[:,idx].cuda())
        return result

    def do_pass(self, key_k, key_v, idx, end_idx):
        closest_ti = end_idx

        K, CK, _, H, W = key_k.shape
        _, CV, _, _, _ = key_v.shape

        for i, oi in enumerate(self.enabled_obj):
            if oi not in self.mem_banks:
                self.mem_banks[oi] = MemoryBank(k=1, top_k=self.top_k)
            self.mem_banks[oi].add_memory(key_k, key_v[i:i+1])

        last_ti = idx

        # Note that we never reach closest_ti, just the frame before it
        this_range = range(idx+1, closest_ti)
        step = +1
        end = closest_ti - 1

        for ti in this_range: 
            is_mem_frame = (abs(ti-last_ti) >= self.mem_every)
            # Why even work on it if it is not required for memory/output
            if (not is_mem_frame) and (not self.include_last) and (self.req_frames is not None) and (ti not in self.req_frames):
                continue

            k16, qv16, qf16, qf8, qf4 = self.encode_key(ti)

            # After this step all keys will have the same size
            out_mask = torch.cat([
                self.prop_net.segment_with_query(self.mem_banks[oi], qf8, qf4, k16, qv16)
            for oi in self.enabled_obj], 0)

            out_mask = aggregate(out_mask, keep_bg=True)
            self.prob[0,ti] = out_mask[0]
            for i, oi in enumerate(self.enabled_obj):
                self.prob[oi,ti] = out_mask[i+1]

            if ti != end:
                if self.include_last or is_mem_frame:
                    prev_value = self.prop_net.encode_value(self.images[:,ti].cuda(), qf16, out_mask[1:])
                    prev_key = k16.unsqueeze(2)
                    for i, oi in enumerate(self.enabled_obj):
                        self.mem_banks[oi].add_memory(prev_key, prev_value[i:i+1], is_temp=not is_mem_frame)

                    if is_mem_frame:
                        last_ti = ti

        return closest_ti

    def interact(self, mask, frame_idx, end_idx, obj_idx):
        # In youtube mode, we interact with a subset of object id at a time
        mask, _ = pad_divide_by(mask.cuda(), 16)

        # update objects that have been labeled
        self.enabled_obj.extend(obj_idx)

        # Set other prob of mask regions to zero
        mask_regions = (mask[1:].sum(0) > 0.5)
        self.prob[:, frame_idx, mask_regions] = 0
        self.prob[obj_idx, frame_idx] = mask[obj_idx]

        self.prob[:, frame_idx] = aggregate(self.prob[1:, frame_idx], keep_bg=True)

        # KV pair for the interacting frame
        key_k, _, qf16, _, _ = self.encode_key(frame_idx)
        key_v = self.prop_net.encode_value(self.images[:,frame_idx].cuda(), qf16, self.prob[self.enabled_obj,frame_idx].cuda())
        key_k = key_k.unsqueeze(2)

        # Propagate
        self.do_pass(key_k, key_v, frame_idx, end_idx)


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
    def __init__(self, prop_net:STCN, images, pos_embed_new, req_frames, valid_only, mem_len, num_objects):
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
        self.prob[0] = 1e-7

        self.t, self.h, self.w = t, h, w
        self.nh, self.nw = nh, nw
        self.kh = self.nh//16
        self.kw = self.nw//16

        # list of objects with usable memory
        self.enabled_obj = []

        self.mem_masks = dict()
        self.mem_frames = dict()

        pos_embed_new_temp  = interpolate_pos_embed_2D(pos_embed_new, self.kh, self.kw)
        self.prop_net.pos_embed_new = torch.nn.Parameter(pos_embed_new_temp)

        print('after interpolation:')
        print(self.prop_net.pos_embed_new.shape)

        self.req_frames = req_frames
        self.valid_only = valid_only

        self.mem_len = mem_len # including the 1st frame, so the baseline self.mem_len=2


    def do_pass(self, idx, end_idx, mask, obj_idx, layer_index, use_local_window, use_token_learner):
        closest_ti = end_idx
        '''
        mask: N+1, 1, H, W
        '''

        # for i, oi in enumerate(self.enabled_obj):
        #     if oi not in self.mem_masks: # newly added labeled objects
        #         self.mem_masks[oi] = [self.prob[oi,idx].unsqueeze(0).unsqueeze(0)] # B, T, C, H, W
        #         self.mem_frames[oi] = [idx]
        #     else: # if exists; update the memory bank
        #         if len(self.mem_masks[oi]) == 1:
        #             self.mem_frames[oi].append(idx)
        #             self.mem_masks[oi].append(self.prob[oi,idx].unsqueeze(0).unsqueeze(0))
        #         else:
        #             self.mem_frames[oi][1] = idx
        #             self.mem_masks[oi][1] = self.prob[oi,idx].unsqueeze(0).unsqueeze(0)
        
        for i, oi in enumerate(self.enabled_obj):
            if oi not in self.mem_masks: # newly added labeled objects
                # self.mem_masks[oi] = [self.prob[oi,idx].unsqueeze(0).unsqueeze(0)] # B, T, C, H, W
                # self.mem_frames[oi] = [idx]
                if layer_index == 0:  # we do not use any within-frame attention and tokenlearner
                    self.mem_masks[oi] = [mask[oi].unsqueeze(0).unsqueeze(0)]  # 1, 1, h, w
                    self.mem_frames[oi] = [idx]
                else:
                    self.mem_frames[oi] = [idx]
                    # we firstly extract mask-aware features in the previous layer_index frames
                    print('mask shape')
                    print(mask[oi].shape)

                    f1_v = self.prop_net(frames=self.images[:,idx], mode='extract_feat_w_mask', layer_index=layer_index, mask_frames=mask[oi].unsqueeze(0), use_window_partition=use_local_window, use_token_learner=use_token_learner) 
                    if use_token_learner:
                        init_mask = mask.permute(1, 0, 2, 3) # B, N+1, H, W    # mask: N+1, 1 H, W
                        h, w = f1_v.shape[-2:]
                        props = F.interpolate(init_mask, size=(h, w), mode='bilinear', align_corners=False)  # B, N+1, H, W
                        hard_probs = torch.zeros_like(props)
                        max_indices = torch.max(props, dim=1, keepdim=True)[1]
                        hard_probs.scatter_(dim=1, index=max_indices, value=1.0)
                        hard_probs = hard_probs[:, 1:]  # B, N, H, W; The binary segmentation mask
                        converted_masks = torch.stack([props[:, 1:] * hard_probs, (1 - props[:, 1:]) * (1 - hard_probs)], dim=2)  # B, N, 2, H, W;

                        self.mem_masks[oi] = [self.prop_net(qk16=f1_v, mode='tokenlearner_w_masks', mask = converted_masks[:,i])[0]] # B, num_token, c
                    else:
                        self.mem_masks[oi] = [f1_v] # 1, C, H, W

            else: # if exists; update the memory bank
                if len(self.mem_masks[oi]) < self.mem_len:
                    if layer_index == 0:
                        self.mem_frames[oi].append(idx)
                        self.mem_masks[oi].append(self.prob[oi,idx].unsqueeze(0).unsqueeze(0))
                    else:
                        self.mem_frames[oi].append(idx)
                        f1_v = self.prop_net(frames=self.images[:,idx], mode='extract_feat_w_mask', layer_index=layer_index, mask_frames=self.prob[oi,idx].unsqueeze(0), use_window_partition=use_local_window, use_token_learner=use_token_learner) 
                        if use_token_learner:
                            init_mask = self.prob[:,idx].permute(1,0,2,3) #1, N+1, H, W
                            h, w = f1_v.shape[-2:]
                            props = F.interpolate(init_mask, size=(h, w), mode='bilinear', align_corners=False)  # B, N+1, H, W
                            hard_probs = torch.zeros_like(props)
                            max_indices = torch.max(props, dim=1, keepdim=True)[1]
                            hard_probs.scatter_(dim=1, index=max_indices, value=1.0)
                            hard_probs = hard_probs[:, 1:]  # B, N, H, W; The binary segmentation mask
                            converted_masks = torch.stack([props[:, 1:] * hard_probs, (1 - props[:, 1:]) * (1 - hard_probs)], dim=2)  # B, N, 2, H, W;
                            self.mem_masks[oi].append(self.prop_net(qk16=f1_v, mode='tokenlearner_w_masks', mask = converted_masks[:,i])[0]) # B, num_token, c
                        else:
                            self.mem_masks[oi].append(f1_v)
                else: 
                    del self.mem_frames[oi][1]
                    del self.mem_masks[oi][1]
                    if layer_index == 0:
                        self.mem_frames[oi].append(idx)
                        self.mem_masks[oi].append(self.prob[oi,idx].unsqueeze(0).unsqueeze(0))
                    else:
                        self.mem_frames[oi].append(idx)
                        f1_v = self.prop_net(frames=self.images[:,idx], mode='extract_feat_w_mask', layer_index=layer_index, mask_frames=self.prob[oi,idx].unsqueeze(0), use_window_partition=use_local_window, use_token_learner=use_token_learner) 
                        if use_token_learner:
                            init_mask = self.prob[:,idx].permute(1,0,2,3) #1, N+1, H, W
                            h, w = f1_v.shape[-2:]
                            props = F.interpolate(init_mask, size=(h, w), mode='bilinear', align_corners=False)  # B, N+1, H, W
                            hard_probs = torch.zeros_like(props)
                            max_indices = torch.max(props, dim=1, keepdim=True)[1]
                            hard_probs.scatter_(dim=1, index=max_indices, value=1.0)
                            hard_probs = hard_probs[:, 1:]  # B, N, H, W; The binary segmentation mask
                            converted_masks = torch.stack([props[:, 1:] * hard_probs, (1 - props[:, 1:]) * (1 - hard_probs)], dim=2)  # B, N, 2, H, W;
                            self.mem_masks[oi].append(self.prop_net(qk16=f1_v, mode='tokenlearner_w_masks', mask = converted_masks[:,i])[0]) # B, num_token, c
                        else:
                            self.mem_masks[oi].append(f1_v)

        # Note that we never reach closest_ti, just the frame before it
        this_range = range(idx+1, closest_ti)
        end = closest_ti - 1

        for ti in this_range: 
            if self.valid_only:
                if ti not in self.req_frames:
                    continue
            mask_list = []
            for oi in self.enabled_obj: # obj_index
                mem_list = self.mem_frames[oi] # store the frame ids

                if layer_index == 0:
                    if len(mem_list) == 1: #only has the first frame
                        # self.mem_masks[oi][0]: 1,1,1,h,w
                        m16_f1_v1, m8_f1_v1, m4_f1_v1 = self.prop_net(memory_frames=self.images[:,self.mem_frames[oi][0]].unsqueeze(1), mask_frames=self.mem_masks[oi][0],  query_frame=self.images[:,ti], mode='backbone_full')
                        out_mask = self.prop_net(m16=m16_f1_v1, m8 = m8_f1_v1, m4 = m4_f1_v1, mode='segmentation_single_onject')
                    else:
                        mem_images_list = [self.images[:,f_index].unsqueeze(1) for f_index in self.mem_frames[oi]]
                        mem_images_tensor = torch.stack(mem_images_list, dim=1).flatten(1, 2)
                        mem_masks_tensor  = torch.stack(self.mem_masks[oi], dim=1).flatten(1, 2)
                        m16_f1_v1, m8_f1_v1, m4_f1_v1 = self.prop_net(memory_frames=mem_images_tensor, mask_frames=mem_masks_tensor, query_frame=self.images[:,ti], mode='backbone_full')
                        out_mask = self.prop_net(m16=m16_f1_v1, m8 = m8_f1_v1, m4 = m4_f1_v1, mode='segmentation_single_onject')
                else: #layer_index > 0
                    m16_f_ti = self.prop_net(frames=self.images[:,ti], mode='extract_feat_wo_mask', layer_index=layer_index, use_window_partition=use_local_window)  
                    _, L, _ = m16_f_ti.shape
                    if len(mem_list) == 1:
                        m16_f2_v_index, m8_f2_v_index, m4_f2_v_index = self.prop_net(template=self.mem_masks[oi][0], search=m16_f_ti, mode='forward_together', layer_index=layer_index, H=self.images[:,ti].shape[-2], W=self.images[:,ti].shape[-1], L=L)
                        out_mask = self.prop_net(m16=m16_f2_v_index, m8 = m8_f2_v_index, m4 = m4_f2_v_index, mode='segmentation_single_onject')
                    else:
                        mem_masks_tensor  = torch.stack(self.mem_masks[oi], dim=1).flatten(1, 2)
                        m16_f2_v_index, m8_f2_v_index, m4_f2_v_index = self.prop_net(template=mem_masks_tensor, search=m16_f_ti, mode='forward_together', layer_index=layer_index, H=self.images[:,ti].shape[-2], W=self.images[:,ti].shape[-1], L=L)
                        out_mask = self.prop_net(m16=m16_f2_v_index, m8 = m8_f2_v_index, m4 = m4_f2_v_index, mode='segmentation_single_onject')
                    
                mask_list.append(out_mask)
            out_mask = torch.stack(mask_list, dim=0).flatten(0, 1) #num, 1, 1, h, w
            
            out_mask = aggregate(out_mask, keep_bg=True)
            self.prob[0,ti] = out_mask[0]
            # for i, oi in enumerate(self.enabled_obj):
            #     self.prob[oi,ti] = out_mask[i+1]
            #     if ti < end: # we do not uodate our memory frames when at the end of the video or at the frame with new gt mask
            #         if len(self.mem_frames[oi]) == 1:
            #             self.mem_frames[oi].append(ti)
            #             self.mem_masks[oi].append(out_mask[i+1].unsqueeze(0).unsqueeze(0))
            #         else:
            #             self.mem_frames[oi][1] = ti
            #             self.mem_masks[oi][1] = out_mask[i+1].unsqueeze(0).unsqueeze(0)

            for i, oi in enumerate(self.enabled_obj):
                self.prob[oi,ti] = out_mask[i+1]
                if ti < end: # we do not uodate our memory frames when at the end of the video or at the frame with new gt mask
                    
                    if len(self.mem_masks[oi]) < self.mem_len:  # small memories
                        if layer_index == 0:
                            self.mem_frames[oi].append(ti)
                            self.mem_masks[oi].append(out_mask[i+1].unsqueeze(0).unsqueeze(0))
                        else:
                            self.mem_frames[oi].append(ti)
                            f1_v = self.prop_net(frames=self.images[:,ti], mode='extract_feat_w_mask', layer_index=layer_index, mask_frames=out_mask[i+1].unsqueeze(0), use_window_partition=use_local_window, use_token_learner=use_token_learner) 
                            if use_token_learner:
                                init_mask = out_mask.permute(1, 0, 2, 3) # out_mask: N+1, 1, H, W; 
                                h, w = f1_v.shape[-2:]
                                props = F.interpolate(init_mask, size=(h, w), mode='bilinear', align_corners=False)  # B, N+1, H, W
                                hard_probs = torch.zeros_like(props)
                                max_indices = torch.max(props, dim=1, keepdim=True)[1]
                                hard_probs.scatter_(dim=1, index=max_indices, value=1.0)
                                hard_probs = hard_probs[:, 1:]  # B, N, H, W; The binary segmentation mask
                                converted_masks = torch.stack([props[:, 1:] * hard_probs, (1 - props[:, 1:]) * (1 - hard_probs)], dim=2)  # B, N, 2, H, W;
                                self.mem_masks[oi].append(self.prop_net(qk16=f1_v, mode='tokenlearner_w_masks', mask = converted_masks[:,i])[0]) # B, num_token, c
                            else:
                                self.mem_masks[oi].append(f1_v)
                    else: # too many mamoeies
                        del self.mem_frames[oi][1]
                        del self.mem_masks[oi][1]
                        if layer_index == 0:
                            self.mem_frames[oi].append(ti)
                            self.mem_masks[oi].append(out_mask[i+1].unsqueeze(0).unsqueeze(0))
                        else:
                            self.mem_frames[oi].append(ti)
                            f1_v = self.prop_net(frames=self.images[:,ti], mode='extract_feat_w_mask', layer_index=layer_index, mask_frames=out_mask[i+1].unsqueeze(0), use_window_partition=use_local_window, use_token_learner=use_token_learner) 
                            if use_token_learner:
                                init_mask = out_mask.permute(1, 0, 2, 3) #1, N+1, H, W
                                h, w = f1_v.shape[-2:]
                                props = F.interpolate(init_mask, size=(h, w), mode='bilinear', align_corners=False)  # B, N+1, H, W
                                hard_probs = torch.zeros_like(props)
                                max_indices = torch.max(props, dim=1, keepdim=True)[1]
                                hard_probs.scatter_(dim=1, index=max_indices, value=1.0)
                                hard_probs = hard_probs[:, 1:]  # B, N, H, W; The binary segmentation mask
                                converted_masks = torch.stack([props[:, 1:] * hard_probs, (1 - props[:, 1:]) * (1 - hard_probs)], dim=2)  # B, N, 2, H, W;
                                self.mem_masks[oi].append(self.prop_net(qk16=f1_v, mode='tokenlearner_w_masks', mask = converted_masks[:,i])[0]) # B, num_token, c
                            else:
                                self.mem_masks[oi].append(f1_v)



                    # if len(self.mem_frames[oi]) < self.mem_len:
                    #     self.mem_frames[oi].append(ti)
                    #     self.mem_masks[oi].append(out_mask[i+1].unsqueeze(0).unsqueeze(0))
                    # else:
                    #     del self.mem_frames[oi][1]
                    #     del self.mem_masks[oi][1]
                    #     self.mem_frames[oi].append(ti)
                    #     self.mem_masks[oi].append(out_mask[i+1].unsqueeze(0).unsqueeze(0))

    
        return closest_ti

    def interact(self, mask, frame_idx, end_idx, obj_idx, layer_index, use_local_window, use_token_learner):
        # In youtube mode, we interact with a subset of object id at a time
        mask, _ = pad_divide_by(mask.cuda(), 16)  #num_obj+1, 1, h, w, including the background

        # update objects that have been labeled
        self.enabled_obj.extend(obj_idx)

        # Set other prob of mask regions to zero
        mask_regions = (mask[1:].sum(0) > 0.5)  # fine the non-background regions
        self.prob[:, frame_idx, mask_regions] = 0   # obj_idxs, frames, w, h
        self.prob[obj_idx, frame_idx] = mask[obj_idx] # use the GT to replace the original results

        self.prob[:, frame_idx] = aggregate(self.prob[1:, frame_idx], keep_bg=True)

        print('mask shape')
        print(mask.shape)

        # Propagate
        self.do_pass(frame_idx, end_idx, mask, obj_idx, layer_index, use_local_window, use_token_learner)  # for davis, we directly use mask for the initial annotations instead of self.prob


class InferenceCore_ViT_rf_lf:
    def __init__(self, prop_net:STCN, images, pos_embed_two_frame, req_frames, valid_only, num_objects):
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
        self.prob[0] = 1e-7

        self.t, self.h, self.w = t, h, w
        self.nh, self.nw = nh, nw
        self.kh = self.nh//16
        self.kw = self.nw//16

        # list of objects with usable memory
        self.enabled_obj = []

        self.mem_masks = dict()
        self.mem_frames = dict()

        pos_embed_two_frame_new, pos_embed_three_frame_new  = interpolate_pos_embed_2D(pos_embed_two_frame, self.kh, self.kw)  # 75.1 variant
        self.prop_net.pos_embed_two_frame = torch.nn.Parameter(pos_embed_two_frame_new)
        self.prop_net.pos_embed_three_frame = torch.nn.Parameter(pos_embed_three_frame_new)

        print('after interpolation:')
        print(self.prop_net.pos_embed_two_frame.shape)

        self.req_frames = req_frames
        self.valid_only = valid_only

        # self.mem_len = mem_len # including the 1st frame, so the baseline self.mem_len=2
        # self.save_every_frame = save_every_frame


    def do_pass(self, idx, end_idx, mask, obj_idx):
        closest_ti = end_idx
        
        for i, oi in enumerate(self.enabled_obj):
            if oi not in self.mem_masks: # newly added labeled objects w/ GT labels
                # self.mem_masks[oi] = [self.prob[oi,idx].unsqueeze(0).unsqueeze(0)] # B, T, C, H, W
                # self.mem_frames[oi] = [idx]
                self.mem_masks[oi] = [mask[oi].unsqueeze(0).unsqueeze(0)]
                self.mem_frames[oi] = [idx]
            else: # if exists; update the memory bank
                out_mask = self.prob[:,idx] #num_obj, 1, h, w
                out_mask_argmax = torch.argmax(out_mask, dim=0)
                if (out_mask_argmax == oi).sum() > 0:
                    print('quality test passed!!!')
                    if len(self.mem_frames[oi]) > 1: # already have 1st + prev frame
                            del self.mem_frames[oi][1]
                            del self.mem_masks[oi][1]
                            self.mem_frames[oi].append(idx)
                            self.mem_masks[oi].append(self.prob[oi,idx].unsqueeze(0).unsqueeze(0))
                    else:
                            self.mem_frames[oi].append(idx)
                            self.mem_masks[oi].append(self.prob[oi,idx].unsqueeze(0).unsqueeze(0))
                else:
                    print('low quality empty masks, drop!')
                        
        
        # Note that we never reach closest_ti, just the frame before it
        this_range = range(idx+1, closest_ti)
        end = closest_ti - 1

        for ti in this_range: 
            if self.valid_only:
                if ti not in self.req_frames:
                    continue
            mask_list = []
            for oi in self.enabled_obj: # obj_index
                mem_list = self.mem_frames[oi]
                if len(mem_list) == 1: #only has the 1st annotation
                    m16_f1_v1, m8_f1_v1, m4_f1_v1 = self.prop_net(memory_frames=self.images[:,self.mem_frames[oi][0]].unsqueeze(1), mask_frames=self.mem_masks[oi][0],  query_frame=self.images[:,ti], mode='backbone', layer_index=3, is_first_frame=True)
                else: #the latest frame = the previous frame
                    # m16_f1_v1, m8_f1_v1, m4_f1_v1 = self.prop_net(memory_frames=torch.cat((self.images[:,self.mem_frames[oi][0]].unsqueeze(1), self.images[:,self.mem_frames[oi][1]].unsqueeze(1)), dim=1), mask_frames=torch.cat((self.mem_masks[oi][0], self.mem_masks[oi][1]), dim=1),  query_frame=self.images[:,ti], mode='backbone', layer_index=3, is_first_frame=True)
                    mem_images_list = [self.images[:,f_index].unsqueeze(1) for f_index in self.mem_frames[oi]]
                    mem_images_tensor = torch.stack(mem_images_list, dim=1).flatten(1, 2)
                    mem_masks_tensor  = torch.stack(self.mem_masks[oi], dim=1).flatten(1, 2)
                    m16_f1_v1, m8_f1_v1, m4_f1_v1 = self.prop_net(memory_frames=mem_images_tensor, mask_frames=mem_masks_tensor, query_frame=self.images[:,ti], mode='backbone', layer_index=3, is_first_frame=True)
                out_mask = self.prop_net(m16=m16_f1_v1, m8 = m8_f1_v1, m4 = m4_f1_v1, mode='segmentation_single_onject')
                mask_list.append(out_mask)
            out_mask = torch.stack(mask_list, dim=0).flatten(0, 1) #num, 1, 1, h, w
            
            out_mask = aggregate(out_mask, keep_bg=True)
            self.prob[0,ti] = out_mask[0]

            out_mask_argmax = torch.argmax(out_mask, dim=0)

            for i, oi in enumerate(self.enabled_obj):
                self.prob[oi,ti] = out_mask[i+1]
                if ti < end: # we do not uodate our memory frames when at the end of the video or at the frame with new gt mask
                    if (out_mask_argmax == oi).sum() > 0:
                        print('quality test passed!!!')
                        if len(self.mem_frames[oi]) > 1: # already have 1st + prev frame
                                del self.mem_frames[oi][1]
                                del self.mem_masks[oi][1]
                                self.mem_frames[oi].append(ti)
                                self.mem_masks[oi].append(out_mask[i+1].unsqueeze(0).unsqueeze(0))
                        else:
                                self.mem_frames[oi].append(ti)
                                self.mem_masks[oi].append(out_mask[i+1].unsqueeze(0).unsqueeze(0))
                    else:
                        print('low quality empty masks, drop!')
                            
        return closest_ti

    def interact(self, mask, frame_idx, end_idx, obj_idx):
        # In youtube mode, we interact with a subset of object id at a time
        mask, _ = pad_divide_by(mask.cuda(), 16)

        # update objects that have been labeled
        self.enabled_obj.extend(obj_idx)

        # Set other prob of mask regions to zero
        mask_regions = (mask[1:].sum(0) > 0.5)
        self.prob[:, frame_idx, mask_regions] = 0
        self.prob[obj_idx, frame_idx] = mask[obj_idx] # use the GT to replace the original results

        self.prob[:, frame_idx] = aggregate(self.prob[1:, frame_idx], keep_bg=True)

        # Propagate
        self.do_pass(frame_idx, end_idx, mask, obj_idx)  # for davis, we directly use mask for the initial annotations instead of self.prob





class InferenceCore_ViT_w_intermediate_frames:
    def __init__(self, prop_net:STCN, images, pos_embed_two_frame, req_frames, valid_only, save_every_frame, mem_len, num_objects):
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
        self.prob[0] = 1e-7

        self.t, self.h, self.w = t, h, w
        self.nh, self.nw = nh, nw
        self.kh = self.nh//16
        self.kw = self.nw//16

        # list of objects with usable memory
        self.enabled_obj = []

        self.mem_masks = dict()
        self.mem_frames = dict()

        pos_embed_two_frame_new, pos_embed_three_frame_new  = interpolate_pos_embed_2D(pos_embed_two_frame, self.kh, self.kw)  # 75.1 variant
        self.prop_net.pos_embed_two_frame = torch.nn.Parameter(pos_embed_two_frame_new)
        self.prop_net.pos_embed_three_frame = torch.nn.Parameter(pos_embed_three_frame_new)

        print('after interpolation:')
        print(self.prop_net.pos_embed_two_frame.shape)

        self.req_frames = req_frames
        self.valid_only = valid_only

        self.mem_len = mem_len # including the 1st frame, so the baseline self.mem_len=2
        self.save_every_frame = save_every_frame


    def do_pass(self, idx, end_idx, mask, obj_idx):
        closest_ti = end_idx
        
        for i, oi in enumerate(self.enabled_obj):
            if oi not in self.mem_masks: # newly added labeled objects
                # self.mem_masks[oi] = [self.prob[oi,idx].unsqueeze(0).unsqueeze(0)] # B, T, C, H, W
                # self.mem_frames[oi] = [idx]
                self.mem_masks[oi] = [mask[oi].unsqueeze(0).unsqueeze(0)]
                self.mem_frames[oi] = [idx]
            else: # if exists; update the memory bank
                if (idx - self.mem_frames[oi][-1]) >= self.save_every_frame:
                    if len(self.mem_masks[oi]) < self.mem_len:
                        self.mem_frames[oi].append(idx)
                        self.mem_masks[oi].append(self.prob[oi,idx].unsqueeze(0).unsqueeze(0))
                    else:
                        del self.mem_frames[oi][1]
                        del self.mem_masks[oi][1]
                        self.mem_frames[oi].append(idx)
                        self.mem_masks[oi].append(self.prob[oi,idx].unsqueeze(0).unsqueeze(0))
        
        prev_frame_index = idx

        # Note that we never reach closest_ti, just the frame before it
        this_range = range(idx+1, closest_ti)
        end = closest_ti - 1

        for ti in this_range: 
            if self.valid_only:
                if ti not in self.req_frames:
                    continue
            mask_list = []
            for oi in self.enabled_obj: # obj_index
                mem_list = self.mem_frames[oi]
                if len(mem_list) == 1: #only has the 1st annotation
                    m16_f1_v1, m8_f1_v1, m4_f1_v1 = self.prop_net(memory_frames=self.images[:,self.mem_frames[oi][0]].unsqueeze(1), mask_frames=self.mem_masks[oi][0],  query_frame=self.images[:,ti], mode='backbone', layer_index=3, is_first_frame=True)
                elif self.mem_frames[oi][-1] == prev_frame_index: #the latest frame = the previous frame
                    # m16_f1_v1, m8_f1_v1, m4_f1_v1 = self.prop_net(memory_frames=torch.cat((self.images[:,self.mem_frames[oi][0]].unsqueeze(1), self.images[:,self.mem_frames[oi][1]].unsqueeze(1)), dim=1), mask_frames=torch.cat((self.mem_masks[oi][0], self.mem_masks[oi][1]), dim=1),  query_frame=self.images[:,ti], mode='backbone', layer_index=3, is_first_frame=True)
                    mem_images_list = [self.images[:,f_index].unsqueeze(1) for f_index in self.mem_frames[oi]]
                    mem_images_tensor = torch.stack(mem_images_list, dim=1).flatten(1, 2)
                    mem_masks_tensor  = torch.stack(self.mem_masks[oi], dim=1).flatten(1, 2)
                    m16_f1_v1, m8_f1_v1, m4_f1_v1 = self.prop_net(memory_frames=mem_images_tensor, mask_frames=mem_masks_tensor, query_frame=self.images[:,ti], mode='backbone', layer_index=3, is_first_frame=True)
                else: # we use 1st mask + memory frames + previous frame
                    mem_images_list = [self.images[:,f_index].unsqueeze(1) for f_index in self.mem_frames[oi]]
                    mem_images_tensor = torch.stack(mem_images_list, dim=1).flatten(1, 2)
                    mem_masks_tensor  = torch.stack(self.mem_masks[oi], dim=1).flatten(1, 2)
                    m16_f1_v1, m8_f1_v1, m4_f1_v1 = self.prop_net(memory_frames=torch.cat((mem_images_tensor, self.images[:,prev_frame_index].unsqueeze(1)), dim=1), mask_frames=torch.cat((mem_masks_tensor, self.prob[:,prev_frame_index][oi].unsqueeze(0).unsqueeze(0)), dim=1), query_frame=self.images[:,ti], mode='backbone', layer_index=3, is_first_frame=True)

                out_mask = self.prop_net(m16=m16_f1_v1, m8 = m8_f1_v1, m4 = m4_f1_v1, mode='segmentation_single_onject')
                mask_list.append(out_mask)
            out_mask = torch.stack(mask_list, dim=0).flatten(0, 1) #num, 1, 1, h, w
            
            out_mask = aggregate(out_mask, keep_bg=True)
            self.prob[0,ti] = out_mask[0]
            # for i, oi in enumerate(self.enabled_obj):
            #     self.prob[oi,ti] = out_mask[i+1]
            #     if ti < end: # we do not uodate our memory frames when at the end of the video or at the frame with new gt mask
            #         if len(self.mem_frames[oi]) == 1:
            #             self.mem_frames[oi].append(ti)
            #             self.mem_masks[oi].append(out_mask[i+1].unsqueeze(0).unsqueeze(0))
            #         else:
            #             self.mem_frames[oi][1] = ti
            #             self.mem_masks[oi][1] = out_mask[i+1].unsqueeze(0).unsqueeze(0)

            for i, oi in enumerate(self.enabled_obj):
                self.prob[oi,ti] = out_mask[i+1]
                if ti < end: # we do not uodate our memory frames when at the end of the video or at the frame with new gt mask
                    if (ti - self.mem_frames[oi][-1]) >= self.save_every_frame:
                        if len(self.mem_frames[oi]) < self.mem_len:
                            self.mem_frames[oi].append(ti)
                            self.mem_masks[oi].append(out_mask[i+1].unsqueeze(0).unsqueeze(0))
                        else:
                            del self.mem_frames[oi][1]
                            del self.mem_masks[oi][1]
                            self.mem_frames[oi].append(ti)
                            self.mem_masks[oi].append(out_mask[i+1].unsqueeze(0).unsqueeze(0))
            
            prev_frame_index = ti

    
        return closest_ti

    def interact(self, mask, frame_idx, end_idx, obj_idx):
        # In youtube mode, we interact with a subset of object id at a time
        mask, _ = pad_divide_by(mask.cuda(), 16)

        # update objects that have been labeled
        self.enabled_obj.extend(obj_idx)

        # Set other prob of mask regions to zero
        mask_regions = (mask[1:].sum(0) > 0.5)
        self.prob[:, frame_idx, mask_regions] = 0
        self.prob[obj_idx, frame_idx] = mask[obj_idx] # use the GT to replace the original results

        self.prob[:, frame_idx] = aggregate(self.prob[1:, frame_idx], keep_bg=True)

        # Propagate
        self.do_pass(frame_idx, end_idx, mask, obj_idx)  # for davis, we directly use mask for the initial annotations instead of self.prob





class InferenceCore_ViT_argmax:
    def __init__(self, prop_net:STCN, images, pos_embed_two_frame, req_frames, valid_only, mem_len, num_objects):
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
        self.prob[0] = 1e-7

        self.t, self.h, self.w = t, h, w
        self.nh, self.nw = nh, nw
        self.kh = self.nh//16
        self.kw = self.nw//16

        # list of objects with usable memory
        self.enabled_obj = []

        self.mem_masks = dict()
        self.mem_frames = dict()

        pos_embed_two_frame_new, pos_embed_three_frame_new  = interpolate_pos_embed_2D(pos_embed_two_frame, self.kh, self.kw)  # 75.1 variant
        self.prop_net.pos_embed_two_frame = torch.nn.Parameter(pos_embed_two_frame_new)
        self.prop_net.pos_embed_three_frame = torch.nn.Parameter(pos_embed_three_frame_new)

        print('after interpolation:')
        print(self.prop_net.pos_embed_two_frame.shape)

        self.req_frames = req_frames
        self.valid_only = valid_only

        self.mem_len = mem_len # including the 1st frame, so the baseline self.mem_len=2


    def do_pass(self, idx, end_idx):
        closest_ti = end_idx

        # for i, oi in enumerate(self.enabled_obj):
        #     if oi not in self.mem_masks: # newly added labeled objects
        #         self.mem_masks[oi] = [self.prob[oi,idx].unsqueeze(0).unsqueeze(0)] # B, T, C, H, W
        #         self.mem_frames[oi] = [idx]
        #     else: # if exists; update the memory bank
        #         if len(self.mem_masks[oi]) == 1:
        #             self.mem_frames[oi].append(idx)
        #             self.mem_masks[oi].append(self.prob[oi,idx].unsqueeze(0).unsqueeze(0))
        #         else:
        #             self.mem_frames[oi][1] = idx
        #             self.mem_masks[oi][1] = self.prob[oi,idx].unsqueeze(0).unsqueeze(0)
        
        for i, oi in enumerate(self.enabled_obj):
            if oi not in self.mem_masks: # newly added labeled objects, GT annotations
                self.mem_masks[oi] = [self.prob[oi,idx].unsqueeze(0).unsqueeze(0)] # B, T, C, H, W
                self.mem_frames[oi] = [idx]
            else: # if exists; update the memory bank
                out_mask = self.prob[:,idx]
                out_mask_argmax = torch.argmax(out_mask, dim=0)
                if len(self.mem_masks[oi]) < self.mem_len:
                    self.mem_frames[oi].append(idx)
                    self.mem_masks[oi].append((out_mask_argmax == oi).float().unsqueeze(0).unsqueeze(0).cuda())
                else: 
                    del self.mem_frames[oi][1]
                    del self.mem_masks[oi][1]
                    self.mem_frames[oi].append(idx)
                    self.mem_masks[oi].append((out_mask_argmax == oi).float().unsqueeze(0).unsqueeze(0).cuda())

        # Note that we never reach closest_ti, just the frame before it
        this_range = range(idx+1, closest_ti)
        end = closest_ti - 1

        for ti in this_range: 
            if self.valid_only:
                if ti not in self.req_frames:
                    continue
            mask_list = []
            for oi in self.enabled_obj: # obj_index
                mem_list = self.mem_frames[oi]
                if len(mem_list) == 1: #only has the 1st annotation
                    m16_f1_v1, m8_f1_v1, m4_f1_v1 = self.prop_net(memory_frames=self.images[:,self.mem_frames[oi][0]].unsqueeze(1), mask_frames=self.mem_masks[oi][0],  query_frame=self.images[:,ti], mode='backbone', layer_index=3, is_first_frame=True)
                else:
                    # m16_f1_v1, m8_f1_v1, m4_f1_v1 = self.prop_net(memory_frames=torch.cat((self.images[:,self.mem_frames[oi][0]].unsqueeze(1), self.images[:,self.mem_frames[oi][1]].unsqueeze(1)), dim=1), mask_frames=torch.cat((self.mem_masks[oi][0], self.mem_masks[oi][1]), dim=1),  query_frame=self.images[:,ti], mode='backbone', layer_index=3, is_first_frame=True)
                    mem_images_list = [self.images[:,f_index].unsqueeze(1) for f_index in self.mem_frames[oi]]
                    mem_images_tensor = torch.stack(mem_images_list, dim=1).flatten(1, 2)
                    mem_masks_tensor  = torch.stack(self.mem_masks[oi], dim=1).flatten(1, 2)
                    m16_f1_v1, m8_f1_v1, m4_f1_v1 = self.prop_net(memory_frames=mem_images_tensor, mask_frames=mem_masks_tensor, query_frame=self.images[:,ti], mode='backbone', layer_index=3, is_first_frame=True)
                    
                out_mask = self.prop_net(m16=m16_f1_v1, m8 = m8_f1_v1, m4 = m4_f1_v1, mode='segmentation_single_onject')
                mask_list.append(out_mask)
            out_mask = torch.stack(mask_list, dim=0).flatten(0, 1) #num, 1, 1, h, w
            
            out_mask = aggregate(out_mask, keep_bg=True)  #num_obj+1, 1, h, w
            self.prob[0,ti] = out_mask[0]
            # for i, oi in enumerate(self.enabled_obj):
            #     self.prob[oi,ti] = out_mask[i+1]
            #     if ti < end: # we do not uodate our memory frames when at the end of the video or at the frame with new gt mask
            #         if len(self.mem_frames[oi]) == 1:
            #             self.mem_frames[oi].append(ti)
            #             self.mem_masks[oi].append(out_mask[i+1].unsqueeze(0).unsqueeze(0))
            #         else:
            #             self.mem_frames[oi][1] = ti
            #             self.mem_masks[oi][1] = out_mask[i+1].unsqueeze(0).unsqueeze(0)
            # (torch.argmax(self.prob[:,ti-1], dim=0)==(obj_index+1)).float().unsqueeze(0).unsqueeze(0)
            out_mask_argmax = torch.argmax(out_mask, dim=0) # 1, h, w
            for i, oi in enumerate(self.enabled_obj):
                self.prob[oi,ti] = out_mask[i+1]
                if ti < end: # we do not uodate our memory frames when at the end of the video or at the frame with new gt mask
                    if len(self.mem_frames[oi]) < self.mem_len:
                        self.mem_frames[oi].append(ti)
                        self.mem_masks[oi].append((out_mask_argmax == oi).float().unsqueeze(0).unsqueeze(0).cuda())
                    else:
                        del self.mem_frames[oi][1]
                        del self.mem_masks[oi][1]
                        self.mem_frames[oi].append(ti)
                        self.mem_masks[oi].append((out_mask_argmax == oi).float().unsqueeze(0).unsqueeze(0).cuda())

    
        return closest_ti

    def interact(self, mask, frame_idx, end_idx, obj_idx):
        # In youtube mode, we interact with a subset of object id at a time
        mask, _ = pad_divide_by(mask.cuda(), 16)

        # update objects that have been labeled
        self.enabled_obj.extend(obj_idx)

        # Set other prob of mask regions to zero
        mask_regions = (mask[1:].sum(0) > 0.5)
        self.prob[:, frame_idx, mask_regions] = 0
        self.prob[obj_idx, frame_idx] = mask[obj_idx] # use the GT to replace the original results

        self.prob[:, frame_idx] = aggregate(self.prob[1:, frame_idx], keep_bg=True)

        # Propagate
        self.do_pass(frame_idx, end_idx)
