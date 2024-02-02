"""
model.py - warpper and utility functions for network training
Compute loss, back-prop, update parameters, logging, etc.
"""


import os
import time
import torch
import torch.nn as nn
import torch.optim as optim

from model.network import STCN
from model.losses import LossComputer, iou_hooks_mo, iou_hooks_so
from util.log_integrator import Integrator
from util.image_saver import pool_pairs

from torch.nn import functional as F

from model import models_vit
import timm
assert timm.__version__ == "0.3.2" # version check





class STCNModel:
    def __init__(self, para, logger=None, save_path=None, local_rank=0, world_size=1):
        self.para = para
        self.single_object = para['single_object']   # default: False, multiple objects per frame;
        self.local_rank = local_rank

        self.STCN = nn.parallel.DistributedDataParallel(
            STCN(self.single_object).cuda(), 
            device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False)

        # Setup logger when local_rank=0
        self.logger = logger
        self.save_path = save_path
        if logger is not None:
            self.last_time = time.time()
        self.train_integrator = Integrator(self.logger, distributed=True, local_rank=local_rank, world_size=world_size)
        if self.single_object:
            self.train_integrator.add_hook(iou_hooks_so)
        else:
            self.train_integrator.add_hook(iou_hooks_mo)
        self.loss_computer = LossComputer(para)

        self.train()
        self.optimizer = optim.Adam(filter(
            lambda p: p.requires_grad, self.STCN.parameters()), lr=para['lr'], weight_decay=1e-7)
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, para['steps'], para['gamma'])
        if para['amp']:
            self.scaler = torch.cuda.amp.GradScaler()

        # Logging info
        self.report_interval = 100
        self.save_im_interval = 800
        self.save_model_interval = 5000
        if para['debug']:
            self.report_interval = self.save_im_interval = 1

    def do_pass(self, data, it=0):
        # No need to store the gradient outside training
        torch.set_grad_enabled(self._is_train)

        for k, v in data.items():
            if type(v) != list and type(v) != dict and type(v) != int:
                data[k] = v.cuda(non_blocking=True)

        out = {}
        Fs = data['rgb'] # bs, 3, 3, 384, 384
        Ms = data['gt']  # bs, 3, 1, 384, 384

        with torch.cuda.amp.autocast(enabled=self.para['amp']):
            # key features never change, compute once
            # k16: key: 64-D, kf16: key:512-D; kf16, kf8, kf4, which are all output from the backbone
            k16, kf16_thin, kf16, kf8, kf4 = self.STCN('encode_key', Fs)

            if self.single_object:
                ref_v = self.STCN('encode_value', Fs[:,0], kf16[:,0], Ms[:,0])

                # Segment frame 1 with frame 0
                prev_logits, prev_mask = self.STCN('segment', 
                        k16[:,:,1], kf16_thin[:,1], kf8[:,1], kf4[:,1], 
                        k16[:,:,0:1], ref_v)
                prev_v = self.STCN('encode_value', Fs[:,1], kf16[:,1], prev_mask)

                values = torch.cat([ref_v, prev_v], 2)

                del ref_v

                # Segment frame 2 with frame 0 and 1
                this_logits, this_mask = self.STCN('segment', 
                        k16[:,:,2], kf16_thin[:,2], kf8[:,2], kf4[:,2], 
                        k16[:,:,0:2], values)

                out['mask_1'] = prev_mask
                out['mask_2'] = this_mask
                out['logits_1'] = prev_logits
                out['logits_2'] = this_logits
            else:
                sec_Ms = data['sec_gt']         # 4, 3, 1, 384, 384
                selector = data['selector']     # 4, 2


                # Why did you only consider up to 2 objects during training since one image might contain more than 5 objects?
                # Also, in the 'encode_value', other foreground objects mask is also engaged to generate the mask feature for one object. Do you think it could help the model to filter out background information and contribute to the models' performance?
                # Probably. But that just follows from STM and we did not specifically control it.
                
                # two targets in the frame 0
                ref_v1 = self.STCN('encode_value', Fs[:,0], kf16[:,0], Ms[:,0], sec_Ms[:,0]) #Fs: 4, 3, 3, 384, 384; ref_v1: 4, 512, 1, 24, 24
                ref_v2 = self.STCN('encode_value', Fs[:,0], kf16[:,0], sec_Ms[:,0], Ms[:,0]) #ref_v2: 4, 512, 1, 24, 24
                ref_v = torch.stack([ref_v1, ref_v2], 1) # 4, 2, 512, 1, 24, 24

                # Segment frame 1 with frame 0
                prev_logits, prev_mask = self.STCN('segment', 
                        k16[:,:,1], kf16_thin[:,1], kf8[:,1], kf4[:,1], 
                        k16[:,:,0:1], ref_v, selector)
                
                # two targets in the frame 1
                prev_v1 = self.STCN('encode_value', Fs[:,1], kf16[:,1], prev_mask[:,0:1], prev_mask[:,1:2])
                prev_v2 = self.STCN('encode_value', Fs[:,1], kf16[:,1], prev_mask[:,1:2], prev_mask[:,0:1])
                prev_v = torch.stack([prev_v1, prev_v2], 1)
                values = torch.cat([ref_v, prev_v], 3)

                del ref_v

                # Segment frame 2 with frame 0 and 1
                this_logits, this_mask = self.STCN('segment', 
                        k16[:,:,2], kf16_thin[:,2], kf8[:,2], kf4[:,2], 
                        k16[:,:,0:2], values, selector)

                out['mask_1'] = prev_mask[:,0:1]      # frame-1-taregt-1  shape: 4, 1, 384, 384
                out['mask_2'] = this_mask[:,0:1]      # frame-2-target-1
                out['sec_mask_1'] = prev_mask[:,1:2]  # frame-1-target-2
                out['sec_mask_2'] = this_mask[:,1:2]  # frame-2-target-2

                out['logits_1'] = prev_logits         # frame-1 prev_logits: 4, 3, 384, 384: background, fir_target, sec_target
                out['logits_2'] = this_logits         # frame-2: 4, 3, 384, 384: background, fir_target, sec_target

            if self._do_log or self._is_train:
                losses = self.loss_computer.compute({**data, **out}, it)

                # Logging
                if self._do_log:
                    self.integrator.add_dict(losses)
                    if self._is_train:
                        if it % self.save_im_interval == 0 and it != 0:
                            if self.logger is not None:
                                images = {**data, **out}
                                size = (384, 384)
                                self.logger.log_cv2('train/pairs', pool_pairs(images, size, self.single_object), it)

            if self._is_train:
                if (it) % self.report_interval == 0 and it != 0:
                    if self.logger is not None:
                        self.logger.log_scalar('train/lr', self.scheduler.get_last_lr()[0], it)
                        self.logger.log_metrics('train', 'time', (time.time()-self.last_time)/self.report_interval, it)
                    self.last_time = time.time()
                    self.train_integrator.finalize('train', it)
                    self.train_integrator.reset_except_hooks()

                if it % self.save_model_interval == 0 and it != 0:
                    if self.logger is not None:
                        self.save(it)

            # Backward pass
            # This should be done outside autocast
            # but I trained it like this and it worked fine
            # so I am keeping it this way for reference
            self.optimizer.zero_grad(set_to_none=True)
            if self.para['amp']:
                self.scaler.scale(losses['total_loss']).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                losses['total_loss'].backward() 
                self.optimizer.step()
            self.scheduler.step()

    def save(self, it):
        if self.save_path is None:
            print('Saving has been disabled.')
            return
        
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        model_path = self.save_path + ('_%s.pth' % it)
        torch.save(self.STCN.module.state_dict(), model_path)
        print('Model saved to %s.' % model_path)

        self.save_checkpoint(it)

    def save_checkpoint(self, it):
        if self.save_path is None:
            print('Saving has been disabled.')
            return

        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        checkpoint_path = self.save_path + '_checkpoint.pth'
        checkpoint = { 
            'it': it,
            'network': self.STCN.module.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict()}
        torch.save(checkpoint, checkpoint_path)

        print('Checkpoint saved to %s.' % checkpoint_path)

    def load_model(self, path):
        # This method loads everything and should be used to resume training
        map_location = 'cuda:%d' % self.local_rank
        checkpoint = torch.load(path, map_location={'cuda:0': map_location})

        it = checkpoint['it']
        network = checkpoint['network']
        optimizer = checkpoint['optimizer']
        scheduler = checkpoint['scheduler']

        map_location = 'cuda:%d' % self.local_rank
        self.STCN.module.load_state_dict(network)
        self.optimizer.load_state_dict(optimizer)
        self.scheduler.load_state_dict(scheduler)

        print('Model loaded.')

        return it

    def load_network(self, path):
        # This method loads only the network weight and should be used to load a pretrained model
        map_location = 'cuda:%d' % self.local_rank
        src_dict = torch.load(path, map_location={'cuda:0': map_location})

        # Maps SO weight (without other_mask) to MO weight (with other_mask)
        for k in list(src_dict.keys()):
            if k == 'value_encoder.conv1.weight':
                if src_dict[k].shape[1] == 4:
                    pads = torch.zeros((64,1,7,7), device=src_dict[k].device)
                    nn.init.orthogonal_(pads)
                    src_dict[k] = torch.cat([src_dict[k], pads], 1)

        self.STCN.module.load_state_dict(src_dict)
        print('Network weight loaded:', path)

    def train(self):
        self._is_train = True
        self._do_log = True
        self.integrator = self.train_integrator
        # Shall be in eval() mode to freeze BN parameters
        self.STCN.eval()
        return self

    def val(self):
        self._is_train = False
        self._do_log = True
        self.STCN.eval()
        return self

    def test(self):
        self._is_train = False
        self._do_log = False
        self.STCN.eval()
        return self





class ViTSTCNModel:
    def __init__(self, para, logger=None, save_path=None, local_rank=0, world_size=1):
        # we use the ViT model to perform the joint feature exctraction and interaction
        self.para = para
        self.single_object = para['single_object']   # default: False, multiple objects per frame;
        # the same parameters w/ the mae fine-tuning
        vit_model = models_vit.__dict__['vit_base_patch16'](
                num_classes=1000,
                drop_path_rate=0.1,
                global_pool=True,
                single_object = self.single_object,
                deep_low_map = para['deep_low_map'],
                use_tape = para['use_tape'],
                use_pos_emd = para['use_pos_emd'],
                valdim = para['valdim'],
                num_iters = para['num_iters'],
                num_bases = para['num_bases'],
                tau_value = para['tau'],
                num_bases_foreground = para['num_bases_foreground'],
                num_bases_background = para['num_bases_background'],
                img_size=para['img_size'])
        checkpoint = torch.load('/apdcephfs/private_qiangqwu/Projects/STCN/pretrain_models/mae_pretrain_vit_base.pth', map_location='cpu')
        print("Load pre-trained checkpoint from")
        checkpoint_model = checkpoint['model']
        state_dict = vit_model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape !=   state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # interpolate position embedding
        # load pre-trained model
        msg = vit_model.load_state_dict(checkpoint_model, strict=False)
        print(msg)
        
        

        ################### Original Setting used in STCN ##############################
        
        self.local_rank = local_rank

        self.STCN = nn.parallel.DistributedDataParallel(
            vit_model.cuda(), 
            device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False, find_unused_parameters=True)

        # Setup logger when local_rank=0
        self.logger = logger
        self.save_path = save_path
        if logger is not None:
            self.last_time = time.time()
        self.train_integrator = Integrator(self.logger, distributed=True, local_rank=local_rank, world_size=world_size)
        if self.single_object:
            self.train_integrator.add_hook(iou_hooks_so)
        else:
            self.train_integrator.add_hook(iou_hooks_mo)
        self.loss_computer = LossComputer(para)

        self.train()
        # for n, p in self.STCN.named_parameters():
        #     if (".blocks." in n or ".patch_embed." in n) and p.requires_grad:
        #         print(n)
        # print('#########################')
        # for n, p in self.STCN.named_parameters():
        #     if (".blocks." not in n and ".patch_embed." not in n) and p.requires_grad:
        #         print(n)
        # param_dicts = [
        #     {
        #         "params": [p for n, p in self.STCN.named_parameters() if (".blocks." in n or ".patch_embed." in n) and p.requires_grad]}, # backbone parameters
        #     {
        #         "params": [p for n, p in self.STCN.named_parameters() if (".blocks." not in n and ".patch_embed." not in n) and p.requires_grad],
        #         "lr": para['lr'] * para['lr_scale'],
        #     },
        # ]
        self.optimizer = optim.Adam(filter(
            lambda p: p.requires_grad, self.STCN.parameters()), lr=para['lr'], weight_decay=1e-7)
        # self.optimizer = optim.Adam(param_dicts, lr=para['lr'], weight_decay=1e-7)
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, para['steps'], para['gamma'])

        # para['amp'] = False

        if para['amp']:
            self.scaler = torch.cuda.amp.GradScaler()

        # Logging info
        self.report_interval = 100
        self.save_im_interval = 100 #800
        self.save_model_interval = 20000 #50000
        if para['debug']:
            self.report_interval = self.save_im_interval = 1
        self.fusion = para['fusion']
        self.num_bases_foreground = para['num_bases_foreground']
        self.num_bases_background = para['num_bases_background']
        self.img_size = para['img_size']
    

    # def memorize(self, qk16, props, bases_update=None):
    #     h, w = qk16.shape[-2:]  #24, 24
    #     props = F.interpolate(props, size=(h, w), mode='bilinear', align_corners=False)  # B, N+1, H, W
    #     hard_probs = torch.zeros_like(props)
    #     max_indices = torch.max(props, dim=1, keepdim=True)[1]
    #     hard_probs.scatter_(dim=1, index=max_indices, value=1.0)
    #     hard_probs = hard_probs[:, 1:]  # B, N, H, W; The binary segmentation mask

    #     masks = torch.stack([props[:, 1:] * hard_probs, (1 - props[:, 1:]) * (1 - hard_probs)], dim=2)  # B, N, 2, H, W;
    #     bases = self.swem_core.swem(qk16, masks, bases_update) # qk16: [8, 128, 24, 24]; #mv16: [8, 3, 512, 24, 24];

    #     return bases
    
    
    def do_pass(self, data, it=0):
        # No need to store the gradient outside training
        torch.set_grad_enabled(self._is_train)

        for k, v in data.items():
            if type(v) != list and type(v) != dict and type(v) != int:
                data[k] = v.cuda(non_blocking=True)

        out = {}
        Fs = data['rgb'] # bs, 3, 3, 384, 384
        Ms = data['gt']  # bs, 3, 1, 384, 384

        with torch.cuda.amp.autocast(enabled=self.para['amp']):
            # key features never change, compute once
            # k16: key: 64-D, kf16: key:512-D; kf16, kf8, kf4, which are all output from the backbone

            # k16, kf16_thin, kf16, kf8, kf4 = self.STCN('encode_key', Fs)
            # k16 = None
            # kf16_thin = None
            # kf16 = None
            # kf8 = None
            # kf4 = None

            if self.single_object:
                ref_v = self.STCN('encode_value', Fs[:,0], kf16[:,0], Ms[:,0])

                # Segment frame 1 with frame 0
                prev_logits, prev_mask = self.STCN('segment', 
                        k16[:,:,1], kf16_thin[:,1], kf8[:,1], kf4[:,1], 
                        k16[:,:,0:1], ref_v)
                prev_v = self.STCN('encode_value', Fs[:,1], kf16[:,1], prev_mask)

                values = torch.cat([ref_v, prev_v], 2)

                del ref_v

                # Segment frame 2 with frame 0 and 1
                this_logits, this_mask = self.STCN('segment', 
                        k16[:,:,2], kf16_thin[:,2], kf8[:,2], kf4[:,2], 
                        k16[:,:,0:2], values)

                out['mask_1'] = prev_mask
                out['mask_2'] = this_mask
                out['logits_1'] = prev_logits
                out['logits_2'] = this_logits
            else:

                sec_Ms = data['sec_gt']         # 4, 3, 1, 384, 384
                selector = data['selector']     # 4, 2 
                
                # start_a = time.time()

                # a = self.STCN(frames=Fs[:,0], mode='extract_feat_w_mask', layer_index=self.para['layer_index'], mask_frames=Ms[:,1])
                f1_v1 = self.STCN(frames=Fs[:,0], mode='extract_feat_w_mask', layer_index=self.para['layer_index'], mask_frames=Ms[:,0]) #bs, dim, h, w
                f1_v2 = self.STCN(frames=Fs[:,0], mode='extract_feat_w_mask', layer_index=self.para['layer_index'], mask_frames=sec_Ms[:,0]) #bs, dim, h, w
                
                # print('a:%f' %(time.time()-start_a))
                # start_b = time.time()
                
                first_target_mask = Ms[:, 0]  # 1st target in the first frame               # B, 1, H, W
                second_target_mask = sec_Ms[:, 0] # 2nd target in the second frame  # B, 1, H, W
                bg_mask = torch.ones_like(first_target_mask)  - torch.sum(torch.cat((first_target_mask, second_target_mask), dim=1), dim=1, keepdim=True)
                bg_mask[bg_mask != 1] = 0
                init_mask = torch.cat((bg_mask, first_target_mask, second_target_mask), dim=1)     # B, N+1, H, W
                # # token learner
                # bases_fixed_v1 = self.STCN(qk16=f1_v1, mode='tokenlearner')    # bs, num_token, c
                # bases_fixed_v2 = self.STCN(qk16=f1_v2, mode='tokenlearner')    # bs, num_token, c
                # tokenlearner with weights
                h, w = f1_v1.shape[-2:]
                props = F.interpolate(init_mask, size=(h, w), mode='bilinear', align_corners=False)  # B, N+1, H, W
                hard_probs = torch.zeros_like(props)
                max_indices = torch.max(props, dim=1, keepdim=True)[1]
                hard_probs.scatter_(dim=1, index=max_indices, value=1.0)
                hard_probs = hard_probs[:, 1:]  # B, N, H, W; The binary segmentation mask
                masks = torch.stack([props[:, 1:] * hard_probs, (1 - props[:, 1:]) * (1 - hard_probs)], dim=2)  # B, N, 2, H, W;
                bases_fixed_v1 = self.STCN(qk16=f1_v1, mode='tokenlearner_w_masks', mask = masks[:,0]) # B, num_token, c
                bases_fixed_v2 = self.STCN(qk16=f1_v2, mode='tokenlearner_w_masks', mask = masks[:,1]) # B, num_token, c
                
                # print('b:%f' %(time.time()-start_b))
                # start_c = time.time()


                # bases: B, N, 2, L, Ck
                # bases_fixed = self.STCN(qk16=torch.cat((f1_v1.unsqueeze(1), f1_v2.unsqueeze(1)), dim=1), mode='memory', props=init_mask.float(), bases_update=None)
                # bases_fixed = self.STCN(qk16=torch.cat((f1_v1.unsqueeze(1), f1_v2.unsqueeze(1)), dim=1), mode='memory', props=None, bases_update=None)
                # bases_update = bases_fixed
                # kappa = bases_fixed['kappa']
                # _, N, _, L, _ = kappa.shape
                # token learner
                _, L, _ = bases_fixed_v1.shape
                m16_f2 = self.STCN(frames=Fs[:,1], mode='extract_feat_wo_mask', layer_index=self.para['layer_index']) #bs, hw, dim
                
                # 1st target tokens
                # m16_f2_v1, m8_f2_v1, m4_f2_v1 = self.STCN(x=torch.cat((kappa[:, 0, 0], kappa[:, 0, 1], m16_f2), dim=-2), mode='extract_feat_in_later_layer', layer_index=self.para['layer_index'], H=Fs[:,1].shape[-2], W=Fs[:,1].shape[-1], L=L)
                # m16_f2_v1, m8_f2_v1, m4_f2_v1 = self.STCN(x=torch.cat((kappa[:, 0, 0], m16_f2), dim=-2), mode='extract_feat_in_later_layer', layer_index=self.para['layer_index'], H=Fs[:,1].shape[-2], W=Fs[:,1].shape[-1], L=L)
                # token learner
                m16_f2_v1, m8_f2_v1, m4_f2_v1 = self.STCN(x=torch.cat((bases_fixed_v1, m16_f2), dim=-2), mode='extract_feat_in_later_layer', layer_index=self.para['layer_index'], H=Fs[:,1].shape[-2], W=Fs[:,1].shape[-1], L=L)
                # 2nd target tokens
                # m16_f2_v2, m8_f2_v2, m4_f2_v2  = self.STCN(x=torch.cat((kappa[:, 1, 0], kappa[:, 1, 1], m16_f2), dim=-2), mode='extract_feat_in_later_layer', layer_index=self.para['layer_index'], H=Fs[:,1].shape[-2], W=Fs[:,1].shape[-1], L=L)
                # m16_f2_v2, m8_f2_v2, m4_f2_v2  = self.STCN(x=torch.cat((kappa[:, 1, 0], m16_f2), dim=-2), mode='extract_feat_in_later_layer', layer_index=self.para['layer_index'], H=Fs[:,1].shape[-2], W=Fs[:,1].shape[-1], L=L)
                # token learner
                m16_f2_v2, m8_f2_v2, m4_f2_v2 = self.STCN(x=torch.cat((bases_fixed_v2, m16_f2), dim=-2), mode='extract_feat_in_later_layer', layer_index=self.para['layer_index'], H=Fs[:,1].shape[-2], W=Fs[:,1].shape[-1], L=L)
                # Segment frame 1 with frame 0, prev_mask_all: B, N+1, H, W
                prev_logits, prev_mask, prev_mask_all = self.STCN(m16=torch.cat((m16_f2_v1.unsqueeze(1), m16_f2_v2.unsqueeze(1)), dim=1),
                           m8=torch.cat((m8_f2_v1.unsqueeze(1),   m8_f2_v2.unsqueeze(1)), dim=1), 
                           m4=torch.cat((m4_f2_v1.unsqueeze(1),   m4_f2_v2.unsqueeze(1)), dim=1), selector=selector, mode='segmentation')
                
                # print('c:%f' %(time.time()-start_c))
                # start_d = time.time()

                # f2_v1 = self.STCN(frames=Fs[:,1], mode='extract_feat_w_mask', layer_index=self.para['layer_index'], mask_frames=prev_mask_all[:,1].unsqueeze(1)) #bs, dim, h, w
                # f2_v2 = self.STCN(frames=Fs[:,1], mode='extract_feat_w_mask', layer_index=self.para['layer_index'], mask_frames=prev_mask_all[:,2].unsqueeze(1)) #bs, dim, h, w

                # h, w = f2_v1.shape[-2:]
                # props_sec = F.interpolate(prev_mask_all, size=(h, w), mode='bilinear', align_corners=False)  # B, N+1, H, W
                # hard_probs_sec = torch.zeros_like(props_sec)
                # max_indices = torch.max(props_sec, dim=1, keepdim=True)[1]
                # hard_probs_sec.scatter_(dim=1, index=max_indices, value=1.0)
                # hard_probs_sec = hard_probs_sec[:, 1:]  # B, N, H, W; The binary segmentation mask
                # masks_sec = torch.stack([props_sec[:, 1:] * hard_probs_sec, (1 - props_sec[:, 1:]) * (1 - hard_probs_sec)], dim=2)  # B, N, 2, H, W;
                # bases_fixed_v1_sec = self.STCN(qk16=f2_v1, mode='tokenlearner_w_masks', mask = masks_sec[:,0])
                # bases_fixed_v2_sec = self.STCN(qk16=f2_v2, mode='tokenlearner_w_masks', mask = masks_sec[:,1])

                # print('d:%f' %(time.time()-start_d))
                # start_e = time.time()

                # m16_f3 = self.STCN(frames=Fs[:,2], mode='extract_feat_wo_mask', layer_index=self.para['layer_index'])
                # bases_fixed_v1_updated = self.STCN(mode='tokenlearner_temporal', pre_f_qk16=bases_fixed_v1, cur_f_qk16=bases_fixed_v1_sec, fusion=self.fusion)
                # bases_fixed_v2_updated = self.STCN(mode='tokenlearner_temporal', pre_f_qk16=bases_fixed_v2, cur_f_qk16=bases_fixed_v2_sec, fusion=self.fusion)

                # # em clustering
                # f1_models = [bases_fixed_v1, bases_fixed_v2]
                # f2_models = [bases_fixed_v1_sec, bases_fixed_v2_sec]
                # f1_models  = torch.stack(f1_models, dim=1) #bs, N=2, num_token, C
                # f2_models  = torch.stack(f2_models, dim=1) #bs, N=2, num_token, C
                # # bs, N, num_for_tokens*2, c 
                # for_tokens_new = torch.cat((f1_models[:,:,0:self.num_bases_foreground], f2_models[:,:,0:self.num_bases_foreground]), dim=2)
                # # bs, N, num_back_tokens*2, c
                # back_tokens_new = torch.cat((f1_models[:,:,self.num_bases_foreground:], f2_models[:,:,self.num_bases_foreground:]), dim=2)
                # bases_update_for = self.STCN(qk16=for_tokens_new, mode='memory_tokenlearner', bases_update=None, num_cluster=self.num_bases_foreground)
                # bases_update_back = self.STCN(qk16=back_tokens_new, mode='memory_tokenlearner', bases_update=None, num_cluster=self.num_bases_background)
                # m16_f3_v1, m8_f3_v1, m4_f3_v1 = self.STCN(x=torch.cat((bases_update_for['kappa'][:,0].squeeze(1), bases_update_back['kappa'][:,0].squeeze(1), m16_f3), dim=-2), mode='extract_feat_in_later_layer', layer_index=self.para['layer_index'], H=Fs[:,1].shape[-2], W=Fs[:,1].shape[-1], L=L)
                # m16_f3_v2, m8_f3_v2, m4_f3_v2 = self.STCN(x=torch.cat((bases_update_for['kappa'][:,1].squeeze(1), bases_update_back['kappa'][:,1].squeeze(1), m16_f3), dim=-2), mode='extract_feat_in_later_layer', layer_index=self.para['layer_index'], H=Fs[:,1].shape[-2], W=Fs[:,1].shape[-1], L=L)


                # m16_f3_v1, m8_f3_v1, m4_f3_v1 = self.STCN(x=torch.cat((bases_fixed_v1, bases_fixed_v1_sec, m16_f3), dim=-2), mode='extract_feat_in_later_layer_test', layer_index=self.para['layer_index'], H=Fs[:,1].shape[-2], W=Fs[:,1].shape[-1], L=L)
                # m16_f3_v2, m8_f3_v2, m4_f3_v2 = self.STCN(x=torch.cat((bases_fixed_v2, bases_fixed_v2_sec, m16_f3), dim=-2), mode='extract_feat_in_later_layer_test', layer_index=self.para['layer_index'], H=Fs[:,1].shape[-2], W=Fs[:,1].shape[-1], L=L)
                # # m16_f3_v1, m8_f3_v1, m4_f3_v1 = self.STCN(x=torch.cat((bases_fixed_v1_updated, m16_f3), dim=-2), mode='extract_feat_in_later_layer', layer_index=self.para['layer_index'], H=Fs[:,1].shape[-2], W=Fs[:,1].shape[-1], L=L)
                # # m16_f3_v2, m8_f3_v2, m4_f3_v2 = self.STCN(x=torch.cat((bases_fixed_v2_updated, m16_f3), dim=-2), mode='extract_feat_in_later_layer', layer_index=self.para['layer_index'], H=Fs[:,1].shape[-2], W=Fs[:,1].shape[-1], L=L)
                # this_logits, this_mask, _ = self.STCN(m16=torch.cat((m16_f3_v1.unsqueeze(1), m16_f3_v2.unsqueeze(1)), dim=1),
                #            m8=torch.cat((m8_f3_v1.unsqueeze(1),   m8_f3_v2.unsqueeze(1)), dim=1), 
                #            m4=torch.cat((m4_f3_v1.unsqueeze(1),   m4_f3_v2.unsqueeze(1)), dim=1), selector=selector, mode='segmentation')

                # print('e:%f' %(time.time()-start_e))




                # f2_v1 = self.STCN(frames=Fs[:,1], mode='extract_feat_w_mask', layer_index=self.para['layer_index'], mask_frames=prev_mask_all[:,1].unsqueeze(1)) #bs, dim, h, w
                # f2_v2 = self.STCN(frames=Fs[:,1], mode='extract_feat_w_mask', layer_index=self.para['layer_index'], mask_frames=prev_mask_all[:,2].unsqueeze(1)) #bs, dim, h, w
                
                # bases_update = self.STCN(qk16=torch.cat((f2_v1.unsqueeze(1), f2_v2.unsqueeze(1)), dim=1), mode='memory', props=prev_mask_all, bases_update=bases_update)
                # # bases_update = self.STCN(qk16=torch.cat((f2_v1.unsqueeze(1), f2_v2.unsqueeze(1)), dim=1), mode='memory', props=None, bases_update=bases_update)
                # kappa = torch.cat([bases_fixed['kappa'], bases_update['kappa']], dim=-2)
                # m16_f3 = self.STCN(frames=Fs[:,2], mode='extract_feat_wo_mask', layer_index=self.para['layer_index']) #bs, hw, dim
                # _, N, _, L, _ = kappa.shape
                # m16_f3_v1, m8_f3_v1, m4_f3_v1 = self.STCN(x=torch.cat((kappa[:, 0, 0], kappa[:, 0, 1], m16_f3), dim=-2), mode='extract_feat_in_later_layer', layer_index=self.para['layer_index'], H=Fs[:,2].shape[-2], W=Fs[:,2].shape[-1], L=L)
                # m16_f3_v2, m8_f3_v2, m4_f3_v2 = self.STCN(x=torch.cat((kappa[:, 1, 0], kappa[:, 1, 1], m16_f3), dim=-2), mode='extract_feat_in_later_layer', layer_index=self.para['layer_index'], H=Fs[:,2].shape[-2], W=Fs[:,2].shape[-1], L=L)
                # this_logits, this_mask, _ = self.STCN(m16=torch.cat((m16_f3_v1.unsqueeze(1), m16_f3_v2.unsqueeze(1)), dim=1),
                #            m8=torch.cat((m8_f3_v1.unsqueeze(1),   m8_f3_v2.unsqueeze(1)), dim=1), 
                #            m4=torch.cat((m4_f3_v1.unsqueeze(1),   m4_f3_v2.unsqueeze(1)), dim=1), selector=selector, mode='segmentation')


                
                
                

                



                               
                
                # mk16 = self.STCN(frames=Fs[:,0], mode='backbone', is_feat_extra=True, layer_index=self.para['layer_index'])
                # m16_f1_v2, m8_f1_v2, m4_f1_v2= self.STCN(memory_frames=Fs[:,0].unsqueeze(1), mask_frames=sec_Ms[:,0].unsqueeze(1),  query_frame=Fs[:,1], mode='backbone', is_first_frame=True, layer_index=self.para['layer_index'])
                
                # # Segment frame 1 with frame 0
                # prev_logits, prev_mask = self.STCN(m16=torch.cat((m16_f1_v1.unsqueeze(1), m16_f1_v2.unsqueeze(1)), dim=1),
                #            m8=torch.cat((m8_f1_v1.unsqueeze(1),   m8_f1_v2.unsqueeze(1)), dim=1), 
                #            m4=torch.cat((m4_f1_v1.unsqueeze(1),   m4_f1_v2.unsqueeze(1)), dim=1), selector=selector, mode='segmentation')
                
                # del m16_f1_v1
                # del m8_f1_v1
                # del m4_f1_v1
                # del m16_f1_v2
                # del m8_f1_v2
                # del m4_f1_v2

                # m16_f2_v1, m8_f2_v1, m4_f2_v1 = self.STCN(memory_frames=Fs[:,0:2], mask_frames=torch.cat((Ms[:,0].unsqueeze(1), prev_mask[:,0:1].unsqueeze(1)), dim=1),  query_frame=Fs[:,2], mode='backbone')
                # m16_f2_v2, m8_f2_v2, m4_f2_v2 = self.STCN(memory_frames=Fs[:,0:2], mask_frames=torch.cat((sec_Ms[:,0].unsqueeze(1), prev_mask[:,1:2].unsqueeze(1)), dim=1),  query_frame=Fs[:,2], mode='backbone')

                # m16_f2_v1, m8_f2_v1, m4_f2_v1, updated_target_tokens_v1 = self.STCN(memory_frames=Fs[:,0].unsqueeze(1), memory_tokens=updated_target_tokens_v1, query_frame=Fs[:,2], mode='parallel_backbone', is_first_frame=False, mask_tokens=mask_tokens_v1)
                # m16_f2_v2, m8_f2_v2, m4_f2_v2, updated_target_tokens_v2 = self.STCN(memory_frames=Fs[:,0].unsqueeze(1), memory_tokens=updated_target_tokens_v2, query_frame=Fs[:,2], mode='parallel_backbone', is_first_frame=False, mask_tokens=mask_tokens_v2)

                # # # Segment frame 2 with frames 0 and 1
                # this_logits, this_mask = self.STCN(m16=torch.cat((m16_f2_v1.unsqueeze(1), m16_f2_v2.unsqueeze(1)), dim=1),
                #            m8=torch.cat((m8_f2_v1.unsqueeze(1),   m8_f2_v2.unsqueeze(1)), dim=1), 
                #            m4=torch.cat((m4_f2_v1.unsqueeze(1),   m4_f2_v2.unsqueeze(1)), dim=1), selector=selector, mode='segmentation')
                

                # m16_f3_v1, m8_f3_v1, m4_f3_v1, _ = self.STCN(memory_frames=Fs[:,0].unsqueeze(1), memory_tokens=updated_target_tokens_v1, query_frame=Fs[:,3], mode='parallel_backbone', is_first_frame=False, mask_tokens=mask_tokens_v1)
                # m16_f3_v2, m8_f3_v2, m4_f3_v2, _ = self.STCN(memory_frames=Fs[:,0].unsqueeze(1), memory_tokens=updated_target_tokens_v2, query_frame=Fs[:,3], mode='parallel_backbone', is_first_frame=False, mask_tokens=mask_tokens_v2)

                # last_logits, last_mask = self.STCN(m16=torch.cat((m16_f3_v1.unsqueeze(1), m16_f3_v2.unsqueeze(1)), dim=1),
                #            m8=torch.cat((m8_f3_v1.unsqueeze(1),   m8_f3_v2.unsqueeze(1)), dim=1), 
                #            m4=torch.cat((m4_f3_v1.unsqueeze(1),   m4_f3_v2.unsqueeze(1)), dim=1), selector=selector, mode='segmentation')

                


                    
                
                # # two targets in the frame 0
                # ref_v1 = self.STCN('encode_value', Fs[:,0], kf16[:,0], Ms[:,0], sec_Ms[:,0]) #Fs: 4, 3, 3, 384, 384; ref_v1: 4, 512, 1, 24, 24
                # ref_v2 = self.STCN('encode_value', Fs[:,0], kf16[:,0], sec_Ms[:,0], Ms[:,0]) #ref_v2: 4, 512, 1, 24, 24
                # ref_v = torch.stack([ref_v1, ref_v2], 1) # 4, 2, 512, 1, 24, 24

                
                # prev_logits, prev_mask = self.STCN('segment', 
                #         k16[:,:,1], kf16_thin[:,1], kf8[:,1], kf4[:,1], 
                #         k16[:,:,0:1], ref_v, selector)
                
                # # two targets in the frame 1
                # prev_v1 = self.STCN('encode_value', Fs[:,1], kf16[:,1], prev_mask[:,0:1], prev_mask[:,1:2])
                # prev_v2 = self.STCN('encode_value', Fs[:,1], kf16[:,1], prev_mask[:,1:2], prev_mask[:,0:1])
                # prev_v = torch.stack([prev_v1, prev_v2], 1)
                # values = torch.cat([ref_v, prev_v], 3)

                # del ref_v

                # Segment frame 2 with frame 0 and 1
                # this_logits, this_mask = self.STCN('segment', 
                #         k16[:,:,2], kf16_thin[:,2], kf8[:,2], kf4[:,2], 
                #         k16[:,:,0:2], values, selector)

                out['mask_1'] = prev_mask[:,0:1]      # frame-1-taregt-1
                # out['mask_2'] = this_mask[:,0:1]      # frame-2-target-1
                out['sec_mask_1'] = prev_mask[:,1:2]  # frame-1-target-2
                # out['sec_mask_2'] = this_mask[:,1:2]  # frame-2-target-2

                out['logits_1'] = prev_logits          # frame-1 
                # out['logits_2'] = this_logits        # frame-2

            if self._do_log or self._is_train:
                losses = self.loss_computer.compute({**data, **out}, it)

                # Logging
                if self._do_log:
                    self.integrator.add_dict(losses)
                    if self._is_train:
                        if it % self.save_im_interval == 0 and it != 0:  #save_im_interval = 800
                            if self.logger is not None:
                                images = {**data, **out}
                                size = (384, 384)
                                self.logger.log_cv2('train/pairs', pool_pairs(images, size, self.single_object), it)

            if self._is_train:
                if (it) % self.report_interval == 0 and it != 0:
                    if self.logger is not None:
                        self.logger.log_scalar('train/lr', self.scheduler.get_last_lr()[0], it)
                        self.logger.log_metrics('train', 'time', (time.time()-self.last_time)/self.report_interval, it)
                    self.last_time = time.time()
                    self.train_integrator.finalize('train', it)
                    self.train_integrator.reset_except_hooks()

                if it % self.save_model_interval == 0 and it != 0:    # self.save_model_interval = 50000
                    if self.logger is not None:
                        self.save(it)

            # Backward pass
            # This should be done outside autocast
            # but I trained it like this and it worked fine
            # so I am keeping it this way for reference
            self.optimizer.zero_grad(set_to_none=True)
            if self.para['amp']:
                self.scaler.scale(losses['total_loss']).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                losses['total_loss'].backward() 
                self.optimizer.step()
            self.scheduler.step()

    def save(self, it):
        if self.save_path is None:
            print('Saving has been disabled.')
            return
        
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        model_path = self.save_path + ('_%s.pth' % it)
        torch.save(self.STCN.module.state_dict(), model_path)
        print('Model saved to %s.' % model_path)

        self.save_checkpoint(it)

    def save_checkpoint(self, it):
        if self.save_path is None:
            print('Saving has been disabled.')
            return

        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        checkpoint_path = self.save_path + '_checkpoint.pth'
        checkpoint = { 
            'it': it,
            'network': self.STCN.module.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict()}
        torch.save(checkpoint, checkpoint_path)

        print('Checkpoint saved to %s.' % checkpoint_path)

    def load_model(self, path):
        # This method loads everything and should be used to resume training
        map_location = 'cuda:%d' % self.local_rank
        checkpoint = torch.load(path, map_location={'cuda:0': map_location})

        it = checkpoint['it']
        network = checkpoint['network']
        optimizer = checkpoint['optimizer']
        scheduler = checkpoint['scheduler']

        map_location = 'cuda:%d' % self.local_rank
        self.STCN.module.load_state_dict(network)
        self.optimizer.load_state_dict(optimizer)
        self.scheduler.load_state_dict(scheduler)

        print('Model loaded.')

        return it

    def load_network(self, path):
        # This method loads only the network weight and should be used to load a pretrained model
        map_location = 'cuda:%d' % self.local_rank
        src_dict = torch.load(path, map_location={'cuda:0': map_location})

        # Maps SO weight (without other_mask) to MO weight (with other_mask)
        for k in list(src_dict.keys()):
            if k == 'value_encoder.conv1.weight':
                if src_dict[k].shape[1] == 4:
                    pads = torch.zeros((64,1,7,7), device=src_dict[k].device)
                    nn.init.orthogonal_(pads)
                    src_dict[k] = torch.cat([src_dict[k], pads], 1)

        self.STCN.module.load_state_dict(src_dict)
        print('Network weight loaded:', path)

    def train(self):
        self._is_train = True
        self._do_log = True
        self.integrator = self.train_integrator
        # Shall be in eval() mode to freeze BN parameters # do we need to freeze all BN weights for MAE?
        self.STCN.train()
        return self

    def val(self):
        self._is_train = False
        self._do_log = True
        self.STCN.eval()
        return self

    def test(self):
        self._is_train = False
        self._do_log = False
        self.STCN.eval()
        return self

