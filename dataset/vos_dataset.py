import os
from os import path

import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image
import numpy as np

from dataset.range_transform import im_normalization, im_mean
from dataset.reseed import reseed


# class VOSDataset(Dataset):
#     """
#     Works for DAVIS/YouTubeVOS/BL30K training
#     For each sequence:
#     - Pick three frames
#     - Pick two objects
#     - Apply some random transforms that are the same for all frames
#     - Apply random transform to each of the frame
#     - The distance between frames is controlled
#     """
#     def __init__(self, im_root, gt_root, max_jump, is_bl, subset=None):
#         self.im_root = im_root
#         self.gt_root = gt_root
#         self.max_jump = max_jump
#         self.is_bl = is_bl

#         self.videos = []
#         self.frames = {}

#         vid_list = sorted(os.listdir(self.im_root))
#         # Pre-filtering   at least have 3 frames!!!
#         for vid in vid_list:
#             if subset is not None:
#                 if vid not in subset:
#                     continue
#             frames = sorted(os.listdir(os.path.join(self.im_root, vid)))
#             if len(frames) < 3:
#                 continue
#             self.frames[vid] = frames
#             self.videos.append(vid)

#         print('%d out of %d videos accepted in %s.' % (len(self.videos), len(vid_list), im_root))

#         # These set of transform is the same for im/gt pairs, but different among the 3 sampled frames
#         self.pair_im_lone_transform = transforms.Compose([
#             transforms.ColorJitter(0.01, 0.01, 0.01, 0),
#         ])

#         self.pair_im_dual_transform = transforms.Compose([
#             transforms.RandomAffine(degrees=15, shear=10, interpolation=InterpolationMode.BICUBIC, fill=im_mean),
#         ])

#         self.pair_gt_dual_transform = transforms.Compose([
#             transforms.RandomAffine(degrees=15, shear=10, interpolation=InterpolationMode.NEAREST, fill=0),
#         ])

#         # These transform are the same for all pairs in the sampled sequence
#         self.all_im_lone_transform = transforms.Compose([
#             transforms.ColorJitter(0.1, 0.03, 0.03, 0),
#             transforms.RandomGrayscale(0.05),
#         ])

#         if self.is_bl: # this is just for another dataset BL, which is quite large
#             # Use a different cropping scheme for the blender dataset because the image size is different
#             self.all_im_dual_transform = transforms.Compose([
#                 transforms.RandomHorizontalFlip(),
#                 transforms.RandomResizedCrop((384, 384), scale=(0.25, 1.00), interpolation=InterpolationMode.BICUBIC)
#             ])

#             self.all_gt_dual_transform = transforms.Compose([
#                 transforms.RandomHorizontalFlip(),
#                 transforms.RandomResizedCrop((384, 384), scale=(0.25, 1.00), interpolation=InterpolationMode.NEAREST)
#             ])
#         else:
#             self.all_im_dual_transform = transforms.Compose([
#                 transforms.RandomHorizontalFlip(),
#                 transforms.RandomResizedCrop((384, 384), scale=(0.36,1.00), interpolation=InterpolationMode.BICUBIC)
#             ])

#             self.all_gt_dual_transform = transforms.Compose([
#                 transforms.RandomHorizontalFlip(),
#                 transforms.RandomResizedCrop((384, 384), scale=(0.36,1.00), interpolation=InterpolationMode.NEAREST)
#             ])

#         # Final transform without randomness
#         self.final_im_transform = transforms.Compose([
#             transforms.ToTensor(),
#             im_normalization,
#         ])

#     def __getitem__(self, idx):
#         video = self.videos[idx]   # 3464 videos for VOS
#         info = {}
#         info['name'] = video

#         vid_im_path = path.join(self.im_root, video)  # get the image path
#         vid_gt_path = path.join(self.gt_root, video)  # get the seg. gt path
#         frames = self.frames[video]   # get the sorted frames of the video

#         trials = 0
#         while trials < 5:   
#             info['frames'] = [] # Appended with actual frames

#             # Don't want to bias towards beginning/end
#             this_max_jump = min(len(frames), self.max_jump)                     # the max_jump shouldn't > len(frames)
#             start_idx = np.random.randint(len(frames)-this_max_jump+1)          
#             f1_idx = start_idx + np.random.randint(this_max_jump+1) + 1         # + 1 is used to make the f1_idx 100% > start_idx
#             f1_idx = min(f1_idx, len(frames)-this_max_jump, len(frames)-1)      # maximum value: len(frames)-2

#             f2_idx = f1_idx + np.random.randint(this_max_jump+1) + 1
#             f2_idx = min(f2_idx, len(frames)-this_max_jump//2, len(frames)-1)

#             frames_idx = [start_idx, f1_idx, f2_idx]
#             if np.random.rand() < 0.5:
#                 # Reverse time
#                 frames_idx = frames_idx[::-1]

#             sequence_seed = np.random.randint(2147483647)
#             images = []
#             masks = []
#             target_object = None
#             for f_idx in frames_idx:
#                 jpg_name = frames[f_idx][:-4] + '.jpg'  # for RGB images
#                 png_name = frames[f_idx][:-4] + '.png'  # for gt
#                 info['frames'].append(jpg_name)
                

#                 # images: all_im_dual_transform ---> all_im_lone_transform ---> pair_im_dual_transform ---> final_im_transform
#                 # gt:     all_gt_dual_transform ---> pair_gt_dual_transform

#                 # self.all_im_dual_transform = transforms.Compose([
#                 # transforms.RandomHorizontalFlip(),
#                 # transforms.RandomResizedCrop((384, 384), scale=(0.36,1.00), interpolation=InterpolationMode.BICUBIC)])

#                 # self.all_im_lone_transform = transforms.Compose([
#                 # transforms.ColorJitter(0.1, 0.03, 0.03, 0),
#                 # transforms.RandomGrayscale(0.05),
#                 # ])

#                 # self.pair_im_dual_transform = transforms.Compose([
#                 # transforms.RandomAffine(degrees=15, shear=10, interpolation=InterpolationMode.BICUBIC, fill=im_mean),
#                 # ])

#                 # self.final_im_transform = transforms.Compose([
#                 # transforms.ToTensor(),
#                 # im_normalization,
#                 # ])



#                 # self.all_gt_dual_transform = transforms.Compose([
#                 # transforms.RandomHorizontalFlip(),
#                 # transforms.RandomResizedCrop((384, 384), scale=(0.36,1.00), interpolation=InterpolationMode.NEAREST)
#                 # ])

#                 # self.pair_gt_dual_transform = transforms.Compose([
#                 # transforms.RandomAffine(degrees=15, shear=10, interpolation=InterpolationMode.NEAREST, fill=0),
#                 # ])


#                 reseed(sequence_seed)
#                 this_im = Image.open(path.join(vid_im_path, jpg_name)).convert('RGB')
#                 this_im = self.all_im_dual_transform(this_im)
#                 this_im = self.all_im_lone_transform(this_im)
#                 reseed(sequence_seed)
#                 this_gt = Image.open(path.join(vid_gt_path, png_name)).convert('P')
#                 this_gt = self.all_gt_dual_transform(this_gt)

#                 pairwise_seed = np.random.randint(2147483647)
#                 reseed(pairwise_seed)
#                 this_im = self.pair_im_dual_transform(this_im)
#                 this_im = self.pair_im_lone_transform(this_im)
#                 reseed(pairwise_seed)
#                 this_gt = self.pair_gt_dual_transform(this_gt)

#                 this_im = self.final_im_transform(this_im)
#                 this_gt = np.array(this_gt)

#                 images.append(this_im)
#                 masks.append(this_gt)

#             images = torch.stack(images, 0)  # 3, 3, 384, 384

#             labels = np.unique(masks[0]) # [] list size=3, we treat 0 sa the first frame, our goal is to track the objects in the following frames
#             # Remove background
#             labels = labels[labels!=0]

#             if self.is_bl:
#                 # Find large enough labels
#                 good_lables = []
#                 for l in labels:
#                     pixel_sum = (masks[0]==l).sum()
#                     if pixel_sum > 10*10:
#                         # OK if the object is always this small
#                         # Not OK if it is actually much bigger
#                         if pixel_sum > 30*30:
#                             good_lables.append(l)
#                         elif max((masks[1]==l).sum(), (masks[2]==l).sum()) < 20*20:
#                             good_lables.append(l)
#                 labels = np.array(good_lables, dtype=np.uint8)
            
#             if len(labels) == 0:
#                 target_object = -1 # all black if no objects
#                 has_second_object = False
#                 trials += 1
#             else:
#                 target_object = np.random.choice(labels)
#                 has_second_object = (len(labels) > 1)
#                 if has_second_object:   # we randomly select another target sa the second object. 
#                     labels = labels[labels!=target_object]
#                     second_object = np.random.choice(labels)
#                 break

#         masks = np.stack(masks, 0) # convert the list to array masks: 3, 384, 384
#         tar_masks = (masks==target_object).astype(np.float32)[:,np.newaxis,:,:] # 3, 1, 384, 384
#         if has_second_object:
#             sec_masks = (masks==second_object).astype(np.float32)[:,np.newaxis,:,:]
#             selector = torch.FloatTensor([1, 1])
#         else:
#             sec_masks = np.zeros_like(tar_masks)
#             selector = torch.FloatTensor([1, 0])

#         cls_gt = np.zeros((3, 384, 384), dtype=np.int)
#         cls_gt[tar_masks[:,0] > 0.5] = 1
#         cls_gt[sec_masks[:,0] > 0.5] = 2

#         data = {
#             'rgb': images,
#             'gt': tar_masks,
#             'cls_gt': cls_gt,
#             'sec_gt': sec_masks,
#             'selector': selector,
#             'info': info,
#         }

#         return data

#     def __len__(self):
#         return len(self.videos)



class VOSDataset(Dataset):
    """
    Works for DAVIS/YouTubeVOS/BL30K training
    For each sequence:
    - Pick three frames
    - Pick two objects
    - Apply some random transforms that are the same for all frames
    - Apply random transform to each of the frame
    - The distance between frames is controlled
    """
    def __init__(self, im_root, gt_root, max_jump, is_bl, subset=None, img_size=None):
        self.im_root = im_root
        self.gt_root = gt_root
        self.max_jump = max_jump
        self.is_bl = is_bl

        self.videos = []
        self.frames = {}

        self.img_size = img_size

        vid_list = sorted(os.listdir(self.im_root))
        # Pre-filtering   at least have 2 frames!!!
        for vid in vid_list:
            if subset is not None:
                if vid not in subset:
                    continue
            frames = sorted(os.listdir(os.path.join(self.im_root, vid)))
            if len(frames) < 2:
                continue
            self.frames[vid] = frames
            self.videos.append(vid)

        print('%d out of %d videos accepted in %s.' % (len(self.videos), len(vid_list), im_root))

        # These set of transform is the same for im/gt pairs, but different among the 3 sampled frames
        self.pair_im_lone_transform = transforms.Compose([
            transforms.ColorJitter(0.01, 0.01, 0.01, 0),
        ])

        self.pair_im_dual_transform = transforms.Compose([
            transforms.RandomAffine(degrees=15, shear=10, interpolation=InterpolationMode.BICUBIC, fill=im_mean),
        ])

        self.pair_gt_dual_transform = transforms.Compose([
            transforms.RandomAffine(degrees=15, shear=10, interpolation=InterpolationMode.NEAREST, fill=0),
        ])

        # These transform are the same for all pairs in the sampled sequence
        self.all_im_lone_transform = transforms.Compose([
            transforms.ColorJitter(0.1, 0.03, 0.03, 0),
            transforms.RandomGrayscale(0.05),
        ])

        if self.is_bl: # this is just for another dataset BL, which is quite large
            # Use a different cropping scheme for the blender dataset because the image size is different
            self.all_im_dual_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomResizedCrop((self.img_size, self.img_size), scale=(0.25, 1.00), interpolation=InterpolationMode.BICUBIC)
            ])

            self.all_gt_dual_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomResizedCrop((self.img_size, self.img_size), scale=(0.25, 1.00), interpolation=InterpolationMode.NEAREST)
            ])
        else:
            self.all_im_dual_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomResizedCrop((self.img_size, self.img_size), scale=(0.36,1.00), interpolation=InterpolationMode.BICUBIC)
            ])

            self.all_gt_dual_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomResizedCrop((self.img_size, self.img_size), scale=(0.36,1.00), interpolation=InterpolationMode.NEAREST)
            ])

        # Final transform without randomness
        self.final_im_transform = transforms.Compose([
            transforms.ToTensor(),
            im_normalization,
        ])

    def __getitem__(self, idx):
        video = self.videos[idx]   # 3464 videos for VOS
        info = {}
        info['name'] = video

        vid_im_path = path.join(self.im_root, video)  # get the image path
        vid_gt_path = path.join(self.gt_root, video)  # get the seg. gt path
        frames = self.frames[video]   # get the sorted frames of the video

        trials = 0
        while trials < 5:   
            info['frames'] = [] # Appended with actual frames

            # Don't want to bias towards beginning/end
            # this_max_jump = min(len(frames), self.max_jump)                     # the max_jump shouldn't > len(frames)
            # start_idx = np.random.randint(len(frames)-this_max_jump+1)          
            # f1_idx = start_idx + np.random.randint(this_max_jump+1) + 1         # + 1 is used to make the f1_idx 100% > start_idx
            # f1_idx = min(f1_idx, len(frames)-1)      # maximum value: len(frames)-2

            # arbitrary sampling it's better (0.3% improvements)
            start_idx = np.random.randint(0, len(frames)-1) # valuues[0, len-1-1]
            if self.max_jump == 1: # select the next frame
                f1_idx = start_idx + 1
            else: # select random frames in a range
                f1_idx = np.random.randint(start_idx+1, min(len(frames), start_idx + self.max_jump))
            
            # f2_idx = f1_idx + np.random.randint(this_max_jump+1) + 1
            # f2_idx = min(f2_idx, len(frames)-this_max_jump//2, len(frames)-1)

            frames_idx = [start_idx, f1_idx]  #f2_idx
            if np.random.rand() < 0.5:
                # Reverse time
                frames_idx = frames_idx[::-1]

            sequence_seed = np.random.randint(2147483647)
            images = []
            masks = []
            target_object = None
            for f_idx in frames_idx:
                jpg_name = frames[f_idx][:-4] + '.jpg'  # for RGB images
                png_name = frames[f_idx][:-4] + '.png'  # for gt
                info['frames'].append(jpg_name)
                

                # images: all_im_dual_transform ---> all_im_lone_transform ---> pair_im_dual_transform ---> final_im_transform
                # gt:     all_gt_dual_transform ---> pair_gt_dual_transform

                # self.all_im_dual_transform = transforms.Compose([
                # transforms.RandomHorizontalFlip(),
                # transforms.RandomResizedCrop((384, 384), scale=(0.36,1.00), interpolation=InterpolationMode.BICUBIC)])

                # self.all_im_lone_transform = transforms.Compose([
                # transforms.ColorJitter(0.1, 0.03, 0.03, 0),
                # transforms.RandomGrayscale(0.05),
                # ])

                # self.pair_im_dual_transform = transforms.Compose([
                # transforms.RandomAffine(degrees=15, shear=10, interpolation=InterpolationMode.BICUBIC, fill=im_mean),
                # ])

                # self.final_im_transform = transforms.Compose([
                # transforms.ToTensor(),
                # im_normalization,
                # ])



                # self.all_gt_dual_transform = transforms.Compose([
                # transforms.RandomHorizontalFlip(),
                # transforms.RandomResizedCrop((384, 384), scale=(0.36,1.00), interpolation=InterpolationMode.NEAREST)
                # ])

                # self.pair_gt_dual_transform = transforms.Compose([
                # transforms.RandomAffine(degrees=15, shear=10, interpolation=InterpolationMode.NEAREST, fill=0),
                # ])


                reseed(sequence_seed)
                this_im = Image.open(path.join(vid_im_path, jpg_name)).convert('RGB')
                this_im = self.all_im_dual_transform(this_im)
                this_im = self.all_im_lone_transform(this_im)
                reseed(sequence_seed)
                this_gt = Image.open(path.join(vid_gt_path, png_name)).convert('P')
                this_gt = self.all_gt_dual_transform(this_gt)

                pairwise_seed = np.random.randint(2147483647)
                reseed(pairwise_seed)
                this_im = self.pair_im_dual_transform(this_im)
                this_im = self.pair_im_lone_transform(this_im)
                reseed(pairwise_seed)
                this_gt = self.pair_gt_dual_transform(this_gt)

                this_im = self.final_im_transform(this_im)
                this_gt = np.array(this_gt)

                images.append(this_im)
                masks.append(this_gt)

            images = torch.stack(images, 0)  # 3, 3, 384, 384

            labels = np.unique(masks[0]) # [] list size=3, we treat 0 sa the first frame, our goal is to track the objects in the following frames
            # Remove background
            labels = labels[labels!=0]

            if self.is_bl:
                # Find large enough labels
                good_lables = []
                for l in labels:
                    pixel_sum = (masks[0]==l).sum()
                    if pixel_sum > 10*10:
                        # OK if the object is always this small
                        # Not OK if it is actually much bigger
                        if pixel_sum > 30*30:
                            good_lables.append(l)
                        elif max((masks[1]==l).sum(), (masks[2]==l).sum()) < 20*20:
                            good_lables.append(l)
                labels = np.array(good_lables, dtype=np.uint8)
            
            if len(labels) == 0:
                target_object = -1 # all black if no objects
                has_second_object = False
                trials += 1
            else:
                target_object = np.random.choice(labels)
                has_second_object = (len(labels) > 1)
                if has_second_object:   # we randomly select another target sa the second object. 
                    labels = labels[labels!=target_object]
                    second_object = np.random.choice(labels)
                break

        masks = np.stack(masks, 0) # convert the list to array masks: 3, 384, 384
        tar_masks = (masks==target_object).astype(np.float32)[:,np.newaxis,:,:] # 3, 1, 384, 384
        if has_second_object:
            sec_masks = (masks==second_object).astype(np.float32)[:,np.newaxis,:,:]
            selector = torch.FloatTensor([1, 1])
        else:
            sec_masks = np.zeros_like(tar_masks)
            selector = torch.FloatTensor([1, 0])

        cls_gt = np.zeros((2, self.img_size, self.img_size), dtype=np.int) # we only use two frames
        cls_gt[tar_masks[:,0] > 0.5] = 1
        cls_gt[sec_masks[:,0] > 0.5] = 2

        data = {
            'rgb': images,
            'gt': tar_masks,
            'cls_gt': cls_gt,
            'sec_gt': sec_masks,
            'selector': selector,
            'info': info,
        }

        return data

    def __len__(self):
        return len(self.videos)





class VOSDataset_triplet(Dataset):
    """
    Works for DAVIS/YouTubeVOS/BL30K training
    For each sequence:
    - Pick three frames
    - Pick two objects
    - Apply some random transforms that are the same for all frames
    - Apply random transform to each of the frame
    - The distance between frames is controlled
    """
    def __init__(self, im_root, gt_root, max_jump, is_bl, subset=None, img_size=None):
        self.im_root = im_root
        self.gt_root = gt_root
        self.max_jump = max_jump
        self.is_bl = is_bl

        print('VOSDataset_triplet')

        self.videos = []
        self.frames = {}

        self.img_size = img_size

        vid_list = sorted(os.listdir(self.im_root))
        # Pre-filtering   at least have 2 frames!!!
        for vid in vid_list:
            if subset is not None:
                if vid not in subset:
                    continue
            frames = sorted(os.listdir(os.path.join(self.im_root, vid)))
            if len(frames) < 3:
                continue
            self.frames[vid] = frames
            self.videos.append(vid)

        print('%d out of %d videos accepted in %s.' % (len(self.videos), len(vid_list), im_root))

        # These set of transform is the same for im/gt pairs, but different among the 3 sampled frames
        self.pair_im_lone_transform = transforms.Compose([
            transforms.ColorJitter(0.01, 0.01, 0.01, 0),
        ])

        self.pair_im_dual_transform = transforms.Compose([
            transforms.RandomAffine(degrees=15, shear=10, interpolation=InterpolationMode.BICUBIC, fill=im_mean),
        ])

        self.pair_gt_dual_transform = transforms.Compose([
            transforms.RandomAffine(degrees=15, shear=10, interpolation=InterpolationMode.NEAREST, fill=0),
        ])

        # These transform are the same for all pairs in the sampled sequence
        self.all_im_lone_transform = transforms.Compose([
            transforms.ColorJitter(0.1, 0.03, 0.03, 0),
            transforms.RandomGrayscale(0.05),
        ])

        if self.is_bl: # this is just for another dataset BL, which is quite large
            # Use a different cropping scheme for the blender dataset because the image size is different
            self.all_im_dual_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomResizedCrop((self.img_size, self.img_size), scale=(0.25, 1.00), interpolation=InterpolationMode.BICUBIC)
            ])

            self.all_gt_dual_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomResizedCrop((self.img_size, self.img_size), scale=(0.25, 1.00), interpolation=InterpolationMode.NEAREST)
            ])
        else:
            self.all_im_dual_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomResizedCrop((self.img_size, self.img_size), scale=(0.36,1.00), interpolation=InterpolationMode.BICUBIC)
            ])

            self.all_gt_dual_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomResizedCrop((self.img_size, self.img_size), scale=(0.36,1.00), interpolation=InterpolationMode.NEAREST)
            ])

        # Final transform without randomness
        self.final_im_transform = transforms.Compose([
            transforms.ToTensor(),
            im_normalization,
        ])

    def __getitem__(self, idx):
        video = self.videos[idx]   # 3464 videos for VOS
        info = {}
        info['name'] = video

        vid_im_path = path.join(self.im_root, video)  # get the image path
        vid_gt_path = path.join(self.gt_root, video)  # get the seg. gt path
        frames = self.frames[video]   # get the sorted frames of the video

        trials = 0
        while trials < 5:   
            info['frames'] = [] # Appended with actual frames

            # XMEM sampling
            num_frames = 3
            length = len(frames)
            this_max_jump = min(len(frames), self.max_jump)

            # iterative sampling
            frames_idx = [np.random.randint(length)]
            acceptable_set = set(range(max(0, frames_idx[-1]-this_max_jump), min(length, frames_idx[-1]+this_max_jump+1))).difference(set(frames_idx))
            while(len(frames_idx) < num_frames):
                idx = np.random.choice(list(acceptable_set))
                frames_idx.append(idx)
                new_set = set(range(max(0, frames_idx[-1]-this_max_jump), min(length, frames_idx[-1]+this_max_jump+1)))
                acceptable_set = acceptable_set.union(new_set).difference(set(frames_idx))

            frames_idx = sorted(frames_idx)

            if np.random.rand() < 0.5:
                # Reverse time
                frames_idx = frames_idx[::-1]

            sequence_seed = np.random.randint(2147483647)
            images = []
            masks = []
            target_object = None
            for f_idx in frames_idx:
                jpg_name = frames[f_idx][:-4] + '.jpg'  # for RGB images
                png_name = frames[f_idx][:-4] + '.png'  # for gt
                info['frames'].append(jpg_name)
                

                # images: all_im_dual_transform ---> all_im_lone_transform ---> pair_im_dual_transform ---> final_im_transform
                # gt:     all_gt_dual_transform ---> pair_gt_dual_transform

                # self.all_im_dual_transform = transforms.Compose([
                # transforms.RandomHorizontalFlip(),
                # transforms.RandomResizedCrop((384, 384), scale=(0.36,1.00), interpolation=InterpolationMode.BICUBIC)])

                # self.all_im_lone_transform = transforms.Compose([
                # transforms.ColorJitter(0.1, 0.03, 0.03, 0),
                # transforms.RandomGrayscale(0.05),
                # ])

                # self.pair_im_dual_transform = transforms.Compose([
                # transforms.RandomAffine(degrees=15, shear=10, interpolation=InterpolationMode.BICUBIC, fill=im_mean),
                # ])

                # self.final_im_transform = transforms.Compose([
                # transforms.ToTensor(),
                # im_normalization,
                # ])



                # self.all_gt_dual_transform = transforms.Compose([
                # transforms.RandomHorizontalFlip(),
                # transforms.RandomResizedCrop((384, 384), scale=(0.36,1.00), interpolation=InterpolationMode.NEAREST)
                # ])

                # self.pair_gt_dual_transform = transforms.Compose([
                # transforms.RandomAffine(degrees=15, shear=10, interpolation=InterpolationMode.NEAREST, fill=0),
                # ])


                reseed(sequence_seed)
                this_im = Image.open(path.join(vid_im_path, jpg_name)).convert('RGB')
                this_im = self.all_im_dual_transform(this_im)
                this_im = self.all_im_lone_transform(this_im)
                reseed(sequence_seed)
                this_gt = Image.open(path.join(vid_gt_path, png_name)).convert('P')
                this_gt = self.all_gt_dual_transform(this_gt)

                pairwise_seed = np.random.randint(2147483647)
                reseed(pairwise_seed)
                this_im = self.pair_im_dual_transform(this_im)
                this_im = self.pair_im_lone_transform(this_im)
                reseed(pairwise_seed)
                this_gt = self.pair_gt_dual_transform(this_gt)

                this_im = self.final_im_transform(this_im)
                this_gt = np.array(this_gt)

                images.append(this_im)
                masks.append(this_gt)

            images = torch.stack(images, 0)  # 3, 3, 384, 384

            labels = np.unique(masks[0]) # [] list size=3, we treat 0 sa the first frame, our goal is to track the objects in the following frames
            # Remove background
            labels = labels[labels!=0]

            if self.is_bl:
                # Find large enough labels
                good_lables = []
                for l in labels:
                    pixel_sum = (masks[0]==l).sum()
                    if pixel_sum > 10*10:
                        # OK if the object is always this small
                        # Not OK if it is actually much bigger
                        if pixel_sum > 30*30:
                            good_lables.append(l)
                        elif max((masks[1]==l).sum(), (masks[2]==l).sum()) < 20*20:
                            good_lables.append(l)
                labels = np.array(good_lables, dtype=np.uint8)
            
            if len(labels) == 0:
                target_object = -1 # all black if no objects
                has_second_object = False
                trials += 1
            else:
                target_object = np.random.choice(labels)
                has_second_object = (len(labels) > 1)
                if has_second_object:   # we randomly select another target sa the second object. 
                    labels = labels[labels!=target_object]
                    second_object = np.random.choice(labels)
                break

        masks = np.stack(masks, 0) # convert the list to array masks: 3, 384, 384
        tar_masks = (masks==target_object).astype(np.float32)[:,np.newaxis,:,:] # 3, 1, 384, 384
        if has_second_object:
            sec_masks = (masks==second_object).astype(np.float32)[:,np.newaxis,:,:]
            selector = torch.FloatTensor([1, 1])
        else:
            sec_masks = np.zeros_like(tar_masks)
            selector = torch.FloatTensor([1, 0])

        cls_gt = np.zeros((3, self.img_size, self.img_size), dtype=np.int) # we only use two frames
        cls_gt[tar_masks[:,0] > 0.5] = 1
        cls_gt[sec_masks[:,0] > 0.5] = 2

        data = {
            'rgb': images,
            'gt': tar_masks,
            'cls_gt': cls_gt,
            'sec_gt': sec_masks,
            'selector': selector,
            'info': info,
        }

        return data

    def __len__(self):
        return len(self.videos)