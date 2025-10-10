import os
from os import path
import logging

import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image
import numpy as np
import cv2
from PIL import Image, ImageFilter, ImageOps
from oasis.dataset.utils import im_mean, reseed
from oasis.dataset.tps import random_tps_warp

log = logging.getLogger()
local_rank = int(os.environ["LOCAL_RANK"])


class SyntheticVideoDataset(Dataset):
    """
    Note: data normalization happens within the model instead of here
    Generate pseudo VOS data by applying random transforms on static images.

    parameters is a list of tuples
        (data_root, how data is structured (method 0 or 1), and an oversample multiplier)

    Method 0 - FSS style (class/1.jpg class/1.png)
    Method 1 - Others style (XXX.jpg XXX.png)
    """

    def __init__(self, parameters, *, size=384, seq_length=3, max_num_obj=1):
        self.seq_length = seq_length
        self.max_num_obj = max_num_obj
        self.size = size

        self.im_list = []
        for parameter in parameters:
            root, method, multiplier = parameter
            if method == 0:
                # Get images
                classes = os.listdir(root)
                for c in classes:
                    imgs = os.listdir(path.join(root, c))
                    jpg_list = [im for im in imgs if "jpg" in im[-3:].lower()]

                    joint_list = [path.join(root, c, im) for im in jpg_list]
                    self.im_list.extend(joint_list * multiplier)

            elif method == 1:
                self.im_list.extend(
                    [path.join(root, im) for im in os.listdir(root) if ".jpg" in im]
                    * multiplier
                )

        if local_rank == 0:
            log.info(
                f"SyntheticVideoDataset: {len(self.im_list)} images found in total."
            )

        # The frame transforms are the same for each of the pairs,
        # but different for different pairs in the sequence
        self.frame_image_transform = transforms.Compose(
            [
                transforms.ColorJitter(0.1, 0.05, 0.05, 0),
            ]
        )

        self.frame_image_dual_transform = transforms.Compose(
            [
                transforms.RandomAffine(
                    degrees=20,
                    scale=(0.5, 2.0),
                    shear=10,
                    interpolation=InterpolationMode.BILINEAR,
                    fill=im_mean,
                ),
                transforms.Resize(self.size, InterpolationMode.BILINEAR),
                transforms.RandomCrop(
                    (self.size, self.size), pad_if_needed=True, fill=im_mean
                ),
            ]
        )

        self.frame_mask_dual_transform = transforms.Compose(
            [
                transforms.RandomAffine(
                    degrees=20,
                    scale=(0.5, 2.0),
                    shear=10,
                    interpolation=InterpolationMode.NEAREST,
                    fill=0,
                ),
                transforms.Resize(self.size, InterpolationMode.NEAREST),
                transforms.RandomCrop(
                    (self.size, self.size), pad_if_needed=True, fill=0
                ),
            ]
        )

        # The sequence transforms are the same for all pairs in the sampled sequence
        self.sequence_image_only_transform = transforms.Compose(
            [
                transforms.ColorJitter(0.1, 0.05, 0.05, 0.05),
                transforms.RandomGrayscale(0.05),
            ]
        )

        self.sequence_image_dual_transform = transforms.Compose(
            [
                transforms.RandomAffine(degrees=0, scale=(0.5, 2.0), fill=im_mean),
                transforms.RandomHorizontalFlip(),
            ]
        )

        self.sequence_mask_dual_transform = transforms.Compose(
            [
                transforms.RandomAffine(degrees=0, scale=(0.5, 2.0), fill=0),
                transforms.RandomHorizontalFlip(),
            ]
        )

        self.output_image_transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

        self.output_mask_transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

    def _get_sample(self, idx):
        im = Image.open(self.im_list[idx]).convert("RGB")
        gt = Image.open(self.im_list[idx][:-3] + "png").convert("L")

        this_im_gray = im.convert("L")

        this_gt_gray = gt.convert("L")

        sequence_seed = np.random.randint(2147483647)

        images = []
        masks = []
        edges = []
        im_edges = []
        for _ in range(self.seq_length):

            this_gt_binary = ImageOps.autocontrast(this_gt_gray)
            this_gt_edges = this_gt_binary.filter(ImageFilter.FIND_EDGES)

            reseed(sequence_seed)
            this_im = self.sequence_image_dual_transform(im)
            this_im = self.sequence_image_only_transform(this_im)
            reseed(sequence_seed)
            this_gt = self.sequence_mask_dual_transform(gt)

            reseed(sequence_seed)
            this_gt_edges = self.sequence_mask_dual_transform(this_gt_edges)

            this_im_gray_np = np.array(this_im_gray)
            this_im_edges = cv2.Canny(this_im_gray_np, 50, 200, L2gradient=True)
            this_im_edges = Image.fromarray(this_im_edges)

            reseed(sequence_seed)
            this_im_edges = self.sequence_mask_dual_transform(this_im_edges)

            pairwise_seed = np.random.randint(2147483647)
            reseed(pairwise_seed)
            this_im = self.frame_image_dual_transform(this_im)
            this_im = self.frame_image_transform(this_im)
            reseed(pairwise_seed)
            this_gt = self.frame_mask_dual_transform(this_gt)

            reseed(pairwise_seed)
            this_gt_edges = self.frame_mask_dual_transform(this_gt_edges)

            reseed(pairwise_seed)
            this_im_edges = self.frame_mask_dual_transform(this_im_edges)

            # Use TPS only some of the times
            # Not because TPS is bad -- just that it is too slow and I need to speed up data loading
            if np.random.rand() < 0.33:
                this_im, this_gt = random_tps_warp(this_im, this_gt, scale=0.02)

            this_im = self.output_image_transform(this_im)
            this_gt = self.output_mask_transform(this_gt)
            this_im_edges = self.output_image_transform(this_im_edges)

            images.append(this_im)
            masks.append(this_gt)
            edges.append(this_gt_edges)
            im_edges.append(this_im_edges)

        images = torch.stack(images, 0)
        masks = torch.stack(masks, 0)
        edges = np.stack(edges, 0)
        im_edges = torch.stack(im_edges, 0)

        return images, masks.numpy(), edges, im_edges

    def __getitem__(self, idx):
        additional_objects = np.random.randint(self.max_num_obj)
        indices = [idx, *np.random.randint(self.__len__(), size=additional_objects)]

        # Sample from multiple images and merge them together onto a training sample
        merged_images = None
        merged_masks = np.zeros((self.seq_length, self.size, self.size), dtype=np.int64)
        merged_edges = np.zeros((self.seq_length, self.size, self.size), dtype=float)
        merged_im_edges = torch.zeros(
            (self.seq_length, 1, self.size, self.size), dtype=torch.float
        )  # Now in torch format

        for i, list_id in enumerate(indices):
            images, masks, edges, im_edges = self._get_sample(list_id)
            edges = np.clip(edges, 0.0, 1.0, dtype=float)
            im_edges = torch.clamp(im_edges, 0.0, 1.0).to(dtype=torch.float)
            if merged_images is None:
                merged_images = images
            else:
                merged_images = merged_images * (1 - masks) + images * masks
            merged_masks[masks[:, 0] > 0.5] = i + 1
            merged_edges = np.maximum(merged_edges, edges)  # Merge edges by taking max
            merged_im_edges = torch.max(
                merged_im_edges, im_edges
            )  # torch max for tensors

        masks = merged_masks

        labels = np.unique(masks[0])
        # Remove background
        labels = labels[labels != 0]
        target_objects = labels.tolist()

        # Generate one-hot ground-truth
        cls_gt = np.zeros((self.seq_length, self.size, self.size), dtype=np.int64)
        first_frame_gt = np.zeros(
            (1, self.max_num_obj, self.size, self.size), dtype=np.int64
        )
        for i, l in enumerate(target_objects):
            this_mask = masks == l
            cls_gt[this_mask] = i + 1
            first_frame_gt[0, i] = this_mask[0]
        cls_gt = np.expand_dims(cls_gt, 1)

        info = {}
        info["name"] = self.im_list[idx]
        info["num_objects"] = max(1, len(target_objects))

        # 1 if object exist, 0 otherwise
        selector = [
            1 if i < info["num_objects"] else 0 for i in range(self.max_num_obj)
        ]
        selector = torch.FloatTensor(selector)

        data = {
            "rgb": merged_images,
            "first_frame_gt": first_frame_gt,
            "edges": merged_edges,
            "im_edges": merged_im_edges,
            "cls_gt": cls_gt,
            "selector": selector,
            "info": info,
        }

        return data

    def __len__(self):
        return len(self.im_list)
