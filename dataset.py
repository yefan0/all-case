from torch.utils.data import DataLoader
import torch
import numpy as np
from itertools import combinations


def make_positive_indices(labels):
    positive_indices = []
    persons = np.unique(labels)
    for person in persons:
        img_indices = np.where(labels== person)[0]
        pair_indices = list(combinations(img_indices, 2))
        positive_indices.extend([[pair, person] for pair in pair_indices])
    return positive_indices


class SiameseDataLoader(DataLoader):
    def __init__(self, positve_indices, batch_size, neg_nums, all_images, all_labels, shuffle):
        super().__init__(positve_indices, batch_size, shuffle, collate_fn=self._collate)
        self.neg_nums = neg_nums
        self.all_images = all_images
        self.all_labels = all_labels

    def _collate(self, batch):
        left_images = []
        right_images = []
        pair_labels = []
        for (img_id_l, img_id_r), person in batch:
            left_images.append(self.all_images[img_id_l])
            right_images.append(self.all_images[img_id_r])
            pair_labels.append([1])                     # 1表示正对
            if self.neg_nums > 0:
                neg_indices = np.random.choice(np.where(self.all_labels != person)[0], self.neg_nums * 2, replace=False)
                neg_indices = neg_indices.reshape(-1, 2)
                for neg_id_l, neg_id_r in neg_indices:
                    # 为左图生成负对
                    left_images.append(self.all_images[img_id_l])
                    right_images.append(self.all_images[neg_id_l])
                    pair_labels.append([0])             # 0表示负对
                    # 为右图生成负对
                    left_images.append(self.all_images[img_id_r])
                    right_images.append(self.all_images[neg_id_r])
                    pair_labels.append([0])             # 0表示负对

        left_images = torch.tensor(np.array(left_images), dtype=torch.float32)
        right_images = torch.tensor(np.array(right_images), dtype=torch.float32)
        return left_images, right_images, torch.tensor(pair_labels)
