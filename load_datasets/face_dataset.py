import sys
import os
import torch
from torch.utils import data
from PIL import Image
import numpy as np

if 'win' in sys.platform:
    split_flag = '\\'
else:
    split_flag = '/'

class FaceDataSet(data.Dataset):

    def __init__(self, dataset_name, dataset_dir, transform=None, tasks=None, network=None, seed=-1):
        #所有图片的绝对路径
        self.dataset_name = dataset_name
        self.dataset_dir = dataset_dir
        self.transform = transform
        self.tasks = tasks
        self.network = network
        self.seed = seed
        self.t = 0

        # split dataset
        self.archive = []
        for task_idx, task in enumerate(tasks):
            img_list = []
            targets = []
            for task_id in task:
                if self.dataset_name == 'yaleB':
                    sub_dir = os.path.join(dataset_dir, 'yaleB'+str(task_id))
                else:
                    sub_dir = os.path.join(dataset_dir, self.dataset_name+'-'+str(task_id))

                img_list += [os.path.join(sub_dir,k) for k in os.listdir(sub_dir)]
            for img_path in img_list:
                # 此处的target是真实的类别编号1~subjct_nums
                if self.dataset_name == 'yaleB':
                    target = int(img_path.split(split_flag)[-2].split('B')[-1])
                else:
                    target = int(img_path.split(split_flag)[-2].split('-')[-1])

                targets.append(target)
            targets = np.array(targets)
            # srcs: 标识当前样本属于哪一个task，在测试的时候用
            srcs    = np.full_like(targets, task_idx, dtype=np.uint8) # length:len(data)
            self.archive.append((img_list, targets, srcs))

    def load_new_task(self):
        '''
        载入新的任务
        '''
        self.test_data_label = self.archive[self.t]

    def __getitem__(self, index):

        tmp_img_path_list = self.test_data_label[0]
        targets = np.array(self.test_data_label[1])
        srcs = np.array(self.test_data_label[2])
        img_path, target, src = tmp_img_path_list[index], targets[index], srcs[index]
        with Image.open(img_path) as pil_img:
            if self.transform is not None:
                data=self.transform(pil_img)
                return data, target, src

    def __len__(self):
        data_len = 0
        data_len = len(self.test_data_label[1])
        return data_len