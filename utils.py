import torch
from torchvision import transforms
from PIL import Image
import load_datasets, models
import torch.nn.functional as F
import torch.nn as nn


dataset_stats_rgb = {
    'yaleB': {'mean': (0.5091806, 0.5091806, 0.5091806), 'std' : (0.15323389, 0.15323389, 0.15323389)},
    'FEI':  {'mean': (0.50870687, 0.50870687, 0.50870687), 'std' : (0.14216691, 0.14216691, 0.14216691)},
    'PIE':  {'mean': (0.50574565, 0.50574565, 0.50574565), 'std' : (0.12802121, 0.12802121, 0.12802121)}
                }

def get_transform(dataset='yaleB', phase='test', isPreNet=False):
    transform_list = []
    if phase == 'train':
        transform_list.extend([
            transforms.RandomCrop(224, padding=4),
            transforms.RandomHorizontalFlip(), # 水平镜像
                              ])
    transform_list.extend([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)), # convert 1 channel to 3 channel
        transforms.Normalize(dataset_stats_rgb[dataset]['mean'], dataset_stats_rgb[dataset]['std']),
    ])
    
    return transforms.Compose(transform_list)

def create_model(net, label_len, gpu):
    """
    根据net名称建立对应的模型
    """
    if net == 'resnet_pre':
        print('ResNet18')
        model = models.resnet_pre(label_len = label_len)
    if len(gpu) > 1:
        model = nn.DataParallel(model)
    return model