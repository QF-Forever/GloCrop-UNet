import argparse
import os
import time  # 新增：用于计时
from glob import glob
import random
import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn
import yaml
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import archs
from dataset import Dataset
from metrics import iou_score, indicators
from utils import AverageMeter
from albumentations import RandomRotate90, Resize

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='GloCrop-UNet', help='model name')
    args = parser.parse_args()
    return args

def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# 计算模型参数量（以百万为单位）
def count_parameters(model):
    return sum(p.numel() for p in model.parameters()) / 1e6

def main():
    seed_torch()
    args = parse_args()

    with open('models/%s/config.yml' % args.name, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    print('-' * 20)
    for key in config.keys():
        print('%s: %s' % (key, str(config[key])))
    print('-' * 20)

    cudnn.benchmark = True

    print("=> creating model %s" % config['arch'])
    model = archs.__dict__[config['arch']](num_classes=config['num_classes'],
                                           input_channels=config['input_channels'],
                                           deep_supervision=config['deep_supervision'])

    # 计算并打印模型参数量
    model_params = count_parameters(model)
    print(f"==> Model Parameters: {model_params:.2f}M")

    model = model.cuda()
    model.load_state_dict(torch.load('models/%s/model.pth' % args.name))
    model.eval()

    # --------------------------新增：单张图片处理速度计算--------------------------
    # 1. 模型预热（避免首次推理因CUDA初始化导致的时间偏差）
    input_size = (1, config['input_channels'], config['input_h'], config['input_w'])  # 单张图片的输入尺寸
    warmup_input = torch.randn(input_size).cuda()  # 生成随机输入张量
    with torch.no_grad():
        for _ in range(10):  # 预热10次
            model(warmup_input)
    torch.cuda.synchronize()  # 等待CUDA操作完成，确保计时准确

    # 2. 初始化计时器，用于记录单张图片的推理时间
    infer_time_meter = AverageMeter()  # 记录平均推理时间
    # ----------------------------------------------------------------------------

    # 数据加载
    img_ids = glob(os.path.join('inputs', config['dataset'], 'images', '*' + config['img_ext']))
    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]

    _, val_img_ids = train_test_split(img_ids, test_size=0.2, random_state=41)

    val_transform = Compose([
        Resize(config['input_h'], config['input_w']),
        transforms.Normalize(),
    ])

    val_dataset = Dataset(
        img_ids=val_img_ids,
        img_dir=os.path.join('inputs', config['dataset'], 'images'),
        mask_dir=os.path.join('inputs', config['dataset'], 'masks'),
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        transform=val_transform)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False)

    # 指标记录器
    iou_avg_meter = AverageMeter()
    dice_avg_meter = AverageMeter()
    hd_avg_meter = AverageMeter()
    hd95_avg_meter = AverageMeter()
    recall_avg_meter = AverageMeter()
    specificity_avg_meter = AverageMeter()
    precision_avg_meter = AverageMeter()

    # 创建输出目录
    for c in range(config['num_classes']):
        os.makedirs(os.path.join('outputs', args.name, str(c)), exist_ok=True)

    with torch.no_grad():
        for input, target, meta in tqdm(val_loader, total=len(val_loader)):
            input = input.cuda()
            target = target.cuda()

            # --------------------------新增：计算单张图片推理时间--------------------------
            # 对批次中的每张图片单独计时（避免批次处理的并行影响）
            batch_size = input.size(0)
            for i in range(batch_size):
                single_input = input[i:i+1]  # 取出单张图片（保持batch维度为1）
                torch.cuda.synchronize()  # 确保之前的操作完成
                start_time = time.time()  # 开始计时
                output = model(single_input)  # 单张图片推理
                torch.cuda.synchronize()  # 等待推理完成
                end_time = time.time()  # 结束计时
                infer_time = (end_time - start_time) * 1000  # 转换为毫秒
                infer_time_meter.update(infer_time, 1)  # 累计到平均计时器
            # ----------------------------------------------------------------------------

            # 计算批次的输出和指标（使用完整批次提高效率）
            output = model(input)
            iou, dice, hd, hd95, recall, specificity, precision = indicators(output, target)
            iou_avg_meter.update(iou, input.size(0))
            dice_avg_meter.update(dice, input.size(0))
            hd_avg_meter.update(hd, input.size(0))
            hd95_avg_meter.update(hd95, input.size(0))
            recall_avg_meter.update(recall, input.size(0))
            specificity_avg_meter.update(specificity, input.size(0))
            precision_avg_meter.update(precision, input.size(0))

            # 保存输出结果
            output = torch.sigmoid(output).cpu().numpy()
            output[output >= 0.5] = 1
            output[output < 0.5] = 0

            for i in range(len(output)):
                for c in range(config['num_classes']):
                    cv2.imwrite(os.path.join('outputs', args.name, str(c), meta['img_id'][i] + '.png'),
                                (output[i, c] * 255).astype('uint8'))

    # 打印所有指标（包含新增的推理时间）
    print('-' * 50)
    print('IoU: %.4f' % iou_avg_meter.avg)
    print('Dice: %.4f' % dice_avg_meter.avg)
    print('Hd: %.4f' % hd_avg_meter.avg)
    print('Hd95: %.4f' % hd95_avg_meter.avg)
    print('Recall: %.4f' % recall_avg_meter.avg)
    print('Specificity: %.4f' % specificity_avg_meter.avg)
    print('Precision: %.4f' % precision_avg_meter.avg)
    print(f'Average inference time per image: {infer_time_meter.avg:.2f} ms')  # 单张图片平均处理时间（毫秒）
    print(f'FPS (Frames Per Second): {1000 / infer_time_meter.avg:.2f}')  # 每秒可处理的图片数量
    print('-' * 50)

    torch.cuda.empty_cache()

if __name__ == '__main__':
    main()