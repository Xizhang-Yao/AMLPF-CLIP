import argparse
import copy
import csv
import json
import logging
import random
import time
from datetime import datetime
import math
import numpy as np
import pytz
import torch.nn as nn
import timm
import torch
import os
from scipy.stats import entropy
from matplotlib import pyplot as plt
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import warnings
from PIL import Image
from torchmetrics import Accuracy, Precision, Recall, F1Score, AUROC, CohenKappa, MatthewsCorrCoef
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from CLIP import clip
import clip as clip_module
from torch.nn import functional as F

from models import UNI_CLIP_Model, UNI_Classifier
from utils import _preprocess2, _preprocess3
# 省略警告
warnings.filterwarnings("ignore")
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import entropy
import numpy as np


def beijing(sec, what):
    current_time = datetime.utcnow()
    # 设置东八区时区
    eastern_eight = pytz.timezone('Asia/Shanghai')
    # 将当前时间转换为东八区时间
    current_time_eight = current_time.astimezone(eastern_eight)
    return current_time_eight.timetuple()


def set_logger(logfile_name='logfile.log', log_name='my_logger'):
    logging.Formatter.converter = beijing
    # 创建一个logger实例
    logger = logging.getLogger(log_name)
    logger.setLevel(logging.INFO)  # 设置日志级别为INFO

    # 定义日志格式
    formatter = logging.Formatter('%(asctime)s:   %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    # 创建控制台处理器（StreamHandler），用于将日志输出到控制台
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)  # 设置流处理器的日志级别为INFO
    console_handler.setFormatter(formatter)  # 应用日志格式

    # 创建文件处理器（FileHandler），用于将日志输出到文件
    file_handler = logging.FileHandler(filename=logfile_name)
    file_handler.setLevel(logging.INFO)  # 设置文件处理器的日志级别为INFO
    file_handler.setFormatter(formatter)  # 应用日志格式

    # 将处理器添加到logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger


def get_timestamp_now():
    current_time = datetime.now()
    # 设置东八区时区
    eastern_eight = pytz.timezone('Asia/Shanghai')
    # 将当前时间转换为东八区时间
    current_time_eight = current_time.astimezone(eastern_eight)
    IMESTAMP = "{0:%Y-%m-%d-%H:%M:%S}".format(current_time_eight)
    # print(f'The current time is {IMESTAMP}')
    return IMESTAMP


def freeze_model(f_model, opt):
    f_model.logit_scale.requires_grad = False
    if opt == 0:  # do nothing
        return
    elif opt == 1:  # freeze text encoder
        for p in f_model.token_embedding.parameters():
            p.requires_grad = False
        for p in f_model.transformer.parameters():
            p.requires_grad = False
        f_model.positional_embedding.requires_grad = False
        f_model.text_projection.requires_grad = False
        for p in f_model.ln_final.parameters():
            p.requires_grad = False
    elif opt == 2:  # freeze visual encoder
        for p in f_model.visual.parameters():
            p.requires_grad = False
    elif opt == 3:
        for p in f_model.parameters():
            p.requires_grad = False
    elif opt == 4:
        for p in f_model.parameters():
            p.requires_grad = True





class Standard_Dataset_Patch_Resize(Dataset):
    def __init__(self, args=None, txt_path=None, transform=None, num_classes=2, num_patch=8, test=False):
        super(Standard_Dataset_Patch_Resize, self).__init__()
        self.txt_path = txt_path
        self.transform = transform
        with open(self.txt_path, 'r') as f:
            self.lines = f.readlines()
        self.num_classes = num_classes
        self.num_patch = num_patch
        self.test = test
        self.args = args

    def __len__(self):
        return len(self.lines)
        # return 100
    def __getitem__(self, index):
        line = self.lines[index]
        line = line.strip().split()
        image_path, label = line[0], int(line[1])
        I = Image.open(image_path).convert('RGB')

        # one_hot_label = torch.zeros(self.num_classes)
        # one_hot_label[label] = 1
        I = self.transform(I)
        I = I.unsqueeze(0)
        n_channels = 3
        kernel_h = 224
        kernel_w = 224
        if (I.size(2) >= 1024) | (I.size(3) >= 1024):
            step = 48
        else:
            step = 32

        patches = I.unfold(2, kernel_h, step).unfold(3, kernel_w, step).permute(2, 3, 0, 1, 4, 5).reshape(-1,
                                                                                                          n_channels,
                                                                                                          kernel_h,
                                                                                                          kernel_w)

        assert patches.size(0) >= self.num_patch
        if self.test:
            sel_step = patches.size(0) // self.num_patch
            sel = torch.zeros(self.num_patch)
            for i in range(self.num_patch):
                sel[i] = sel_step * i
            sel = sel.long()
        else:
            sel = torch.randint(low=0, high=patches.size(0), size=(self.num_patch,))
            sel = sel.long()
        patches = patches[sel, ...]

        if self.args.model == "CLIP_Patch_NoResize":
            return patches, label

        I_resized = F.interpolate(I, size=(kernel_h, kernel_w), mode='bilinear', align_corners=False)
        patches = torch.cat([patches, I_resized], dim=0)

        return patches, label,image_path


def train_Classifier(args, split_num):
    logger.info(args)
    logger.info(f"split_num: {split_num}")
    model = UNI_Classifier()
    preprocess2 = _preprocess2()
    preprocess3 = _preprocess3()
    dataset_train = Standard_Dataset_Patch_Resize(args=args,
                                                  txt_path=f'/public/yxz/UNI/Datasplit/{args.Dataset}/{split_num}/train.txt',
                                                  transform=preprocess3, num_classes=num_classes,
                                                  num_patch=args.num_patch)
    dataset_test = Standard_Dataset_Patch_Resize(args=args,
                                                 txt_path=f'/public/yxz/UNI/Datasplit/{args.Dataset}/{split_num}/test.txt',
                                                 transform=preprocess2, num_classes=num_classes,
                                                 num_patch=args.num_patch, test=True)
    batch_size = 24
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=10)
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=10)
    criterion = nn.CrossEntropyLoss()
    criterion_test = nn.CrossEntropyLoss(reduction="sum")  # 用于测试集的loss计算,因为测试集的batch_size不为1
    lr_init = 1e-5
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_init)
    model.train()
    model.to("cuda")
    writer = SummaryWriter(
        f"tensorboard/{args.Dataset}/{args.model}/SplitNum_{split_num}_Classifier-{args.description}-{get_timestamp_now()}")
    # 初始化度量计算对象

    accuracy = Accuracy(task='MULTICLASS', num_classes=num_classes)
    precision = Precision(task='MULTICLASS', num_classes=num_classes, average='macro')  # num_classes 是类别的数量
    recall = Recall(task='MULTICLASS', num_classes=num_classes, average='macro')
    f1score = F1Score(task='MULTICLASS', num_classes=num_classes, average='macro')
    auc = AUROC(task='MULTICLASS', num_classes=num_classes)
    Accuracy_Per_Class = Accuracy(task='MULTICLASS', num_classes=num_classes, average=None)
    best_epoch = 0
    best_acc = 0.0
    best_precision = 0.0
    best_recall = 0.0
    best_f1 = 0.0
    best_auc = 0.0
    epoch_num = args.epoch_num_classification
    best_state_dict = model.state_dict().copy()
    logger.info(args)
    logger.info(
        f"{'Epoch':<10}{'Train Loss':<15}{'Train Acc':<15}{'Test Loss':<15}{'Test Acc':<15}{'learning rate':<15}")
    for epoch in range(1, epoch_num + 1):
        if epoch == 1:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 3e-4
        if epoch == 2:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 1e-4
        if epoch == 20:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 5e-5
        model.train()
        labels_train = []
        labels_train_pred = []
        y_train_pred_logits = []
        loss_train = 0.0
        for i, (images, labels) in enumerate(dataloader_train):
            images = images.to("cuda")
            labels = labels.to("cuda")
            optimizer.zero_grad()
            y_pred_logits, y_pred = model(images)
            loss = criterion(y_pred_logits, labels)
            loss.backward()
            optimizer.step()
            # 评价分类器的性能，用准确率、精确率、召回率、F1值等指标
            labels_train.extend(labels.cpu().tolist())
            labels_train_pred.extend(y_pred.cpu().tolist())
            y_train_pred_logits.extend(y_pred_logits.cpu().tolist())
            loss_train += loss.item()
        labels_train_pred = torch.argmax(torch.tensor(labels_train_pred), dim=1)
        labels_train = torch.tensor(labels_train)
        accuracy_train = accuracy(labels_train_pred, labels_train)
        y_train_pred_logits = torch.tensor(y_train_pred_logits)

        writer.add_scalars('Loss', {f'{args.Dataset}_{args.model}_Train_loss': loss_train / len(dataloader_train)},
                           epoch)

        labels_test = []
        labels_test_pred = []
        loss_test = 0.0
        model.eval()
        with torch.no_grad():
            for i, (images, labels) in enumerate(dataloader_test):
                images = images.to("cuda")
                labels = labels.to("cuda")
                y_pred_logits, y_pred = model.test(images)
                loss = criterion_test(y_pred_logits, labels)
                labels_test.extend(labels.cpu().tolist())
                labels_test_pred.extend(y_pred.cpu().tolist())
                loss_test += loss.item()
            labels_test_pred_original = torch.tensor(labels_test_pred)
            labels_test_pred = torch.argmax(torch.tensor(labels_test_pred), dim=1)
            labels_test = torch.tensor(labels_test)
            accuracy_test = accuracy(labels_test_pred, labels_test)
            precision_test = precision(labels_test_pred, labels_test)
            recall_test = recall(labels_test_pred, labels_test)
            f1_test = f1score(labels_test_pred, labels_test)
            auc_test = auc(labels_test_pred_original, labels_test)

            if accuracy_test > best_acc:
                best_epoch = epoch
                best_acc = accuracy_test
                best_precision = precision_test
                best_recall = recall_test
                best_f1 = f1_test
                best_auc = auc_test
                best_state_dict = copy.deepcopy(model.state_dict())
                logger.info(
                    f"{'* ' + str(epoch):<10}{loss_train / len(dataloader_train):<15.4f}{accuracy_train:<15.4f}{loss_test / len(dataset_test):<15.4f}{accuracy_test:<15.4f}{optimizer.param_groups[0]['lr']:<15.6f}")
            else:
                logger.info(
                    f"{epoch:<10}{loss_train / len(dataloader_train):<15.4f}{accuracy_train:<15.4f}{loss_test / len(dataset_test):<15.4f}{accuracy_test:<15.4f}{optimizer.param_groups[0]['lr']:<15.6f}")

            writer.add_scalars('Loss', {f'{args.Dataset}_{args.model}_Test_loss': loss_test / len(dataset_test)}, epoch)
            writer.add_scalars('Loss', {f'{args.Dataset}_{args.model}_Train_Accuracy': accuracy_train,
                                        f'{args.Dataset}_{args.model}_Test_Accuracy': accuracy_test}, epoch)
    writer.close()

    if not os.path.exists(f"checkpionts/{args.Dataset}/{args.Loss_Type}/{args.model}/{split_num}"):
        os.makedirs(f"checkpionts/{args.Dataset}/{args.Loss_Type}/{args.model}/{split_num}")
    torch.save(best_state_dict,
               f"checkpionts/{args.Dataset}/{args.Loss_Type}/{args.model}/{split_num}/{get_timestamp_now()}_epoch_{epoch_num}_{args.description}_best_model_{best_acc:.4f}.pth")
    logger.info(
        f"SpliteNum_{split_num} {args.Dataset}_Test Best epoch: {best_epoch},Best Accuracy: {best_acc:.4f}, Best Precision: {best_precision:.4f}, Best Recall: {best_recall:.4f}, Best F1: {best_f1:.4f}")
    best_acc_list.append(best_acc)
    best_precision_list.append(best_precision)
    best_recall_list.append(best_recall)
    best_f1_list.append(best_f1)
    best_auc_list.append(best_auc)



def train_UNI_CLIP(args, split_num=1):
    logger.info(args)
    logger.info(f"split_num: {split_num}")
    model = UNI_CLIP_Model(args=args)
    model.to("cuda")

    preprocess2 = _preprocess2()
    preprocess3 = _preprocess3()
    dataset_train = Standard_Dataset_Patch_Resize(args=args,
                                                  txt_path=f'/public/yxz/UNI/Datasplit/{args.Dataset}/{split_num}/train.txt',
                                                  transform=preprocess3, num_classes=num_classes,
                                                  num_patch=args.num_patch)
    dataset_test = Standard_Dataset_Patch_Resize(args=args,
                                                 txt_path=f'/public/yxz/UNI/Datasplit/{args.Dataset}/{split_num}/test.txt',
                                                 transform=preprocess2, num_classes=num_classes,
                                                 num_patch=args.num_patch, test=True)
    batch_size = 18

    drop_last_train = True
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=10,
                                  drop_last=drop_last_train)
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=10, drop_last=False)
    if args.Loss_Type == "Cross_Entropy":
        criterion = nn.CrossEntropyLoss()
        criterion_test = nn.CrossEntropyLoss(reduction="sum")  # 用于测试集的loss计算,因为测试集的batch_size不为1
    model.train()

    if "UNI_CLIP" in args.model:
        writer = SummaryWriter(
            f"tensorboard/{args.Dataset}/{args.Loss_Type}/{args.model}/{split_num}/Semi_{args.semi_loss.upper()}-{get_timestamp_now()}")
    else:
        writer = SummaryWriter(
            f"tensorboard/{args.Dataset}/{args.Loss_Type}/{args.model}/{split_num}/CLIP_{get_timestamp_now()}")

    # 初始化度量计算对象
    accuracy = Accuracy(task='MULTICLASS', num_classes=num_classes)
    precision = Precision(task='MULTICLASS', num_classes=num_classes, average='macro')  # num_classes 是类别的数量
    recall = Recall(task='MULTICLASS', num_classes=num_classes, average='macro')
    f1score = F1Score(task='MULTICLASS', num_classes=num_classes, average='macro')
    auc = AUROC(task='MULTICLASS', num_classes=num_classes)

    epoch_num = args.epoch_num

    if "UNI_CLIP" in args.model:
        logger.info(
            f"{'Epoch':<10}{'Train Loss':<15}{'Train Sup Loss':<20}{'Train Semi Loss':<20}{'Train Acc':<15}{'Test Loss':<15}{'Test Acc':<15}{'learning rate':<15}")
    else:
        logger.info(
            f"{'Epoch':<10}{'Train Loss':<15}{'Train Acc':<15}{'Test Loss':<15}{'Test Acc':<15}{'learning rate':<15}")


    initial_lr2 = 5e-4
    weight_decay = 0.001
    optimizer = torch.optim.AdamW(model.parameters(), lr=initial_lr2, weight_decay=weight_decay)
    best_epoch = 0
    best_acc, best_precision, best_recall, best_f1, best_auc = 0.0, 0.0, 0.0, 0.0, 0.0
    # formal training
    y_train_pred_logits = []
    y_train_label = []
    for epoch in range(1, epoch_num + 1):
        if epoch == 1:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 1e-4
        if epoch == 10:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 5e-5
        if epoch == 20:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 1e-5
        if epoch == 40:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 5e-6

        labels_train = []
        labels_train_pred = []
        labels_train_pred_logits = []
        loss_train = 0.0
        loss_train_sup = 0.0
        loss_train_semi = 0.0

        if args.model == "CLIP_Resample":
            with open(f'{args.Dataset}/{split_num}/train.txt', 'r') as f:
                lines = f.readlines()
            labels = torch.tensor([int(line.strip().split()[1]) for line in lines])
            label_counts = torch.bincount(labels)
            label_ratio = label_counts.float() / label_counts.sum().float()
            if epoch == 1:
                class_weights = torch.tensor([1.0 / num_classes for _ in range(num_classes)])
            else:
                class_weights = torch.zeros(num_classes)
                label_index = [[] for _ in range(num_classes)]
                for index in range(len(y_train_label)):
                    label = y_train_label[index]
                    label_index[label].append(index)
                for i in range(num_classes):
                    y_train_pred_logits_i = y_train_pred_logits[label_index[i]]
                    y_train_label_i = y_train_label[label_index[i]]
                    BCE_loss = F.cross_entropy(y_train_pred_logits_i, y_train_label_i, reduction='none')
                    pt = torch.exp(-BCE_loss)  # 计算预测正确的概率
                    class_weight = (1 - pt).pow(args.resample_gamma)
                    class_weights[i] = class_weight.mean()
            sample_weights = class_weights[labels]
            sampler = WeightedRandomSampler(sample_weights, len(dataset_train), replacement=True)
            dataloader_train = DataLoader(dataset_train, batch_size=batch_size, num_workers=10, sampler=sampler)
        model.train()
        model.stage = "train"
        for i, batch in enumerate(dataloader_train):
            optimizer.zero_grad()
            images, labels = batch[0], batch[1]
            images = images.to("cuda")
            labels = labels.to("cuda")
            if args.model == "UNI_CLIP":
                y_pred_CLIP_logits, y_pred_CLIP, loss_distillation = model(images)
                loss_sup = criterion(y_pred_CLIP_logits, labels)
                loss = loss_sup + loss_distillation
            else:
                y_pred_CLIP_logits, y_pred_CLIP = model(images, labels, epoch)
                loss = criterion(y_pred_CLIP_logits, labels)
            loss.backward()
            optimizer.step()
            # 评价分类器的性能，用准确率、精确率、召回率、F1值等指标
            labels_train.extend(labels.cpu().tolist())
            labels_train_pred.extend(y_pred_CLIP.cpu().tolist())
            labels_train_pred_logits.extend(y_pred_CLIP_logits.cpu().tolist())
            loss_train += loss.item()
        y_train_pred_original = torch.tensor(labels_train_pred)

        labels_train_pred = torch.argmax(torch.tensor(labels_train_pred), dim=1)
        y_train_pred_logits = torch.tensor(labels_train_pred_logits)
        labels_train = torch.tensor(labels_train)
        if labels_train.dim() == 2:
            labels_train = torch.argmax(labels_train, dim=1)
        y_train_label = torch.tensor(labels_train)
        accuracy_train = accuracy(labels_train_pred, labels_train)
        writer.add_scalars('Loss', {f'{args.model}_Train_loss': loss_train / len(dataloader_train)}, epoch)
        if "UNI_CLIP" in args.model:
            writer.add_scalars('Loss', {f'{args.model}_Train_sup_loss': loss_train_sup / len(dataloader_train),
                                        f'{args.model}_Train_semi_loss': loss_train_semi / len(dataloader_train)},
                               epoch)
        labels_test = []
        labels_test_pred = []
        loss_test = 0.0
        loss_test_semi = 0.0
        model.eval()
        model.stage = "test"
        with torch.no_grad():
            for i,batch in enumerate(dataloader_test):
                images, labels = batch[0], batch[1]
                images = images.to("cuda")
                labels = labels.to("cuda")
                if "UNI_CLIP" in args.model:
                    y_pred_CLIP_logits, y_pred_CLIP, loss_semi = model(images)
                else:
                    y_pred_CLIP_logits, y_pred_CLIP = model(images)
                loss = criterion_test(y_pred_CLIP_logits, labels)
                labels_test.extend(labels.cpu().tolist())
                if "UNI_CLIP" in args.model:
                    loss_test_semi += loss_semi.item()
                labels_test_pred.extend(y_pred_CLIP.cpu().tolist())
                loss_test += loss.item()
            labels_test_pred_original = torch.tensor(labels_test_pred)
            labels_test_pred = torch.argmax(torch.tensor(labels_test_pred), dim=1)
            labels_test = torch.tensor(labels_test)
            accuracy_test = accuracy(labels_test_pred, labels_test)
            precision_test = precision(labels_test_pred, labels_test)
            recall_test = recall(labels_test_pred, labels_test)
            f1_test = f1score(labels_test_pred, labels_test)
            auc_test = auc(labels_test_pred_original, labels_test)
            if accuracy_test > best_acc:
                best_epoch = epoch
                best_acc = accuracy_test
                best_precision = precision_test
                best_recall = recall_test
                best_f1 = f1_test
                best_auc = auc_test
                best_state_dict = copy.deepcopy(model.state_dict())
                if "UNI_CLIP" in args.model:
                    logger.info(
                        f"{'* ' + str(epoch):<10}{loss_train / len(dataloader_train):<15.4f}{loss_train_sup / len(dataloader_train):<20.4f}{loss_train_semi / len(dataloader_train):<20.4f}{accuracy_train:<15.4f}{loss_test / len(dataset_test):<15.4f}{accuracy_test:<15.4f}{optimizer.param_groups[0]['lr']:<15.6f}")
                else:
                    logger.info(
                        f"{'* ' + str(epoch):<10}{loss_train / len(dataloader_train):<15.4f}{accuracy_train:<15.4f}{loss_test / len(dataset_test):<15.4f}{accuracy_test:<15.4f}{optimizer.param_groups[0]['lr']:<15.6f}")
            else:
                if "UNI_CLIP" in args.model:
                    logger.info(
                        f"{epoch:<10}{loss_train / len(dataloader_train):<15.4f}{loss_train_sup / len(dataloader_train):<20.4f}{loss_train_semi / len(dataloader_train):<20.4f}{accuracy_train:<15.4f}{loss_test / len(dataset_test):<15.4f}{accuracy_test:<15.4f}{optimizer.param_groups[0]['lr']:<15.7f}")
                else:
                    logger.info(
                        f"{epoch:<10}{loss_train / len(dataloader_train):<15.4f}{accuracy_train:<15.4f}{loss_test / len(dataset_test):<15.4f}{accuracy_test:<15.4f}{optimizer.param_groups[0]['lr']:<15.7f}")

            writer.add_scalars('Loss', {f'{args.model}_Test_loss': loss_test / len(dataset_test)}, epoch)
            writer.add_scalars('Loss', {f'{args.model}_Train_Accuracy': accuracy_train,
                                        f'{args.model}_Test_Accuracy': accuracy_test}, epoch)
            if "UNI_CLIP" in args.model:
                writer.add_scalars('Loss', {f'{args.model}_Test_semi_loss': loss_test_semi / len(dataset_test)}, epoch)
    if not os.path.exists(f"checkpionts/{args.Dataset}/{args.Loss_Type}/{args.model}/{split_num}"):
        os.makedirs(f"checkpionts/{args.Dataset}/{args.Loss_Type}/{args.model}/{split_num}")

    if "UNI_CLIP" in args.model:
        if args.Loss_Type == "Cross_Entropy":
            torch.save(best_state_dict,
                       f"checkpionts/{args.Dataset}/{args.Loss_Type}/{args.model}/{split_num}/{get_timestamp_now()}_{args.model}_Semi-{args.semi_loss}_{best_acc:.4f}.pth")
        elif args.Loss_Type == "Focal_Loss":
            torch.save(best_state_dict,
                       f"checkpionts/{args.Dataset}/{args.Loss_Type}/{args.model}/{split_num}/"
                       f"{get_timestamp_now()}_{args.model}_Semi-{args.semi_loss}_alpha_{args.focal_loss_alpha}_gamma{args.focal_loss_gamma}_{best_acc:.4f}.pth")
    else:
        if args.Loss_Type == "Cross_Entropy":
            torch.save(best_state_dict,
                       f"checkpionts/{args.Dataset}/{args.Loss_Type}/{args.model}/{split_num}/{get_timestamp_now()}_{args.model}_{best_acc:.4f}.pth")
        elif args.Loss_Type == "Focal_Loss":
            torch.save(best_state_dict,
                       f"checkpionts/{args.Dataset}/{args.Loss_Type}/{args.model}/{split_num}/"
                       f"{get_timestamp_now()}_{args.model}_alpha_{args.focal_loss_alpha}_gamma{args.focal_loss_gamma}_{best_acc:.4f}.pth")
    logger.info(
        f"Split {split_num}:Best Accuracy: {best_acc:.4f} \n")
    writer.close()
    best_acc_list.append(best_acc)
    best_precision_list.append(best_precision)
    best_recall_list.append(best_recall)
    best_f1_list.append(best_f1)
    best_auc_list.append(best_auc)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"Set Random Seed: {seed}")


def test_Model(args, split_num):
    args.stage = "test"
    if args.model == "UNI":
        model = UNI_Classifier()
    else:
        model = UNI_CLIP_Model()

    preprocess2 = _preprocess2()
    preprocess3 = _preprocess3()
    dataset_test = Standard_Dataset_Patch_Resize(args=args,
                                                     txt_path=f'/public/yxz/UNI/Datasplit/{args.Dataset}/{split_num}/test.txt',
                                                     transform=preprocess2, num_classes=num_classes,
                                                     num_patch=args.num_patch, test=True)
    batch_size = 18
    pth_path = f"checkpionts/{args.Dataset}/{args.Loss_Type}/{args.model}/{split_num}"
    # 列出所有的pth文件
    pth_files = os.listdir(pth_path)
    pth_files = [file for file in pth_files if file.endswith(".pth")]
    # 选择最好的模型
    pth_files.sort()
    best_model = pth_files[-1]
    # 加载模型
    best_model_path = os.path.join(pth_path, best_model)
    logger.info(best_model_path)
    model.load_state_dict(torch.load(best_model_path, map_location="cuda"), strict=False)
    raw_weight.append(model.raw_weights)
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=10, drop_last=False)
    # 初始化度量计算对象
    accuracy = Accuracy(task='MULTICLASS', num_classes=num_classes)
    precision = Precision(task='MULTICLASS', num_classes=num_classes, average='macro')  # num_classes 是类别的数量
    recall = Recall(task='MULTICLASS', num_classes=num_classes, average='macro')
    f1score = F1Score(task='MULTICLASS', num_classes=num_classes, average='macro')
    AUC = AUROC(task='MULTICLASS', num_classes=num_classes, average='macro')
    recall_individual = Recall(task='MULTICLASS', num_classes=num_classes, average='none')
    cohen_kappa = CohenKappa(task="multiclass", num_classes=num_classes)
    mcc = MatthewsCorrCoef(task="multiclass", num_classes=num_classes)
    model.to("cuda")
    model.eval()
    labels_all = []
    labels_pred = []
    with torch.no_grad():
        # 取dataloader_test的前100个batch进行测试
        for i, batch in enumerate(dataloader_test):
            images, labels = batch[0], batch[1]
            images = images.to("cuda")
            labels = labels.to("cuda")
            if "UNI_CLIP" in args.model:
                y_pred_CLIP_logits, y_pred_CLIP = model.test(images)
            else:
                y_pred_CLIP_logits, y_pred_CLIP = model(images)
            labels_all.extend(labels.cpu().tolist())
            labels_pred.extend(y_pred_CLIP.cpu().tolist())
        labels_pred_original = torch.tensor(labels_pred)
        labels_pred = torch.argmax(torch.tensor(labels_pred), dim=1)
        labels_all = torch.tensor(labels_all)
        accuracy_average = accuracy(labels_pred, labels_all)
        precision_average = precision(labels_pred, labels_all)
        recall_average = recall(labels_pred, labels_all)
        f1_average = f1score(labels_pred, labels_all)
        AUC_average = AUC(labels_pred_original, labels_all)
        test_recall_individual = recall_individual(labels_pred, labels_all)
        test_kappa = cohen_kappa(labels_pred, labels_all)
        test_mcc = mcc(labels_pred, labels_all)
        test_bacc = test_recall_individual.mean()

    print(
        f"Test Accuracy: {accuracy_average:.4f}, Precision: {precision_average:.4f}, Recall: {recall_average:.4f}, F1: {f1_average:.4f}, AUC: {AUC_average:.4f}")
    Accuracy_Average.append(float(accuracy_average*100))
    Precision_Average.append(float(precision_average*100))
    Recall_Average.append(float(recall_average*100))
    F1_Average.append(float(f1_average*100))
    AUC_Average.append(float(AUC_average*100))
    Kappa_Average.append(float(test_kappa*100))
    MCC_Average.append(float(test_mcc*100))
    BACC_Average.append(float(test_bacc*100))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="UNI", help="UNI or ImageNet21k",
                        choices=["UNI",
                                 "CLIP",
                                 "CLIP_Resample",
                                 "UNI_CLIP",
                                 ])
    # stage:train or test
    parser.add_argument("--stage", type=str, default="train", help="train or test", choices=["train", "test"])
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID')
    parser.add_argument("--semi_loss", type=str, default="kl", help="cross_entropy or mse or kl",
                        choices=["cross_entropy", "mse", "kl", "mse_softmax", "l2_norm"])
    parser.add_argument("--Dataset", type=str, default="LungHist700_3class", help="Chaoyang or ImageNet21k",
                        choices=["Chaoyang", "BreaKHis_v1_8class", "LungHist700_3class"])
    parser.add_argument("--epoch_num_classification", type=int, default=100, help="The number of epoch")
    parser.add_argument("--epoch_num", type=int, default=100, help="The number of epoch")
    parser.add_argument("--split_num_start", type=int, default=1, help="The number of epoch")
    parser.add_argument("--split_num_end", type=int, default=10, help="The number of epoch")
    # description
    parser.add_argument("--description", type=str, default="", help="The description of the experiment")
    # random seed
    parser.add_argument("--seed", type=int, default=2022, help="The random seed")
    # train_flag
    parser.add_argument("--train_flag", type=int, default=0, help="The train flag", choices=[0, 1])
    # test_flag
    parser.add_argument("--test_flag", type=int, default=0, help="The test flag", choices=[0, 1])
    # Loss_type
    parser.add_argument("--Loss_Type", type=str, default="Cross_Entropy", help="The Loss type",
                        choices=["Cross_Entropy", "Focal_Loss"])
    # num_patch:8
    parser.add_argument("--num_patch", type=int, default=8, help="The number of patch")
    # resample_gamma: 2.0
    parser.add_argument("--resample_gamma", type=float, default=2.0, help="The gamma of resample")
    # pretrained_teacher_path: None
    parser.add_argument("--pretrained_teacher_path", type=str, default=None, help="The path of the pretrained teacher model")
    args = parser.parse_args()
    logfilename = f"logs/{args.Dataset}/{args.Loss_Type}/{args.model}/{args.Dataset}_{args.model}_{args.description}_split{args.split_num_start}_{args.split_num_end}.log"
    if not os.path.exists(f"logs/{args.Dataset}/{args.Loss_Type}/{args.model}"):
        os.makedirs(f"logs/{args.Dataset}/{args.Loss_Type}/{args.model}", exist_ok=True)
    logger = set_logger(
        logfile_name=logfilename,
        log_name='my_logger')
    logger.info(args)

    seed = args.seed
    set_seed(seed)

    # args.model = "UNI_CLIP"
    torch.cuda.set_device(args.gpu_id)
    if not os.path.exists(f"logs/{args.Dataset}/{args.model}"):
        os.makedirs(f"logs/{args.Dataset}/{args.model}")

    # print(args)
    # 建立字典保存num_classes
    num_classes_dict = {"Chaoyang": 4, "BreaKHis_v1_8class": 8,  "LungHist700_3class": 3}
    Dataset_ClassesName = {"Chaoyang": ["Normal", "Serrated", "Adenocarcinoma", "Adenoma"],
                           "BreaKHis_v1_8class": ["Adenosis", "Fibroadenoma", "Phyllodes Tumor", "Tubular Adenoma",
                                                  "Ductal Carcinoma", "Lobular Carcinoma",
                                                  "Mucinous Carcinoma", "Papillary Carcinoma"],
                           "LungHist700_3class": ["Adenocarcinomas", "Squamous Cell Carcinomas", "Normal Lung Tissues"],}
    num_classes = num_classes_dict[args.Dataset]
    # 每个类别的样本数量
    num_samples_dict = {"Chaoyang": [705, 321, 840, 273],
                        "BreaKHis_v1_8class": [444, 1014, 453, 569, 3451, 626, 792, 560],
                        "LungHist700_3class": [280, 260, 151]}

    Transforms = transforms.Compose([
            transforms.Resize((224, 224)),  # 调整大小
            transforms.ToTensor(),  # 转换为张量并将像素值缩放到 [0, 1] 范围
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 归一化
        ])

    best_acc_list, best_precision_list, best_recall_list, best_f1_list, best_auc_list = [], [], [], [], []
    if args.train_flag:

        for split_num in tqdm(range(args.split_num_start, args.split_num_end + 1), desc="Split Num"):
            if args.model == "UNI":
                train_Classifier(args, split_num)
            else:
                train_UNI_CLIP(args, split_num)
        # 打印最好的结果
        logger.info(args)
        logger.info(f"Best Acc: {best_acc_list}")
        logger.info(f"Best Precision: {best_precision_list}")
        logger.info(f"Best Recall: {best_recall_list}")
        logger.info(f"Best F1: {best_f1_list}")
        logger.info(f"Average Best Acc: {np.mean(best_acc_list):.4f}")
        logger.info(f"Average Best Precision: {np.mean(best_precision_list):.4f}")
        logger.info(f"Average Best Recall: {np.mean(best_recall_list):.4f}")
        logger.info(f"Average Best F1: {np.mean(best_f1_list):.4f}")
        logger.info(f"Average Best AUC: {np.mean(best_auc_list):.4f}")

    if args.test_flag:
        Accuracy_Average, Precision_Average, Recall_Average, F1_Average, AUC_Average = [], [], [], [], []
        Kappa_Average, MCC_Average, BACC_Average = [], [], []
        Accuracy_Per_Class, Precision_Per_Class, Recall_Per_Class, F1_Per_Class, AUC_Per_Class = [[] for _ in
                                                                                                  range(num_classes)], [
            [] for _ in range(num_classes)], [[] for _ in range(num_classes)], [[] for _ in range(num_classes)], [[] for
                                                                                                                  _ in
                                                                                                                  range(
                                                                                                                      num_classes)]
        logger.info(f"Test Flag: {args.test_flag}")
        if "Text-three-mode-Adaptive" in args.model:
            raw_weight = []
        for split_num in tqdm(range(args.split_num_start, args.split_num_end + 1), desc="Split Num"):
            test_Model(args, split_num)
        if "Text-three-mode-Adaptive" in args.model:
            for weight in raw_weight:
                logger.info(f"Weight: {weight}")

        # 打印最好的结果
        logger.info(args)
        # 打印数据集
        logger.info(f"Dataset: {args.Dataset}")
        # 打印Accuracy_Average、Precision_Average、Recall_Average、F1_Average、AUC_Average的所有元素
        logger.info(f"Accuracy_Average: {Accuracy_Average}%")
        logger.info(f"Precision_Average: {Precision_Average}%")
        logger.info(f"Recall_Average: {Recall_Average}%")
        logger.info(f"F1_Average: {F1_Average}%")
        logger.info(f"AUC_Average: {AUC_Average}%")

        # 打印平均值
        logger.info(f"Accuracy_Average: {np.average(Accuracy_Average):.2f}%")
        logger.info(f"Precision_Average: {np.average(Precision_Average):.2f}%")
        logger.info(f"Recall_Average: {np.average(Recall_Average):.2f}%")
        logger.info(f"F1_Average: {np.average(F1_Average):.2f}%")
        logger.info(f"AUC_Average: {np.average(AUC_Average):.2f}%")
        logger.info(f"Kappa_Average: {np.average(Kappa_Average):.2f}%")
        logger.info(f"MCC_Average: {np.average(MCC_Average):.2f}%")
        logger.info(f"BACC_Average: {np.average(BACC_Average):.2f}%")
        logger.info('------------------------------------------------')
        # 打印无偏标准差
        logger.info(f"Accuracy_Std: {np.std(Accuracy_Average, ddof=1):.2f}")
        logger.info(f"Precision_Std: {np.std(Precision_Average, ddof=1):.2f}")
        logger.info(f"Recall_Std: {np.std(Recall_Average, ddof=1):.2f}")
        logger.info(f"F1_Std: {np.std(F1_Average, ddof=1):.2f}")
        logger.info(f"AUC_Std: {np.std(AUC_Average, ddof=1):.2f}")
        logger.info(f"Kappa_Std: {np.std(Kappa_Average, ddof=1):.2f}")
        logger.info(f"MCC_Std: {np.std(MCC_Average, ddof=1):.2f}")
        logger.info(f"BACC_Std: {np.std(BACC_Average, ddof=1):.2f}")
        logger.info('------------------------------------------------')