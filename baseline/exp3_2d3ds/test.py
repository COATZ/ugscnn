import math
import argparse
import sys
sys.path.append("../../meshcnn")
import numpy as np
import pickle, gzip
import os
import shutil

from loader import SemSegLoader
from models import ResNetDUCHDC, FCN8s, UNet, UNet_sphe

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tabulate import tabulate

classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
class_names = ["unknown", "beam", "board", "bookcase", "ceiling", "chair", "clutter", "column", 
               "door", "floor", "sofa", "table", "wall", "window", "invalid"]
drop = [0, 14]
keep = np.setdiff1d(classes, drop)
label_ratio = [0.04233976974675504, 0.014504436907968913, 0.017173225930738712, 
               0.048004778186652164, 0.17384037404789865, 0.028626771620973622, 
               0.087541966989014, 0.019508096683310605, 0.08321331842901526, 
               0.17002664771895903, 0.002515611224467519, 0.020731298851232174, 
               0.2625963729249342, 0.016994731594287146, 0.012382599143792165]
# label_weight = 1/np.array(label_ratio)/np.sum((1/np.array(label_ratio))[keep])
label_weight = 1 / np.log(1.02 + np.array(label_ratio))
label_weight[drop] = 0
label_weight = label_weight.astype(np.float32)

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    import itertools
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center", fontsize=14,
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label', fontsize=14)
    plt.xlabel('Predicted label', fontsize=14)

def iou_score(pred_cls, true_cls, nclass=15, drop=drop):
    """
    compute the intersection-over-union score
    both inputs should be categorical (as opposed to one-hot)
    """
    intersect_ = []
    union_ = []
    #pred_cls = float(pred_cls)
    #true_cls = float(true_cls)
    for i in range(nclass):
        if i not in drop:
            #print((1*(pred_cls == i) + 1*(true_cls == i)).eq(2).sum().item())
            intersect = (1*(pred_cls == i) + 1*(true_cls == i)).eq(2).sum().item()
            union = (1*(pred_cls == i) + 1*(true_cls == i)).ge(1).sum().item()
            intersect_.append(intersect)
            union_.append(union)
    return np.array(intersect_), np.array(union_)

def accuracy(pred_cls, true_cls, nclass=15, drop=drop):
    positive = torch.histc(true_cls.cpu().float(), bins=nclass, min=0, max=nclass, out=None)
    per_cls_counts = []
    tpos = []
    for i in range(nclass):
        if i not in drop:
            true_positive = (1*(pred_cls == i) + 1*(true_cls == i)).eq(2).sum().item()
            tpos.append(true_positive)
            per_cls_counts.append(positive[i])
    return np.array(tpos), np.array(per_cls_counts)

def test(args, model, test_loader, epoch, device):
    w = torch.tensor(label_weight).to(device)
    model.eval()
    test_loss = 0
    ints_ = np.zeros(len(classes)-len(drop))
    unis_ = np.zeros(len(classes)-len(drop))
    per_cls_counts = np.zeros(len(classes)-len(drop))
    accs = np.zeros(len(classes)-len(drop))
    count = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            output = model(data)
            n_data = data.size()[0]

            test_loss += F.cross_entropy(output, target, weight=w).item() # sum up batch loss
            pred = output.max(dim=1, keepdim=False)[1] # get the index of the max log-probability
            int_, uni_ = iou_score(pred, target)
            tpos, pcc = accuracy(pred, target)
            ints_ += int_
            unis_ += uni_
            accs += tpos
            per_cls_counts += pcc
            count += n_data
    ious = ints_ / unis_
    accs /= per_cls_counts
    test_loss /= count
    print(per_cls_counts)
    print('[Epoch {} {} stats]: MIoU: {:.4f}; Mean Accuracy: {:.4f}; Avg loss: {:.4f}'.format(
        epoch, test_loader.dataset.partition, np.mean(ious), np.mean(accs), test_loss))
    # tabulate mean iou 
    print(tabulate(dict(zip(class_names[1:-1], [[iou] for iou in ious])), headers="keys"))
    return np.mean(np.mean(ious))

    # # plot and save confusion matrix
    # from sklearn.metrics import confusion_matrix
    # cnf_matrix = confusion_matrix(true_.ravel(), pred_.ravel())
    # # Plot normalized confusion matrix
    # plt.figure()
    # plot_confusion_matrix(cnf_matrix, classes=["Background", "TC", "AR"], normalize=True, title='Normalized confusion matrix')
    # plt.savefig("confusion_matrix.png", dpi=200)



def export(args, model, test_loader):
    model.eval()
    data_ = []
    target_ = []
    pred_ = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            output = model(data)
            n_data = data.size()[0]
            pred = output.max(dim=1, keepdim=False)[1] # get the index of the max log-probability
            data_.append(data.cpu().numpy())
            target_.append(target.cpu().numpy())
            pred_.append(pred.cpu().numpy())

    data_ = np.concatenate(data_, 0)
    target_ = np.concatenate(target_, 0)
    pred_ = np.concatenate(pred_, 0)
    print("Saving export file...")
    np.savez(args.export_file, data=data_, labels=target_, predict=pred_)
    print("Success!")
    
def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Segmentation Example')
    parser.add_argument('--test-batch-size', type=int, default=16, metavar='N',
                        help='input batch size for testing (default: 64)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--mesh_folder', type=str, default="../../mesh_files",
                        help='path to mesh folder (default: ../../mesh_files)')
    parser.add_argument('--ckpt', type=str, default="log/log_f32_cv1_l5_lw/checkpoint_latest.pth.tar_UNet_best.pth.tar")
    parser.add_argument('--data_folder', type=str, default="data_small",
                        help='path to data folder (default: processed_data)')
    parser.add_argument('--feat', type=int, default=32, help='filter dimensions')
    parser.add_argument('--export_file', type=str, default='samples.npz', help='file name for exporting samples')
    parser.add_argument('--in_ch', type=str, default="rgbd", choices=["rgb", "rgbd"], help="input channels")
    parser.add_argument('--fold', type=int, choices=[1, 2, 3], default=1, help="choice among 3 fold for cross-validation")
    parser.add_argument('--model', type=str, choices=["ResNetDUCHDC", "FCN8s", "UNet", "UNet_sphe"], required=True, help="model of choice")


    args = parser.parse_args()
    print("%s", repr(args))
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")


    torch.manual_seed(args.seed)

    if args.model == "FCN8s":
        model = FCN8s(len(classes), pretrained=False, feat=args.feat, in_ch=len(args.in_ch))
    elif args.model == "UNet":
        model = UNet(len(classes), len(args.in_ch), feat=args.feat)
    elif args.model == "UNet_sphe":
        model = UNet_sphe(len(classes), len(args.in_ch), feat=args.feat)
    model = nn.DataParallel(model)
    model.to(device)

    # load checkpoint
    assert(os.path.isfile(args.ckpt))
    print("=> loading checkpoint '{}'".format(args.ckpt))
    resume_dict = torch.load(args.ckpt)
    start_ep = resume_dict['epoch']
    best_miou = resume_dict['best_miou']

    def load_my_state_dict(self, state_dict, exclude='none'):
        from torch.nn.parameter import Parameter
 
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                continue
            if exclude in name:
                continue
            if isinstance(param, Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            own_state[name].copy_(param)

    load_my_state_dict(model, resume_dict['state_dict'])  
    print("=> loaded checkpoint '{}' (epoch {} loss {:.03f}) "
          .format(args.ckpt, resume_dict['epoch'], best_miou))
    testset = SemSegLoader(args.data_folder, "test", fold=args.fold, in_ch=len(args.in_ch))
    test_loader = DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, drop_last=False)
    print(model)

    if args.export_file and False:
        export(args, model, test_loader)
        with open(args.model+"_v1_files.txt", "w") as f:
            f.writelines([l+"\n" for l in testset.rgb_list])
    epoch = 0
    test(args, model, test_loader, epoch, device)
        
if __name__ == "__main__":
    main()
