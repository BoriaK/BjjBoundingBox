import os
import shutil
import time
import torch
from torch.utils.data import DataLoader, Subset
# from Common_Files.Datasets_XML import ObjectDetectDataset
from Common_Files.Datasets_from_CSV import ObjectDetectDataset
import argparse
from torchvision.transforms import v2
from Common_Files.Models import Detector
from Common_Files.checkpoint_config import ACTIVE_CHECKPOINT
import Common_Files.train_helpers as helpers
import math
import sys
import cv2
import numpy as np
import Common_Files.utils as utils
from torchvision.ops import nms, box_iou


parser = argparse.ArgumentParser()
parser.add_argument('--resume_epoch', type=int, default=None,
                    help='resume from <root_chkps>/bjj_<N>.pth instead of the inference checkpoint')
parser.add_argument('--resume_checkpoint', type=str, default=ACTIVE_CHECKPOINT,
                    help='checkpoint file to resume from (default: the checkpoint used by inference)')
parser.add_argument('--from_scratch', action='store_true',
                    help='ignore any checkpoint and start training from the pretrained COCO weights')
parser.add_argument('--n_epochs', type=int, default=20, help='number of additional epochs to train')
parser.add_argument('--batch_size', type=int, default=4, help='size of the batches')
# parser.add_argument('--data', type=str, default=r'./Dataset/Sub_Set_v01/train/bed', help='root directory of the '
#                                                                                          'dataset')
parser.add_argument('--data', type=str, default=r'./Dataset/Sub_Set_v01/train/bjj/labeled',
                    help='root directory of the dataset')
parser.add_argument('--no_save_results', dest='save_results', action='store_false',
                    help='do not save result images after each training epoch')
parser.add_argument('--results_dir', type=str, default='./Results_for_Detector',
                    help='folder for the per-epoch result images')
parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'cpu'],
                    help="computation device ('auto' uses the GPU when available)")
parser.add_argument('--root_chkps', type=str, default='./checkpoints', help='checkpoint folder')
parser.add_argument('--n_classes', type=int, default=2, help='number of total classes')
parser.add_argument('--seed', type=int, default=0,
                    help='seed of the fixed train/validation split (the same seed gives the same split every run)')
parser.add_argument('--val_fraction', type=float, default=0.1, help='fraction of the images used for validation')
parser.add_argument('--best_metric', type=str, default='val_loss', choices=['val_loss', 'mean_iou'],
                    help='validation metric used to select the best checkpoint')
parser.add_argument('--patience', type=int, default=0,
                    help='stop early after this many epochs without validation improvement (0 = disabled)')
parser.add_argument('--num_workers', type=int, default=4, help='data loading worker processes')
parser.add_argument('--no_amp', action='store_true', help='disable mixed precision (it is used on GPU by default)')
parser.add_argument('--lr', type=float, default=None,
                    help='starting learning rate of the run, decayed with cosine annealing '
                         '(default: 2e-4 when resuming, 5e-3 with --from_scratch)')
parser.add_argument('--no_augment', action='store_true', help='disable training augmentations')
parser.add_argument('--log_file', type=str, default='./training_log.csv',
                    help='CSV file the per-epoch metrics are appended to')
args = parser.parse_args()

'''train'''
# use our dataset and defined transformations
transforms_train = None if args.no_augment else v2.Compose(
    [v2.RandomHorizontalFlip(p=0.5),
     v2.RandomVerticalFlip(p=0.5),
     v2.RandomRotation(degrees=10),
     v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
     v2.RandomInvert(p=0.2)])
# transforms_train = Sequence([RandomHorizontalFlip(0.5),RandomRotate(10), RandomShear(0.2), RandomHSV(30,30,30), RandomColor(0.5), RandomTranslate(0.5)])
# transforms_train = None

# the validation images are never augmented
transforms_test = None


# Debug Path
# root = r'C:\Users\bkupcha\OneDrive - Intel Corporation\Documents\PythonProjects\Furnecher_Project_Pytorch\Dataset' \
#        r'\Sub_Set_v01\train\bed'
# print(os.path.abspath(args.data))


def evaluate_and_save(model, data_loader_test, epoch):
    """Run detection on the validation images; returns the mean IoU of the top box with the ground truth."""
    conf = 0.9
    ious = []
    model.eval()
    with torch.no_grad():
        for i, (images, targets) in enumerate(data_loader_test):
            images = [image.to(args.device) for image in images]
            predictions = model(images, [], False)
            boxes = predictions[0]['boxes'].cpu()
            scores = predictions[0]['scores'].cpu()
            boxes_keep = nms(boxes, scores, 0.5)
            boxes = boxes[boxes_keep].numpy()
            scores = scores[boxes_keep].numpy()
            if len(boxes) > 0:
                top = int(scores.argmax())
                ious.append(box_iou(torch.from_numpy(boxes[top:top + 1]), targets[0]['boxes'])[0, 0].item())
            else:
                ious.append(0.)
            if not args.save_results:
                continue
            im_ = images[0].cpu().numpy()
            im_ = np.moveaxis(im_, 0, -1)[:, :, ::-1]
            im_ = np.ascontiguousarray(im_)  # this is due to a memorry related assert in the code
            im_ = (im_ * 255).astype(np.uint8)
            for j, b in enumerate(boxes):
                if scores[j] > conf:
                    im_ = cv2.rectangle(im_, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 255, 0), 5)
                    im_ = cv2.putText(im_, str(np.round(scores[j] * 100) / 100), (int(b[0]), int(b[3])),
                                      cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 2, (255, 0, 0), 10, cv2.LINE_AA)
            os.makedirs(args.results_dir, exist_ok=True)
            cv2.imwrite(os.path.join(args.results_dir, 'epoch' + str(epoch) + '_Img' + str(i) + '.jpg'), im_)
    return float(np.mean(ious)) if ious else 0.


def trainDetector():
    args.device = helpers.resolve_device(args.device)
    print(args)
    print(args.device)
    if not 0 < args.val_fraction < 1:
        exit('--val_fraction must be between 0 and 1')
    os.makedirs(args.root_chkps, exist_ok=True)
    use_amp = args.device.type == 'cuda' and not args.no_amp
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    dataset = ObjectDetectDataset(args, transforms_train, 'train')
    dataset_test = ObjectDetectDataset(args, transforms_test, 'test')

    # split the dataset in train and validation set (fixed by --seed, stable across runs)
    train_indices, val_indices = helpers.split_indices(dataset.imgs, args.val_fraction, args.seed)
    print('train images: {}, validation images: {}'.format(len(train_indices), len(val_indices)))
    dataset = Subset(dataset, train_indices)
    dataset_test = Subset(dataset_test, val_indices)

    # define training and validation data loaders
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                             persistent_workers=args.num_workers > 0, pin_memory=args.device.type == 'cuda',
                             collate_fn=utils.collate_fn, drop_last=True)
    if len(data_loader) == 0:
        exit('not enough training images for batch size {}'.format(args.batch_size))

    data_loader_test = DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=0, collate_fn=utils.collate_fn)

    # get the model using our helper function
    # move model to the right device
    model = Detector(n_classes=args.n_classes, device=args.device)

    # construct an optimizer
    base_lr = args.lr if args.lr is not None else (0.005 if args.from_scratch else 2e-4)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=base_lr,
                                momentum=0.9, weight_decay=0.0005)

    if args.from_scratch:
        resume_path = None
    elif args.resume_epoch is not None:
        resume_path = helpers.checkpoint_path(args.root_chkps, args.resume_epoch)
    else:
        resume_path = args.resume_checkpoint

    if resume_path is not None:
        if not os.path.isfile(resume_path):
            exit('no pth file: ' + resume_path)
        checkpoint = helpers.load_checkpoint(resume_path, args.device)
        print('Resuming from checkpoint: {} (epoch {})'.format(resume_path, checkpoint['epoch']))
        model.load_state_dict(checkpoint['model_state'])
        # keeps the momentum buffers; the saved learning rate and schedule are replaced below
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        for group in optimizer.param_groups:
            group['lr'] = base_lr
            group['initial_lr'] = base_lr
        checkpoint_epoch = checkpoint['epoch']
        start_epoch = checkpoint_epoch + 1
    else:
        print('Training from scratch')
        start_epoch = 0
    end_epoch = start_epoch + args.n_epochs

    # a fresh schedule for every run: cosine decay from the base learning rate over this run's epochs
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.n_epochs, 1),
                                                              eta_min=base_lr * 0.01)
    print('learning rate: {} -> {} (cosine over {} epochs)'.format(base_lr, base_lr * 0.01, args.n_epochs))

    existing = [helpers.checkpoint_path(args.root_chkps, e) for e in range(start_epoch, end_epoch)
                if os.path.exists(helpers.checkpoint_path(args.root_chkps, e))]
    if existing:
        print('Warning: these checkpoints already exist and may be overwritten: ' + ', '.join(existing))

    latest_path = helpers.checkpoint_path(args.root_chkps, 'latest')
    best_path = None
    best_score = float('inf')
    epochs_without_improvement = 0
    last_epoch = None

    '''train loop'''
    for epoch in range(start_epoch, end_epoch):
        epoch_start = time.time()
        model.train()
        loss_av = 0.

        for batch_idx, (images, targets) in enumerate(data_loader):
            images = list(image.to(args.device, non_blocking=True) for image in images)
            targets = [{k: v.to(args.device) for k, v in t.items()} for t in targets]
            with torch.autocast(device_type=args.device.type, enabled=use_amp):
                loss_dict = model(images, targets, True)
                losses = sum(loss for loss in loss_dict.values())

            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = utils.reduce_dict(loss_dict)
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())

            loss_value = losses_reduced.item()

            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                print(loss_dict_reduced)
                sys.exit(1)

            optimizer.zero_grad()
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()

            loss_av += losses.item()

            if batch_idx % 5 == 0:
                print('epoch:{}/{}, batch:{}/{}, lr={:4f}, '
                      'loss-classifier:{:2f}, '
                      'loss_box_reg:{:2f}, '
                      'loss_objectness={:2f}, '
                      'loss_rpn_box_reg={:2f}'.
                      format(epoch, end_epoch,
                             batch_idx, len(data_loader),
                             optimizer.param_groups[0]["lr"],
                             loss_dict['loss_classifier'].item(),
                             loss_dict['loss_box_reg'].item(),
                             loss_dict['loss_objectness'].item(),
                             loss_dict['loss_rpn_box_reg'].item()))

        loss_av /= len(data_loader)
        epoch_lr = optimizer.param_groups[0]["lr"]
        if lr_scheduler is not None:
            lr_scheduler.step()

        val_loss = helpers.compute_val_loss(model, data_loader_test, args.device, args.seed)
        mean_iou = evaluate_and_save(model, data_loader_test, epoch)
        score = val_loss if args.best_metric == 'val_loss' else -mean_iou
        improved = score < best_score

        checkpoint = {'epoch': epoch,
                      'model_state': model.state_dict(),
                      'optimizer_state': optimizer.state_dict(),
                      'lr_scheduler_state': lr_scheduler.state_dict(),
                      'val_loss': val_loss,
                      'mean_iou': mean_iou}
        # always keep the most recent state so an interrupted run can be resumed from it
        helpers.save_checkpoint(latest_path, checkpoint)
        last_epoch = epoch
        if improved:
            best_score = score
            epochs_without_improvement = 0
            new_best_path = helpers.checkpoint_path(args.root_chkps, epoch)
            shutil.copyfile(latest_path, new_best_path)
            # remove the previous best checkpoint of this run (never the one we resumed from)
            if best_path is not None and os.path.abspath(best_path) != os.path.abspath(resume_path or ''):
                helpers.remove_if_exists(best_path)
            best_path = new_best_path
        else:
            epochs_without_improvement += 1

        print('epoch:{}/{} done in {:.0f}s, train_loss={:.4f}, val_loss={:.4f}, val_mean_iou={:.4f}{}'.format(
            epoch, end_epoch, time.time() - epoch_start, loss_av, val_loss, mean_iou,
            ' (best)' if improved else ''))
        helpers.append_log(args.log_file, {'epoch': epoch, 'lr': epoch_lr, 'train_loss': loss_av,
                                           'val_loss': val_loss, 'val_mean_iou': mean_iou,
                                           'seconds': round(time.time() - epoch_start, 1), 'best': int(improved)})

        if args.patience > 0 and epochs_without_improvement >= args.patience:
            print('No validation improvement for {} epochs, stopping early'.format(args.patience))
            break

    # also keep the final epoch's checkpoint
    if last_epoch is not None:
        last_path = helpers.checkpoint_path(args.root_chkps, last_epoch)
        if last_path != best_path:
            shutil.copyfile(latest_path, last_path)
        print('Best checkpoint ({}): {}'.format(args.best_metric, best_path))
        print('Last checkpoint: ' + last_path)
        print('To use the best checkpoint for inference, set ACTIVE_CHECKPOINT in '
              'Common_Files/checkpoint_config.py to it.')


if __name__ == "__main__":
    trainDetector()
