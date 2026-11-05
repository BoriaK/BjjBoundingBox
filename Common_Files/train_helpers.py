import csv
import hashlib
import os

import torch

CHECKPOINT_PREFIX = 'bjj_'


def checkpoint_path(root, name):
    return os.path.join(root, CHECKPOINT_PREFIX + str(name) + '.pth')


def resolve_device(name):
    cuda_available = torch.cuda.is_available()
    if name == 'cuda' and not cuda_available:
        raise SystemExit('--device cuda was requested but CUDA is not available')
    if name == 'auto':
        name = 'cuda' if cuda_available else 'cpu'
    return torch.device(name)


def split_indices(names, val_fraction, seed):
    """Deterministic split keyed on the image name and the seed.

    Unlike a shuffled split, an image keeps its side of the split across runs,
    even when new images are added to the dataset.
    """
    def bucket(name):
        digest = hashlib.md5(f'{seed}:{name}'.encode(), usedforsecurity=False).digest()
        return int.from_bytes(digest[:8], 'big') / 2 ** 64

    buckets = [bucket(n) for n in names]
    val = [i for i, b in enumerate(buckets) if b < val_fraction]
    if not val and names and val_fraction > 0:
        val = [min(range(len(names)), key=buckets.__getitem__)]
    val_set = set(val)
    train = [i for i in range(len(names)) if i not in val_set]
    return train, val


def save_checkpoint(path, payload):
    # write to a temp file first so an interrupted save never corrupts an existing checkpoint
    tmp_path = path + '.tmp'
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)


def load_checkpoint(path, device):
    return torch.load(path, map_location=device, weights_only=True)


def remove_if_exists(path):
    try:
        os.remove(path)
    except FileNotFoundError:
        pass


def compute_val_loss(model, data_loader, device, seed):
    """Average total detection loss over a loader, without updating the model.

    torchvision detection models only return losses in train mode; the backbone
    uses frozen batch-norm, so this does not change any weights or statistics.
    """
    was_training = model.training
    model.train()
    total, count = 0., 0
    try:
        with torch.no_grad(), torch.random.fork_rng(devices=[device] if device.type == 'cuda' else []):
            torch.manual_seed(seed)  # the RPN/ROI sampling is random; fix it so the metric is comparable
            for images, targets in data_loader:
                images = [image.to(device) for image in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                loss_dict = model(images, targets, True)
                total += sum(loss.item() for loss in loss_dict.values()) * len(images)
                count += len(images)
    finally:
        model.train(was_training)
    return total / max(count, 1)


def append_log(path, row):
    is_new = not os.path.isfile(path)
    with open(path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if is_new:
            writer.writeheader()
        writer.writerow(row)
