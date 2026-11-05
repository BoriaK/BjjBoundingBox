import os

# Single source of truth for the checkpoint used by inference, and the default starting
# point of the next training run (see run_Train_Detector.py).
#
# By default it is the most recent training checkpoint (bjj_latest.pth, written after every
# training epoch). Until a training run has produced one, BASELINE_CHECKPOINT is used.
# To use a specific checkpoint instead, set PINNED_CHECKPOINT, e.g. './checkpoints/bjj_52.pth'.
LATEST_CHECKPOINT = './checkpoints/bjj_latest.pth'
BASELINE_CHECKPOINT = './checkpoints/bjj_39.pth'
PINNED_CHECKPOINT = None


def resolve_active_checkpoint():
    if PINNED_CHECKPOINT:
        return PINNED_CHECKPOINT
    if os.path.isfile(LATEST_CHECKPOINT):
        return LATEST_CHECKPOINT
    return BASELINE_CHECKPOINT


ACTIVE_CHECKPOINT = resolve_active_checkpoint()


if __name__ == '__main__':
    # python -m Common_Files.checkpoint_config  -> shows which checkpoint (and epoch) is in use
    import torch
    checkpoint = torch.load(ACTIVE_CHECKPOINT, map_location='cpu', weights_only=True)
    print('Active checkpoint: {}'.format(ACTIVE_CHECKPOINT))
    print('Epoch: {} (the next training run starts at epoch {})'.format(checkpoint['epoch'], checkpoint['epoch'] + 1))
    print('Validation loss: {}, mean IoU: {}'.format(checkpoint.get('val_loss'), checkpoint.get('mean_iou')))
