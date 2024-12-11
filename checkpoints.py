import os
import torch

def save_checkpoint(model, optimizer, epoch, loss, train_loss_list, validation_loss_list, filename='checkpoint.pth'):
    """Save model, optimizer state, and loss histories to a file."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        #'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'train_loss_list': train_loss_list,
        'validation_loss_list': validation_loss_list
    }
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved at {filename}")

def load_checkpoint(filename, model, optimizer):
    """Load model, optimizer state, and loss histories from a file."""
    if os.path.isfile(filename):
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint['model_state_dict'])
        #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        loss = checkpoint['loss']
        train_loss_list = checkpoint.get('train_loss_list', [])
        validation_loss_list = checkpoint.get('validation_loss_list', [])
        print(f"Checkpoint loaded from {filename}")
        return start_epoch, loss, train_loss_list, validation_loss_list
    else:
        print(f"No checkpoint found at {filename}")
        return 0, None, [], []