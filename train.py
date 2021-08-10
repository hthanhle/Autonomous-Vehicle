import numpy as np
import torch
from PIL import Image
import segmentation_models_pytorch as smp  # https://github.com/qubvel/segmentation_models.pytorch
from utils.load_dataset import MapillaryVistasDataset
from utils.utils import iou
from torch.utils.data import DataLoader
from tqdm import tqdm


def train_model():
    model.train()
    train_loss = 0
    mIoU = 0

    for iteration, (img, gt) in enumerate(tqdm(train_loader)):
        img = img.to(device, dtype=torch.float)
        gt = gt.to(device, dtype=torch.long)

        # Perform a forward pass
        logits = model(img)

        # Compute the batch loss
        loss = loss_fn(logits, gt)

        # Compute gradients and update the weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accumulate the batch loss
        train_loss += loss.item()

        # Accumulate the metric for every iteration
        seg_maps = logits.cpu().detach().numpy().argmax(axis=1)
        gt = gt.cpu().detach().numpy()
        mIoU += iou(seg_maps, gt)

    return train_loss / len(train_loader), mIoU / len(train_loader)


def validate_model():
    model.eval()
    valid_loss = 0
    mIoU = 0

    with torch.no_grad():
        for i, (img, gt) in enumerate(val_loader):
            img, gt = img.to(device, dtype=torch.float), gt.to(device, dtype=torch.long)
            logits = model(img)

            loss = loss_fn(logits, gt)
            valid_loss += loss.item()

            seg_maps = logits.cpu().detach().numpy().argmax(axis=1)
            gt = gt.cpu().detach().numpy()
            mIoU += iou(seg_maps, gt)

    return valid_loss / len(val_loader), mIoU / len(val_loader)


if __name__ == '__main__':

    # Setup CUDA
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Create a PSPNet model
    encoder_name = 'resnet34'
    num_classes = 3  # background vs. road vs. marker
    model = smp.PSPNet(encoder_name=encoder_name,
                       classes=num_classes,
                       activation=None,
                       encoder_weights='imagenet')
    model = model.to(device)

    # Load the dataset
    train_dataset = MapillaryVistasDataset(root_dir='./data/')
    val_dataset = MapillaryVistasDataset(root_dir='./data/')
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=True, num_workers=4)

    # Define the loss function and optimizer
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    # Start training the model
    num_epochs = 100
    max_perf = 0

    for epoch in range(num_epochs):
        _, train_perf = train_model()
        _, val_perf = validate_model()
        print('Epoch: {}\tTraining acc: {:.4f}\tValid acc: {:.4f}'.format(epoch, train_perf, val_perf))

        # Save the model if the validation accuracy is improved
        if val_perf > max_perf:
            print('Valid mIoU increased ({:.4f} --> {:.4f}). Model saved'.format(max_perf, val_perf))
            torch.save(model.state_dict(), './checkpoints/PSPNet_epoch_'
                       + str(epoch) + '_acc_{0:.4f}'.format(val_perf) + '.pt')
            max_perf = val_perf
