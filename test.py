import argparse
import numpy as np
from PIL import Image
from utils.visualize import visualize
import cv2
import torch
import torchvision.transforms as T
from torchvision import transforms
import segmentation_models_pytorch as smp

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--image', type=str, help='Full path of a test image')
    args = parser.parse_args()

    # Set-up CUDA
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Load the trained model
    encoder_name = 'resnet34'
    num_classes = 3  # background vs. road vs. marker
    weight_path = './checkpoints/PSPNet_epoch_111_acc_0.9574.pt'

    # Create a new segmentation model
    model = smp.PSPNet(encoder_name=encoder_name,
                       classes=num_classes,
                       activation=None,
                       encoder_weights='imagenet')
    model = model.to(device)

    # Load the pretrained weight to the model
    model.load_state_dict(torch.load(weight_path, device))
    model.eval()
    print('Loading the trained model done')

    # Open an input image
    input_img = Image.open(args.image)
    height = input_img.size[1]
    width = input_img.size[0]

    # Apply transforms and convert the image to a Pytorch tensor
    img = transforms.Resize((320, 320), interpolation=Image.NEAREST)(input_img)
    img = T.ToTensor()(img).unsqueeze(dim=0).to(device)

    # Perform a forward pass
    logits = model(img)

    # Resize the logits back to the original resolution for a better visualization
    logits = torch.nn.Upsample(size=(height, width),
                               mode='bilinear',
                               align_corners=True)(logits)

    # Produce a segmentation map from the logits
    logits = logits.squeeze(0)
    logits = logits.cpu().detach().numpy()
    seg_map = np.argmax(logits, axis=0)

    # Visualize the segmentation map
    overlaid_img = visualize(seg_map, np.asarray(input_img))

    # nCombine the input image with the overlaid image
    combined_img = np.concatenate((np.asarray(input_img), overlaid_img),
                                  axis=1)

    # Save the visual result
    cv2.imwrite(args.image.replace('.', '_out.'), combined_img)
