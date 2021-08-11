import argparse
import numpy as np
from PIL import Image
import cv2
from utils.utils import visualize
import torch
import torchvision.transforms as T
from torchvision import transforms
import segmentation_models_pytorch as smp

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--video', type=str, help='Full path of an input video')
    args = parser.parse_args()

    # Set-up CUDA
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

    # Process the input video
    in_video = cv2.VideoCapture(args.video)
    width = int(in_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(in_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    seg_video = cv2.VideoWriter(args.video.replace('.', '_out.'), fourcc, 24, (width, height))

    while True:
        # Read a frame from the video
        result, frame = in_video.read()
        if not result:
            break

        # Convert the frame to a PIL image
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)

        # Apply transforms and convert the image to a Pytorch tensor
        _frame = transforms.Resize((640, 640), interpolation=Image.NEAREST)(frame)
        _frame = T.ToTensor()(_frame).unsqueeze(dim=0).to(device)

        # Perform a forward pass
        logits = model(_frame)

        # Resize the logits back to the original resolution for a better visualization
        logits = torch.nn.Upsample(size=(height, width),
                                   mode='bilinear',
                                   align_corners=True)(logits)

        # Produce a segmentation map from the logits
        logits = logits.squeeze(0)
        logits = logits.cpu().detach().numpy()
        seg_map = np.argmax(logits, axis=0)

        # Visualize the segmentation map
        overlaid_img = visualize(seg_map, np.asarray(frame))

        # Save the output frame
        seg_video.write(cv2.cvtColor(overlaid_img, cv2.COLOR_RGB2BGR))

        # Early break if ESC is pressed
        if cv2.waitKey(1) & 0xff == 27:
            seg_video.release()
            in_video.release()
            cv2.destroyAllWindows()
            break

    # Close input and output video files
    seg_video.release()
    in_video.release()
    cv2.destroyAllWindows()
