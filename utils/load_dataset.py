import numpy as np
import torch
from PIL import Image
import collections
import torchvision.transforms as T
from torch.utils.data import Dataset


class MapillaryVistasDataset(Dataset):
    def __init__(self, root_dir, subset='training', gt_type='labels_3', img_size=(640, 640)):
        """
        Parameters
        ----------
        root_dir: Directory containing train, test and val folders
        subset: Subset which we are working with
        gt_type: labels_1: Road vs. Background
                 labels_2: Lane marker vs. Background
                 labels_3: Road vs. Lane marker vs. Background.
        img_size: image size
        """
        super(MapillaryVistasDataset, self).__init__()
        self.filenames = collections.defaultdict(list)
        self.img_size = img_size
        self.resize_raw = T.Resize(img_size, interpolation=Image.BILINEAR)
        self.resize_gt = T.Resize(img_size, interpolation=Image.NEAREST)
        self.to_tensor = T.ToTensor()
        self.train_test_val = subset
        self.gt_type = gt_type
        self.data_path = root_dir + "/" + subset
        name_filepath = "{}/{}/{}.txt".format(root_dir, subset,
                                              subset)  # f"{root_dir}/{train_test_val}/{train_test_val}.txt"
        with open(name_filepath, 'r') as f:
            self.filenames = f.read().splitlines()
        print("Loaded {} subset with {} images using label {}".format(subset, self.__len__(), gt_type))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        filename = self.filenames[index]
        img = Image.open(self.data_path + '/images/' + filename + '.jpg')
        gt = Image.open(self.data_path + '/' + self.gt_type + '/' + filename + '.png')
        
        # Resize image to the desired size
        if self.img_size is not None:
            img = self.resize_raw(img) 
            gt = self.resize_gt(gt)
        
        # Convert both to tensor
        img = self.to_tensor(img)
        if self.gt_type == 'labels_3':
            gt = torch.from_numpy(np.array(gt, np.int16, copy=False))
        else:
            gt = self.to_tensor(gt)  # this operation normalize img/255 too
        gt = gt.squeeze_().long()
        return img, gt
