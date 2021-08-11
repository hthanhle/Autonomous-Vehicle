# Autonomous vehicles
## Descriptions
This project develops a semantic segmentation method for self-driving cars. In our method, image pixels are classified into i) road, ii) marker, and iii) background categories.

1. We use the Pytorch segmentation library: https://github.com/qubvel/segmentation_models.pytorch. 
2. The segmentation model is PSPNet [[1]](https://ieeexplore.ieee.org/document/8100143) with the encoder ResNet34 [[2]](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf).
3. The dataset is Mapillary Vistas [[3]](https://www.mapillary.com/dataset/vistas) with the three aforementioned classes. The data should be under: `./data/training/`

**Demo video:**

## Quick start
### Installation
1. Install PyTorch=1.2.0 following [the official instructions](https://pytorch.org/)
2. git clone https://github.com/hthanhle/Autonomous-Vehicle
3. Install dependencies: `pip install -r requirements.txt`

### Test
Please run the following commands: 

1. Train on a single image: `python get_age.py --input test.jpg --output test_out.jpg --detector retinaface --estimator ssrnet`
2. Test on camera: 

**Example 1:** `python get_age_cam_coral.py`

**Example 2:** `python get_age_cam_basic.py`

4. Test on a single video: 
 
**Example 1:** `python get_age_video_ssrnet.py --input test.mp4 --output test_out.mp4`

**Example 2:** `python get_age_video_coral.py --input test.mp4 --output test_out.mp4`
