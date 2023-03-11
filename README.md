# a-ipm
## Inverse Perspective Mapping
Inverse perspective mapping allows us to extrapolate a bird's eye view (BEV) perspective on the current frame. To run ipm.py, first create two folders named input and output, where the input folder contains original images from the Cityscapes dataset and the output folder contains the resulting segmented images from the model output.

## Compute Steering Angle
The steering angle computation is done by iterating through pixels located at the boundaries of the bot's physically reachable area and finding a contiguous segment of pixels that is labelled as traversable terrain (dictated by our segmentation). To run postprocess.py, first create an output folder that contains resulting segmented images from the model output.