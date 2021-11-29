# Image_Edit
This program was created to automatically edit images, removing the background, squaring the image and ensuring a consistent border size across all images (10%).

## Installation
### Pre-requisites:
1. Install brew, follow the guide up to and including Step 3: https://learnopencv.com/install-opencv3-on-macos/
2. Tap this version of brew (Follow instructions on the github page): https://github.com/esimov/homebrew-opencv
3. In the terminal type the following command ```brew install theora```
4. Build OpenCV from source, following this guide: https://www.timpoulsen.com/2018/build-opencv-and-install-from-source.html

### Program Installation:
1. From the terminal, cd to the desired install location and type the following command: ```git clone https://github.com/Danielwoodh/Image_Edit.git```

## Usage

1. Open the terminal window and Cd into the directory containing the code.
2. Run the following command in the terminal: ```python3 opencv_test.py```
3. Follow the instructions on the terminal screen.

## Potential Issues

- If the image to be edited is white/cream, the detection may not work properly and it may break the program.
- Some images with abnormal aspect ratios may cause bugs.
