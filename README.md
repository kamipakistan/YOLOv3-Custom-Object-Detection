# **Custom object training with YOLOv3 and Darknet**

The phrase "You Only Look Once" is referred to as YOLO. 
This method detects and recognizes various objects in images (in real-time). 
YOLO performs object detection as a regression problem and outputs the class probabilities of the detected photos.


Convolutional neural networks (CNN) are used by the YOLO method to recognize items instantly. 
As the name implies, the algorithm only needs one forward propagation through a neural network to detect objects.


As a result, a single algorithm run is used to do prediction throughout the entire image. 
Simultaneously, several class probabilities and bounding boxes are predicted using CNN.


There are numerous variations of the YOLO algorithm. One of them is 
**YOLOv3**.


# **Step 1**
## Enabling and testing the GPU

The notebook's GPUs must first be enabled:

- Select Notebook Settings under Edit.
- choose GPU using the Hardware Accelerator drop-down

Next, we'll check if Tensorflow can connect to the GPU: By executing the following code, you may quickly determine whether the GPU is enabled.
```
# Check if NVIDIA GPU is enabled
!nvidia-smi
```

# **Step 2**
Mounting the Drive to store and load files.

```
from google.colab import drive
drive.mount('/content/gdrive')
```
Let's make the required files and directories for training with custom objects.


1. **YOLOV3_Custom/images**
2. **YOLOV3_Custom/custom.names** 
3. **YOLOV3_Custom/train.txt**
4. **YOLOV3_Custom/test.txt**
5. **YOLOV3_Custom/backup**
6. **YOLOV3_Custom/detector.data**
7. **YOLOV3_Custom/cfg**


**Changing directory to drive Directory**
```
# changing directory to the google drive
import os
drive_path = os.path.join(os.getcwd(), "gdrive/MyDrive")
%cd {drive_path}
```
  /content/gdrive/MyDrive

# **Step 3**
Creating Home directory to store all the files

```
HOME = os.path.join(drive_path, "YOLOV3_Custom")
HOME
```
  /content/gdrive/MyDrive/YOLOV3_Custom

```
os.mkdir(f"{HOME}")
```
**Assigning path of the home directory to `HOME` for easy file handling**
```
%cd {HOME}
```
---- /content/gdrive/MyDrive/YOLOV3_Custom
