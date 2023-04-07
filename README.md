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

# **Step 3**

**Assigning path of the home directory to `HOME` for easy file handling**
```
HOME = os.path.join(drive_path, "YOLOV3_Custom")
HOME
```

Creating Home directory to store all the files
```
os.mkdir(f"{HOME}")
```
Changing current directory to `HOME`
```
%cd {HOME}
```

# **Step 4**
## *Creating Image Directory*
We will train our model to recognise pistols in this project, thus we must gather the images and its annotaions and save them in the ***YOLOV3_Custom/images*** directory.

## Unzip Files
Your Google Drive file location that you wish to unzip is the path in the cell below.

In our case images and text files should be saved in **YOLOV3_Custom/images** directory. For e.g. **image1.jpg** should have a text file **image1.txt**.

```
%cd {HOME}
!unzip "zip/images.zip"
```

# **Step 5**
## *Creating Custom.names file*
Labels of our objects should be saved in **YOLOV3_Custom/custom.names** file, each line in the file corresponds to an object. In our case since we have only one object class, the file should contain the following.


**custom.names**
```
Pistol
```


# **Step 8**
## *Creating YOLO data file*
Make a file called "detector.data" in the "YOLOV3_Custom" directory that contains details about the train and test data sets.

**detector.data**
```
classes=1
train=/train.txt
valid=/test.txt
names=/custom.names
backup=/backup
```

# **Step 9**
## *Cloning Directory to use Darknet*
Darknet, an open source neural network framework, will be used to train the detector. Download and create a dark network

```
%cd {HOME}
!git clone https://github.com/AlexeyAB/darknet
```
```
# Change current working directory to Darknet
%cd darknet
```
### Change makefile to have GPU and OPENCV enabled, and other parameters for faster computation.
```
!sed -i 's/OPENCV=0/OPENCV=1/' Makefile 
!sed -i 's/GPU=0/GPU=1/' Makefile
!sed -i 's/CUDNN=0/CUDNN=1/' Makefile
!sed -i 's/CUDNN_HALF=0/CUDNN_HALF=1/' Makefile
```
### Model Compilation
```
# compiling the model
!make
```

# **Step 10**
## *Making changes in the yolo Configuration file*
We can choose the YOLOv3 configuration file based on the performance that is necessary. **Yolov3.cfg** will be used in this example. The file from **darknet/cfg/yolov3.cfg** can be copied to **YOLOV3_Custom/cfg/yolov3-custom.cfg**.

_

The neural network weights are iteratively adjusted as the images are trained. We may use large training sets, making it resource-intensive to update the weights for the full training set in a single cycle. The **batch** parameter is specified to utilise a small set of images to iteratively update the weights. It is initially set at 64.

_

The parameter **max batches** determines the maximum number of iterations for which our network should be trained. You can calculate its value with the formula **max_batches = classes*2000**, i.e. **2000** for a single class, and modify the line steps to **80%** and **90%** of max batches, which is **1600,1800**. 

when the model reaches to **2000 epochs** during training for 1 class, it will stop the training, because we defined that in our max_batches parameter, if you want to train your model beyound from that, increase the value of max_batches, and also increase its steps, like when you change the value to **max batches=6000**, modify the line steps to **80%** and **90%** of max batches, i.e. steps=4800,5400.  

--

The **classes** and **filters** parameters of the [yolo] and [convolutional] layers immediately preceding the [yolo] layers must be updated.


Since there is just one class in this project (Pistol), we will update the class parameter in the [yolo] layers to **1** at lines **610, 696, and 783**.

The **filters** parameter will also need to be updated based on the classes count: **filters=(classes + 5) * 3**. We should set **filters=18** for a single class at **line numbers: 603, 689, 776**.

YOLOV3_Custom/cfg/yolov3-custom.cfg contains all configuration changes.

# **Step 11**
## *Downloading Pre-trained weights*
To train our object detector, we can use the pre-trained weights that have already been trained on a large data sets. The pre-trained weights are available [here](https://pjreddie.com/media/files/darknet53.conv.74), and they can be downloaded to the root directory.

```
%cd {HOME}/pt-weights

# Download weights darknet model 53
!wget https://pjreddie.com/media/files/darknet53.conv.74
```

```
# changing current drive to the darknet
%cd {HOME}/darknet
```

# **Step 12**
## *Training the model*
As soon as we have all the necessary files and annotated photographs, we can begin our training.
Up till the loss reaches a predefined level, we can keep training. Weights for the custom detector are initially saved once every 100 iterations until 1,000 iterations, after which they are saved once every 10,000 iterations by default.

We can do detection using the generated weights after the training is finished.

```
!./darknet detector train {HOME}/detector.data {HOME}/cfg/yolov3-custom.cfg {HOME}/pt-weights/darknet53.conv.74 -dont_show
```

## *Continue training from where you left*
Continue training from where you left off, your Model training can be stopped due to multiple reasons, like the notebook time out, notebook craches, due to network issues,  and many more,  so you can start your training from where you left off, by passing the previous trained weights.

```
!./darknet detector train {HOME}/detector.data {HOME}/cfg/yolov3-custom.cfg {HOME}/backup/yolov3-custom_4000.weights -dont_show
```

# **Step 13**
## *Calculating Mean average precision of Specific Weights*
As we have trained our weights up to 2000 epochs, we will calculate the mean average precision of our trained weights.

```
!./darknet detector map {HOME}/detector.data {HOME}/cfg/yolov3-custom.cfg {HOME}/backup/yolov3-custom_4000.weights -dont_show
```

# **Step 14** 
## *Test your custom Object Detector*
**Make changes to your custom config file**
*   change line batch to batch=1
*   change line subdivisions to subdivisions=1

You can do it either manually or by simply running the code below
```
#set your custom cfg to test mode 
%cd {HOME}/cfg
!sed -i 's/batch=64/batch=1/' yolov3-custom.cfg
!sed -i 's/subdivisions=16/subdivisions=1/' yolov3-custom.cfg
```

## *Run detector on an image*
```
# define helper function imShow
def imShow(path):
  import cv2
  import matplotlib.pyplot as plt
  %matplotlib inline

  image = cv2.imread(path)
  height, width = image.shape[:2]
  resized_image = cv2.resize(image,(3*width, 3*height), interpolation = cv2.INTER_CUBIC)

  fig = plt.gcf()
  fig.set_size_inches(18, 10)
  plt.axis("off")
  plt.imshow(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))
  plt.show()

```
```
%cd {HOME}/darknet
# run your custom detector with this command (upload an image to your google drive to test, the thresh flag sets the minimum accuracy required for object detection)
!./darknet detector test {HOME}/detector.data {HOME}/cfg/yolov3-custom.cfg {HOME}/backup/yolov3-custom_2000.weights {HOME}/pistol_Image.jpg -thresh 0.3
```

```
%cd {HOME}/darknet
imShow('predictions.jpg')
```
![predictions](https://user-images.githubusercontent.com/76246927/230632314-8c2fa863-36ab-4eb6-adf7-a4c9095ca577.jpg)
