import glob, os

HOME = "/content/gdrive/MyDrive/YOLOV3_Custom"

dataset_path = f'{HOME}/images'


# Percentage of images to be used for the test set
percentage_test = 10;

# Create and/or truncate train.txt and test.txt
file_train = open(f'{HOME}/train.txt', 'w')
file_test = open(f'{HOME}/test.txt', 'w')

# Populate train.txt and test.txt
counter = 1
index_test = round(100 / percentage_test)
for pathAndFilename in glob.iglob(os.path.join(dataset_path, "*.jpg")):
    title, ext = os.path.splitext(os.path.basename(pathAndFilename))

    if counter == index_test:
        counter = 1
        file_test.write(f"{dataset_path}/{title}.jpg\n")
    else:
        file_train.write(f"{dataset_path}/{title}.jpg\n")
        counter = counter + 1