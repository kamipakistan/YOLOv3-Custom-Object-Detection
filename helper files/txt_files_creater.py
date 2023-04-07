import os


# This function will create train.txt and test.txt files
def trainNtestFilesCreater(dataSetPath, mappingPath, train_size=0.80):
    listD = os.listdir(dataSetPath)
    # filtering the images
    images = [file for file in listD if file.split(".")[1] == "jpg"]
    # getting the training size
    trainS = int(len(images) * train_size)

    # creating txt files
    train = open("train.txt", "wt")
    test = open("test.txt", "wt")

    counter = 1
    for file in images:
        if counter <= trainS:
            # writing in train.txt file
            train.writelines(f"{mappingPath}/{file}\n")
        else:
            # writing in text.txt file
            test.writelines(f"{mappingPath}/{file}\n")
        counter += 1
    train.close()
    test.close()
    print(f"{counter} files processed")


# Path of the custom dataset which includes images along with its annotations
dataSetPath = r"/home/kamipakistan/PycharmProjects/AI/FYP Yolov3/images"
# provide the complete path of the drive you want to write in your train.txt file and test.txt fiel
mappingPath = "/content/gdrive/MyDrive/custom_data/images"

# calling creater function to generate files
trainNtestFilesCreater(dataSetPath, mappingPath, train_size= 0.80)
