import os

# path of the images folder which you want to rename it.
sourcePath = r"/home/kamipakistan/PycharmProjects/AI/PixeletsAI/Renaming Files/split-500"
# path to which you want to place the images after renaming it
destPath = "/home/kamipakistan/PycharmProjects/AI/PixeletsAI/Renaming Files/renamed"
count = 0
for i in os.listdir(sourcePath):
    src = f"{sourcePath}/{i}"
    dest = f"{destPath}/image_{count}.jpg"
    if i.split('.')[-1] == "jpg":
        os.rename(src, dest)
        count += 1

print(f"{count} JPG Images Renamed")
