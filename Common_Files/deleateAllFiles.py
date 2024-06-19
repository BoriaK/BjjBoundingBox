import os
import shutil

# this script goes to every subfolder in root and delets every file in there
# Root = r'../dataSet/CheXpert-v1.0-small'  # path
Root = r'..\Dataset\Sub_Set_v01\train\bed'


def delAllFiles(source_path):
    for root, dirs, files in os.walk(source_path):  # recursively walk through all the folders/subfolders in root
        for file in files:
            if file.endswith(".json"):
                path_file = os.path.join(root, file)
                os.remove(path_file)

    # try:
    #     shutil.rmtree(source_path)  # to remove the entire folder after the files have been removed
    # except OSError as e:
    #     print("Error: %s : %s" % (source_path, e.strerror))


def main():
    delAllFiles(Root)


main()
