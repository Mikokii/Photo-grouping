from sentence_transformers import SentenceTransformer, util
from PIL import Image
import os
import shutil

def LoadDirectoriesMenu():
    images_paths = []
    while True:
        print("Type path to directory containing images:")
        inp = input()
        if os.path.isdir(inp):
            images_paths = LoadDirectory(inp, recursive, images_paths)
            while True:
                inp = input("Do you want to load images from another directory (Y or N)? ")
                if inp.lower() == 'y':
                    break
                elif inp.lower() == 'n':
                    return images_paths
                else:
                    print("Type Y or N.")
        else:
            print("Typed path is wrong or is not a directory. Try again")

def LoadDirectory(inp, recursive, images_paths):
    if recursive:
        for root, _, files in os.walk(inp):
            if not(root in all_directories):
                all_directories.append(root)
            for file in files:
                if file.endswith(supported_extensions):
                    if not(os.path.join(root, file) in images_paths):
                        images_paths.append(os.path.join(root, file))
    else:
        if not(inp in all_directories):
            all_directories.append(inp)
            for file in os.listdir(inp):
                if file.endswith(supported_extensions):
                    images_paths.append(os.path.join(inp, file))
    return images_paths

def DeleteDuplicates(images_paths):
    for i in range(len(images_paths)):
        for j in range(i+1, len(images_paths)):
            if open(images_paths[i], "rb").read() == open(images_paths[j], "rb").read():
                images_paths.pop(i)
    return images_paths

def FindDirIndex(element, matrix):
    for i in range(len(matrix)):
        if element in matrix[i]:
            return i

def FindSimilarImages(threshold, ceiling = 2):
    similar_images = []
    for image in processed_images:
        if image[0] > threshold:
            if image[0] <= ceiling:
                similar_images.append(image)
        else:
            return similar_images
    return similar_images

def FirstDivision(threshold):
    similar_images = FindSimilarImages(threshold)
    splitted_directories = []
    for i in range(len(images_paths)):
        img = images_paths[i]
        for _, image_id1, image_id2 in similar_images:
            if i == image_id1:
                img2 = images_paths[image_id2]
            elif i == image_id2:
                img2 = images_paths[image_id1]
            else:
                continue
            if any(img in directory for directory in splitted_directories):
                index = FindDirIndex(img, splitted_directories)
                if not(img2 in splitted_directories[index]):
                    splitted_directories[index].append(img2)
            elif any(img2 in directory for directory in splitted_directories):
                index = FindDirIndex(img2, splitted_directories)
                if not(img in splitted_directories[index]):
                    splitted_directories[index].append(img)
            else:
                splitted_directories.append([img, img2])
    return splitted_directories

def SecondDivision(threshold, splitted_directories):
    ceiling = 0.9
    similar_images = FindSimilarImages(threshold, ceiling)
    not_used_images = []
    for img in images_paths:
        if not(any(img in directory for directory in splitted_directories)):
            not_used_images.append(img)
            img_index = images_paths.index(img)
            for i in range(len(splitted_directories)):
                break_status = True
                directory = splitted_directories[i]
                for el in directory:
                    dir_image_index = images_paths.index(el)
                    if not(any(img_index in pair and dir_image_index in pair for pair in similar_images)):
                        break_status = False
                        break
                if break_status:
                    splitted_directories[i].append(img)
                    break
            if break_status:
                not_used_images.remove(img)
    for i in range(len(not_used_images)):
        img1 = not_used_images[i]
        index1 = images_paths.index(img1)
        for j in range(i+1, len(not_used_images)):
            img2 = not_used_images[j]
            index2 = images_paths.index(img2)
            if any(index1 in pair and index2 in pair for pair in similar_images):
                if (any(img1 in directory for directory in splitted_directories)):
                    index = FindDirIndex(img1, splitted_directories)
                    splitted_directories[index].append(img2)
                elif (any(img2 in directory for directory in splitted_directories)):
                    index = FindDirIndex(img2, splitted_directories)
                    splitted_directories[index].append(img1)
                else:
                    splitted_directories.append([img1, img2])
    return splitted_directories

def MergeSimilarDirectories(threshold, splitted_directories):
    similar_images = FindSimilarImages(threshold)
    merged_status = True
    while merged_status:
        merged_status = False
        for i in range(len(splitted_directories)):
            dir1 = splitted_directories[i]
            for j in range(i+1, len(splitted_directories)):
                break_status = False
                dir2 = splitted_directories[j]
                for img1 in dir1:
                    index1 = images_paths.index(img1)
                    for img2 in dir2:
                        index2 = images_paths.index(img2)
                        if not(any(index1 in pair and index2 in pair for pair in similar_images)):
                            break_status = True
                            break
                    if break_status == True:
                        break
                if not(break_status):
                    merged_status = True
                    splitted_directories[i].extend(splitted_directories.pop(j))
                    break
            if merged_status:
                break
    return splitted_directories

def AddNotUsedImages(splitted_directories):
    splitted_directories.append([])
    for img in images_paths:
        if not(any(img in dir for dir in splitted_directories)):
            splitted_directories[-1].append(img)
    if splitted_directories[-1] == []:
        splitted_directories.pop(-1)
        return False
    return splitted_directories

# Load the OpenAI CLIP Model
print('Loading CLIP Model...')
model = SentenceTransformer('clip-ViT-B-32')

exts = Image.registered_extensions()
supported_extensions = tuple(ex for ex, f in exts.items() if f in Image.OPEN)

all_directories = []
while True:
    inp = input("Do you want to scan directories inside chosen directories recursively (Y or N)? ")
    if inp.lower() == 'y':
        recursive = True
        break
    elif inp.lower() == 'n':
        recursive = False
        break
    else:
        print("Type Y or N.")
images_paths = LoadDirectoriesMenu()
images_paths = DeleteDuplicates(images_paths)

# Compute the embeddings
encoded_images = model.encode([Image.open(filepath) for filepath in images_paths], batch_size=128, convert_to_tensor=True, show_progress_bar=True)

# Compare images aganist all other images and return a list sorted by the pairs that have the highest cosine similarity score
processed_images = util.paraphrase_mining_embeddings(encoded_images)

first_threshold = 0.9
splitted_directories = FirstDivision(first_threshold)
second_threshold = 0.85
splitted_directories = SecondDivision(second_threshold, splitted_directories)
splitted_directories = MergeSimilarDirectories(second_threshold, splitted_directories)
if not(AddNotUsedImages(splitted_directories)):
    last_dir_status = False
else:
    last_dir_status = True
print("{} images saved in {} direcotories".format(len(images_paths), len(splitted_directories)))
while True:
    print("Type path to directory in which you want to have directory with images:")
    inp = input()
    if os.path.isdir(inp):
        parent_dir_path = inp
        break
    else:
        print("Typed path is wrong or is not a directory. Try again")
while True:
    print("Type name of directory to save images:")
    inp = input()
    dir_path = os.path.join(parent_dir_path, inp)
    if os.path.isdir(dir_path):
        print("Directory with that name already exists. Choose another name for directory.")
    else:
        os.mkdir(dir_path)
        break
for i in range(len(splitted_directories)):
    dir = splitted_directories[i]
    if i == len(splitted_directories) - 1 and last_dir_status:
        subdir_path = os.path.join(dir_path, "Other")
    else:
        subdir_path = os.path.join(dir_path, str(i+1))
    os.mkdir(subdir_path)
    for img in dir:
        shutil.copy(img, subdir_path)
while True:
    print("Images saved successfully")
    inp = input("Check if you want to save changes (Y or N):")
    if inp.lower() == "y":
        while True:
            inp = input("Do you want to delete images from the previous locations? (Y or N):")
            if inp.lower() == "y":
                for img in images_paths:
                    os.remove(img)
                for dir in reversed(all_directories):
                    if len(os.listdir(dir)) == 0:
                        os.rmdir(dir)
                print("Images succesfully deleted.")
                print("All done!")
                exit()
            elif inp.lower() == "n":
                print("All done!")
                exit()
            else:
                print("Type Y or N.")
    elif inp.lower() == "n":
        shutil.rmtree(dir_path)
        print("Changes undone.")
        break
    else:
        print("Type Y or N.")