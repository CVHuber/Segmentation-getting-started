import os
def rename(img_dir, mask_dir):
    img_list = os.listdir(img_dir)
    mask_list = os.listdir(mask_dir)
    for f in img_list:
        os.rename(os.path.join(img_dir,f), os.path.join(img_dir,f.replace("_training.tif", ".tif")))
    for f in mask_list:
        os.rename(os.path.join(mask_dir,f), os.path.join(mask_dir,f.replace("_manual1.gif", ".gif")))

if __name__ == "__main__":
    rename("./data/training/images", "./data/training/1st_manual")