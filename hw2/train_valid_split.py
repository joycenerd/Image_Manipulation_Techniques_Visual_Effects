import glob
import argparse
import math
import random
import os
import shutil



parser = argparse.ArgumentParser()
parser.add_argument('--data-root', type=str, default='/eva_data/zchin/cityscapes/all_train',
                    help='trainig image saving directory')
parser.add_argument('--ratio', type=float, default=0.2, help='validation data ratio')
args = parser.parse_args()

if __name__ == '__main__':
    train_dir=args.data_root.replace('all_train','train')
    val_dir=args.data_root.replace('all_train','val')
    if not os.path.isdir(train_dir):
        os.mkdir(train_dir)
        os.mkdir(val_dir)


    data_size = len(os.listdir(args.data_root))
    print(f'data size: {data_size}')
    valid_size = math.floor(data_size * args.ratio)

    img_list = [img_path for img_path in glob.glob(args.data_root+'/*')]

    idx = random.sample(range(data_size), valid_size)

    for i,src_img_path in enumerate(img_list):
        img_name=src_img_path.split('/')[-1]
        if i in idx:
            dest_img_path=os.path.join(val_dir,img_name)
        else:
            dest_img_path=os.path.join(train_dir,img_name)
        shutil.copyfile(src_img_path,dest_img_path)

    train_size = len(glob.glob1(train_dir, "*.jpg"))
    valid_size = len(glob.glob1(val_dir, "*.jpg"))
    print(f'train size: {train_size}\tvalid size: {valid_size}')