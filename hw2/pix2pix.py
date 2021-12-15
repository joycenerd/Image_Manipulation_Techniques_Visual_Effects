import os
import glob 
import random
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import os


parser=argparse.ArgumentParser()
parser.add_argument('--dataroot',type=str,default='/eva_data/zchin/cityscapes',help='dir to save the data')
parser.add_argument('--logdir',type=str,default='/eva_data/zchin/imvfx_hw2/pix2pix_cityscapes',help='logging information save dir')
parser.add_argument('--device',type=str,default='cuda:0',help='gpu device')
parser.add_argument('--epochs',type=int,default=100,help='number of epochs to train')
parser.add_argument('--bs',type=int,default=1,help='batch size for training')
parser.add_argument('--mode',type=str,default='train',help='train or test')
args=parser.parse_args()


def same_seeds(seed):
    # Python built-in random module
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class CustomDataset(Dataset):
    def __init__(self, root, subfolder='train', transform=None):
        super(CustomDataset, self).__init__()
        self.path = os.path.join(root, subfolder)
        self.subfolder = subfolder
        self.image_filenames = [x for x in sorted(os.listdir(self.path))]
        self.resize_scale = 286
        self.crop_size = 256
        self.transform = transform

    def __getitem__(self, index):
        # Load Image
        img_path = os.path.join(self.path, self.image_filenames[index])
        img = Image.open(img_path)
        # Pay attention to the position of the source and target images
        input = img.crop((img.width // 2, 0, img.width, img.height))
        target = img.crop((0, 0, img.width // 2, img.height))

        if self.subfolder == 'train':
            # Data augumentation for training set
            # Resize 
            input = input.resize((self.resize_scale, self.resize_scale), Image.BILINEAR)
            target = target.resize((self.resize_scale, self.resize_scale), Image.BILINEAR)
            # Random crop
            x = random.randint(0, self.resize_scale - self.crop_size + 1)
            y = random.randint(0, self.resize_scale - self.crop_size + 1)
            input = input.crop((x, y, x + self.crop_size, y + self.crop_size))
            target = target.crop((x, y, x + self.crop_size, y + self.crop_size))
            # Random Horizontal Flip
            if random.random() < 0.5:
                input = input.transpose(Image.FLIP_LEFT_RIGHT)
                target = target.transpose(Image.FLIP_LEFT_RIGHT)

        if self.transform is not None:
            input = self.transform(input)
            target = self.transform(target)
     
        return input, target

    def __len__(self):
        return len(self.image_filenames)


def data_loader():
    # Define training data augmentation
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Create the train dataset
    train_dataset = CustomDataset(dataroot, subfolder='train', transform=transform)
    # Create the train dataloader
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers)

    # Create the validation dataset
    val_dataset = CustomDataset(dataroot, subfolder='val', transform=transform)
    # Create the validation dataloader
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=workers)
    
    return train_dataloader,val_dataloader


# custom weights initialization called on netG and netD
def weights_init(net):
    def init_func(m):
      classname = m.__class__.__name__
      if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
      elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
      elif classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
      
    net.apply(init_func)


class Generator(nn.Module):
    """Generator network with U-Net."""
    def __init__(self):
        super().__init__()

        # U-Net Encoder 
        # 256 * 256
        self.en_layer1 = nn.Sequential(
            nn.Conv2d(nc, ngf, kernel_size=4, stride=2, padding=1)
        )
        # 128 * 128
        self.en_layer2 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf, ngf * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 2)
        )
        # 64 * 64
        self.en_layer3 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf * 2, ngf * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 4)
        )
        # 32 * 32
        self.en_layer4 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf * 4, ngf * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 8)
        )
        # 16 * 16
        self.en_layer5 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 8)
        )
        # 8 * 8
        self.en_layer6 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 8)
        )
        # 4 * 4
        self.en_layer7 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 8)
        )
        # 2 * 2
        self.en_layer8 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1),
        )


        # U-Net Decoder
        # 1 * 1
        self.de_layer1 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 8),
            nn.Dropout(p=0.5)
        )
        # 2 * 2
        self.de_layer2 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ngf * 8 * 2, ngf * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 8),
            nn.Dropout(p=0.5)
        )
        # 4 * 4
        self.de_layer3 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ngf * 8 * 2, ngf * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 8),
            nn.Dropout(p=0.5)
        )
        # 8 * 8
        self.de_layer4 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ngf * 8 * 2, ngf * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 8)
        )
        # 16 * 16
        self.de_layer5 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ngf * 8 * 2, ngf * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 4)
        )
        # 32 * 32
        self.de_layer6 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ngf * 4 * 2, ngf * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 2)
        )
        # 64 * 64
        self.de_layer7 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ngf * 2 * 2, ngf, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf),
            nn.Dropout(p=0.5)
        )
        # 128 * 128
        self.de_layer8 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ngf * 2, nc, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):

        # 1：Encoder
        en1_out = self.en_layer1(x)
        en2_out = self.en_layer2(en1_out)
        en3_out = self.en_layer3(en2_out)
        en4_out = self.en_layer4(en3_out)
        en5_out = self.en_layer5(en4_out)
        en6_out = self.en_layer6(en5_out)
        en7_out = self.en_layer7(en6_out)
        en8_out = self.en_layer8(en7_out)

        # 2：Decoder
        de1_out = self.de_layer1(en8_out)
        de1_cat = torch.cat([de1_out, en7_out], 1)
        de2_out = self.de_layer2(de1_cat)
        de2_cat = torch.cat([de2_out, en6_out], 1)
        de3_out = self.de_layer3(de2_cat)
        de3_cat = torch.cat([de3_out, en5_out], 1)
        de4_out = self.de_layer4(de3_cat)
        de4_cat = torch.cat([de4_out, en4_out], 1)
        de5_out = self.de_layer5(de4_cat)
        de5_cat = torch.cat([de5_out, en3_out], 1)
        de6_out = self.de_layer6(de5_cat)
        de6_cat = torch.cat([de6_out, en2_out], 1)
        de7_out = self.de_layer7(de6_cat)
        de7_cat = torch.cat([de7_out, en1_out], 1)
        de8_out = self.de_layer8(de7_cat)

        return de8_out


class Discriminator(nn.Module):
    """Discriminator network with PatchGAN."""

    def __init__(self):
        super(Discriminator, self).__init__()

        # 256 * 256
        self.layer1 = nn.Sequential(
            nn.Conv2d(nc * 2, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # 128 * 128
        self.layer2 = nn.Sequential(
            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # 64 * 64
        self.layer3 = nn.Sequential(
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # 32 * 32
        self.layer4 = nn.Sequential(
            nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=1, padding=1),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # 31 * 31
        self.layer5 = nn.Sequential(
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=1, padding=1),
            nn.Sigmoid()
        )
        # 30 * 30 : Output patch size

    def forward(self, x):
        layer1_out = self.layer1(x)
        layer2_out = self.layer2(layer1_out)
        layer3_out = self.layer3(layer2_out)
        layer4_out = self.layer4(layer3_out)
        layer5_out = self.layer5(layer4_out)
        return layer5_out


def train():
    ###################################################################################################
    # TODO: Create the Generator by the class and Apply the weights initialize function
    # Implementation 1-5
    ###################################################################################################
    netG = Generator().to(device)
    weights_init(netG)
    print(netG)

    ######################################################################################################
    # TODO: Create the Discriminator by the class and Apply the weights initialize function
    # Implementation 1-5
    ######################################################################################################
    netD = Discriminator().to(device)
    weights_init(netD)
    print(netD)

    ###############################################################################################################
    # TODO: Initialize Loss functions (BCE Loss & L1 Loss)
    #     and Setup Adam optimizers for both Generator and Discriminator
    # Implementation 1-5
    ################################################################################################################
    criterionGAN = nn.BCELoss().to(device)
    criterionL1=nn.L1Loss()
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

    # Establish convention for real and fake labels during training
    real_label = 1.0
    fake_label = 0.

    # Training Loop
    iters = 0

    errD_list=[]
    errG_list=[]

    print("Starting Training Loop...")
    # For each epoch
    for epoch in range(num_epochs):
        progress_bar = tqdm(train_dataloader)
        netG.train()
        netD.train()
        # For each batch in the dataloader
        for i, data in enumerate(progress_bar):

            ############################
            # (1) Update D network
            ###########################
            ## Train with all-real batch
            netD.zero_grad()
            # Format batch
            real_a, real_b = data[0].to(device), data[1].to(device)
            real_ab = torch.cat((real_a, real_b), 1)  # real pairs

            '''
            TODO:
            1. Forward pass real pairs through Discriminator 
            2. Calculate BCE loss between Discriminator's outputs and real labels
            '''
            netD.zero_grad()
            b_size = real_ab.size(0)
            pred_real = netD(real_ab)
            # Calculate loss on all-real batch
            label = torch.ones_like(pred_real,device=device)
            errD_real = criterionGAN(pred_real, label)

            ## Train with all-fake batch
            '''
            TODO: 
            1. Forward pass real_a through Generator to generate fake image
            2. Concatenate real_a and fake image to make fake pairs
            3. Forward pass fake pairs through Discriminator 
            4. Calculate BCE loss between Discriminator's outputs and fake labels
            5. Sum the loss, backward pass to calculate the gradients and update Discriminator (Hint: page 20 in ppt)
            '''
            fake_b=netG(real_a)
            fake_ab=torch.cat((real_a,fake_b),1)
            pred_fake=netD(fake_ab)
            label=torch.zeros_like(pred_fake,device=device)
            errD_fake=criterionGAN(pred_fake,label)
            errD=(errD_fake+errD_real)*0.5
            errD.backward()
            optimizerD.step()

            ############################
            # (2) Update G network
            ###########################
            netG.zero_grad()
            '''
            TODO: 
            1. Forward pass fake pairs through Discriminator 
            2. Calculate BCE loss between Discriminator's outputs and real labels
            3. Calculate L1 loss between fake image and real_b
            4. Sum the loss (Hint: BCELoss() + lamb * L1Loss())
            5. Pass the loss backward to calculate the gradient and update Generator
            '''
            fake_b=netG(real_a)
            fake_ab=torch.cat((real_a,fake_b),1)
            pred_fake=netD(fake_ab).view(-1)
            label=torch.ones_like(pred_fake,device=device)
            errG_GAN=criterionGAN(pred_fake,label)
            errG_L1=criterionL1(fake_b,real_b)*lamb
            errG=errG_GAN+errG_L1
            errG.backward()
            optimizerG.step()

            # Output training stats\
            # Set the info of the progress bar
            # Note that the value of the GAN loss is not directly related to
            # the quality of the generated images.
            # progress_bar.set_infos({
            #     'Loss_D': round(errD.item(), 4),
            #     'Loss_G': round(errG.item(), 4),
            #     'Epoch': epoch+1,
            #     'Step': iters,
            # })
            progress_bar.set_description(f'[Loss_D: {round(errD.item(), 4)}][Loss_G: {round(errG.item(), 4)}][Epoch: {epoch+1}][Step: {iters}]')
            errD_list.append(errD.item())
            errG_list.append(errG.item()) 
            
            iters += 1

        # Evaluation 
        netG.eval()
        for idx, data in enumerate(val_dataloader):
            input, target = data[0].to(device), data[1].to(device)
            output = netG(input)
            # Show result for test data
            fig_size = (input.size(2) * 3 / 100, input.size(3)/100)
            fig, axes = plt.subplots(1, 3, figsize=fig_size)
            imgs = [input.cpu().data, output.cpu().data, target.cpu().data]
            for ax, img in zip(axes.flatten(), imgs):
                ax.axis('off')
                # Scale to 0-255
                img = img.squeeze()
                img = (((img - img.min()) * 255) / (img.max() - img.min())).numpy().transpose(1, 2, 0).astype(np.uint8)
                ax.imshow(img, cmap=None, aspect='equal')
            axes[0].set_title("Input")
            axes[1].set_title("Generated")
            axes[2].set_title("Target")
            plt.subplots_adjust(wspace=0, hspace=0)
            fig.subplots_adjust(bottom=0)
            fig.subplots_adjust(top=1)
            fig.subplots_adjust(right=1)
            fig.subplots_adjust(left=0)
            # Save evaluation results
            if (epoch+1) % save_steps == 0:
                dir = os.path.join(log_dir, f'Epoch_{epoch+1:03d}')
                os.makedirs(dir, exist_ok=True)
                filename = os.path.join(dir, f'{idx+1}.png')
                plt.savefig(filename)
                if idx == 0:
                    print('Show one evalutation result......')
                    plt.show()  # Show the first result
                else:
                    plt.close()
        print('Evaluation done!')

        # Save the checkpoints.
        if (epoch+1) % save_steps == 0:
            netG_out_path = os.path.join(ckpt_dir, 'netG_epoch_{}.pth'.format(epoch+1))
            netD_out_path = os.path.join(ckpt_dir, 'netD_epoch_{}.pth'.format(epoch+1))
            torch.save(netG.state_dict(), netG_out_path)
            torch.save(netD.state_dict(), netD_out_path)
    
    return errD_list,errG_list


def test(test_dataloader, device, netG):
    netG.eval()
    for i, data in enumerate(test_dataloader):
        # input & target image data
        input, target = data[0].to(device), data[1].to(device)

        ################################################################################################
        # TODO: Forward pass input image through Generator to do image-to-image translation
        # Implementation 1-7
        ################################################################################################
        output = netG(input)

        # Show result for test data
        fig_size = (input.size(2) * 3 / 100, input.size(3)/100)
        fig, axes = plt.subplots(1, 3, figsize=fig_size)
        imgs = [input.cpu().data, output.cpu().data, target.cpu().data]   # The output is generated by Generator 
        for ax, img in zip(axes.flatten(), imgs):
            ax.axis('off')
            # Scale to 0-255
            img = img.squeeze()
            img = (((img - img.min()) * 255) / (img.max() - img.min())).numpy().transpose(1, 2, 0).astype(np.uint8)
            ax.imshow(img, cmap=None, aspect='equal')
        plt.subplots_adjust(wspace=0, hspace=0)
        axes[0].set_title("Input")
        axes[1].set_title("Generated")
        axes[2].set_title("Target")

        # Save result
        save_fn = os.path.join(result_dir, 'Test_result_{:d}.jpg'.format(i+1))
        fig.subplots_adjust(bottom=0)
        fig.subplots_adjust(top=1)
        fig.subplots_adjust(right=1)
        fig.subplots_adjust(left=0)
        plt.savefig(save_fn)

        if i < 5:
            plt.show()  # Show the first five test results
        else:    
            plt.close()


if __name__=='__main__':
    # Set random seed for reproducibility
    same_seeds(123)

    img_path = os.path.join(args.dataroot,'train/2.jpg')
    sample_image = Image.open(img_path)
    # Each original image is of size 512 x 256 containing two 256 x 256 images:
    print(sample_image.size)
    plt.figure()
    plt.axis("off")
    plt.imshow(sample_image)
    plt.savefig('sample_image.png')

    # Root directory for dataset
    dataroot = args.dataroot

    # Number of workers for dataloader
    workers = 3

    # Batch size during training
    batch_size = args.bs

    # The size of images 
    image_size = 256

    # Number of channels in the training images. For color images this is 3
    nc = 3

    # Size of feature maps in generator
    ngf = 64

    # Size of feature maps in discriminator
    ndf = 64

    # Number of training epochs
    num_epochs = args.epochs

    # Learning rate for optimizers
    lr = 0.0002

    # Beta1 hyperparam for Adam optimizers
    beta1 = 0.5

    # The weight for L1 loaa
    lamb = 100

    # Number of GPUs available. Use 0 for CPU mode.
    ngpu = 1

    # Save checkpoints every few epochs
    save_steps = 5

    # Decide which device we want to run on
    device = torch.device(args.device if (torch.cuda.is_available() and ngpu > 0) else "cpu")

    # log(img), checkpoints and results directory
    log_dir = os.path.join(args.logdir, 'logs')
    ckpt_dir = os.path.join(args.logdir, 'checkpoints')
    result_dir = os.path.join(args.logdir, 'results')
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)

    train_dataloader,val_dataloader=data_loader()
    print(f'train size: {len(train_dataloader)}')
    
    if args.mode=='train':
        loss_D,loss_G=train()

        ###############################################################################
        # TODO: Plot the training loss value of discriminator and generator
        # Implementation 1-2
        ###############################################################################
        plt.rcParams["font.family"]="serif"
        plt.figure()
        plt.plot(loss_D,label='Discriminator loss')
        plt.plot(loss_G,label='Generator loss')
        plt.xlabel('iterations')
        plt.ylabel('loss')
        plt.legend()
        plt.savefig('pix2pix_loss.png')
    
    if args.mode=='test':
        netG = Generator()
        netG.load_state_dict(torch.load(os.path.join(ckpt_dir, 'netG_epoch_{}.pth'.format(num_epochs))))
        netG.eval()
        netG.to(device)

        # Define data augmentation
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        # Create the test dataset
        test_dataset = CustomDataset(dataroot, subfolder='test', transform=transform)
        # Create the test dataloader
        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=workers)

        test(test_dataloader,device,netG)
