import os
import glob 
import random
from PIL import Image

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
import seaborn as sns


parser=argparse.ArgumentParser()
parser.add_argument('--dataroot',type=str,default='/eva_data/zchin/mnist_dataset',help='dir to save the data')
parser.add_argument('--logdir',type=str,default='/eva_data/zchin/imvfx_hw2/cGAN',help='logging information save dir')
parser.add_argument('--device',type=str,default='cuda:0',help='gpu device')
parser.add_argument('--epochs',type=int,default=100,help='number of epochs to train')
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
    

def data_loader(dataroot,batch_size,workers):
    # Define training and testing data augmentation
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Create the train dataset
    train_dataset = dset.MNIST(dataroot, train=True, download=True, transform=transform)
    # Create the train dataloader
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers)
    
    return train_dataloader


def show_example(train_dataloader):
    examples = enumerate(train_dataloader)
    batch_idx, (example_data, example_targets) = next(examples)
    plt.rcParams["font.family"]="serif"
    fig = plt.figure()
    for i in range(6):
        plt.subplot(2, 3, i+1)
        plt.tight_layout()
        plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
        plt.title("Ground Truth: {}".format(example_targets[i]))
        plt.xticks([])
        plt.yticks([])
    plt.savefig('train_example.png')


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
    def __init__(self):
        super(Generator, self).__init__()
        # The size of generator input.
        input_dim = nz + nclass
        # The size of generator output.
        output_dim = image_size*image_size

        # It will have a 10-dimensional encoding for all the 10 digits.
        self.label_embedding = nn.Embedding(nclass, nclass)
        
        self.main = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Linear(256, 512),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Linear(512, 1024),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Linear(1024, output_dim),
            nn.Tanh()
        )
    
    def forward(self, x, labels):
        c = self.label_embedding(labels)
        x = torch.cat([x,c], dim=1)
        output = self.main(x)
        return output


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        # The size of discriminator input.
        input_dim = image_size*image_size + nclass
        # The size of discriminator output.
        output_dim = 1
        # It will have a 10-dimensional encoding for all the 10 digits.
        self.label_embedding = nn.Embedding(nclass, nclass)

        self.main = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout(0.3),

            nn.Linear(1024, 512),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout(0.3),

            nn.Linear(256, output_dim),
            nn.Sigmoid()
        )


    def forward(self, x, labels):
        c = self.label_embedding(labels)
        x = torch.cat([x, c], dim=1)
        output = self.main(x)

        return output


def train():
    # Create the generator
    netG = Generator().to(device)

    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.2.
    weights_init(netG)

    # Print the model
    print(netG)

    # Create the Discriminator
    netD = Discriminator().to(device)

    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.2.
    weights_init(netD)

    # Print the model
    print(netD)

    # Initialize Loss functions
    criterionGAN = nn.BCELoss().to(device)

    # Establish convention for real and fake labels during training
    real_label = 1.0
    fake_label = 0.

    # Create the latent vectors and condition labels that we will use to visualize
    # the progression of the generator
    fixed_noise = torch.randn(100, nz, device=device)
    fixed_labels = torch.randint(0, 10, (100,)).to(device)

    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

    # Training Loop
    iters = 0

    lossG_list=[]
    lossD_list=[]

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
            true_img = data[0].view(-1, image_size*image_size).to(device)
            b_size = true_img.size(0)
            digit_labels = data[1].to(device)
            # Forward pass real batch through D
            pred_real = netD(true_img, digit_labels).view(-1)
            # Calculate loss on all-real batch
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            errD_real = criterionGAN(pred_real, label)

            ## Train with all-fake batch
            # Generate fake image batch with G
            noise = torch.randn(b_size, nz, device=device)
            fake_labels = torch.randint(0, 10, (b_size,), device=device)
            fake = netG(noise, fake_labels)
            # Classify all fake batch with D
            pred_fake = netD(fake.detach(), fake_labels).view(-1)
            # Calculate D's loss on the all-fake 
            label = torch.full((b_size,), fake_label, dtype=torch.float, device=device)
            errD_fake = criterionGAN(pred_fake, label)

            # Compute error of D as sum over the fake and the real batches
            errD = (errD_real + errD_fake)*0.5
            # Calculate the gradients for this batch
            errD.backward()
            # Update D
            optimizerD.step()

            ############################
            # (2) Update G network
            ###########################
            netG.zero_grad()
            fake = netG(noise, fake_labels)
            # Since we just updated D, perform another forward pass of all-fake batch through D
            pred_fake = netD(fake, fake_labels).view(-1)
            label.fill_(real_label)   # fake labels are real for generator cost
            # Calculate G's loss based on this output
            errG = criterionGAN(pred_fake, label)
            # Calculate gradients for G
            errG.backward()
            # Update G
            optimizerG.step()

            # Output training stats\
            # Set the info of the progress bar
            # Note that the value of the GAN loss is not directly related to
            # the quality of the generated images.
            # progress_bar.set_description({
            #     'Loss_D': round(errD.item(), 4),
            #     'Loss_G': round(errG.item(), 4),
            #     'Epoch': epoch+1,
            #     'Step': iters,
            # })
            progress_bar.set_description(f'[Loss_D: {round(errD.item(), 4)}][Loss_G: {round(errG.item(), 4)}][Epoch: {epoch+1}][Step: {iters}]')
            lossG_list.append(errG.item())
            lossD_list.append(errD.item())

            iters += 1

        # Evaluation 
        netG.eval()
        with torch.no_grad():
            generated_imgs = netG(fixed_noise, fixed_labels)
            show_imgs = generated_imgs.cpu().view(-1, image_size, image_size)
            # Save evaluation results
            if (epoch+1) % save_steps == 0:
                filename = os.path.join(log_dir, f'Epoch_{epoch+1:03d}.jpg')
                img_samples = (generated_imgs.view(-1, 1, image_size, image_size).data + 1) / 2.0
                torchvision.utils.save_image(img_samples, filename, nrow=10)
            for x in show_imgs:
                plt.title('Condition label: {}'.format(str(fake_labels[0].item())))
                plt.imshow(x.detach().numpy(), interpolation='nearest',cmap='gray')
                plt.show()
                break
        print('Evaluation done!')

        # Save the checkpoints.
        if (epoch+1) % save_steps == 0:
            netG_out_path = os.path.join(ckpt_dir, 'netG_epoch_{}.pth'.format(epoch+1))
            netD_out_path = os.path.join(ckpt_dir, 'netD_epoch_{}.pth'.format(epoch+1))
            torch.save(netG.state_dict(), netG_out_path)
            torch.save(netD.state_dict(), netD_out_path)

    return lossG_list,lossD_list


if __name__=='__main__':
    # Set random seed for reproducibility
    same_seeds(123)

    # Root directory for dataset
    dataroot = args.dataroot

    # Number of workers for dataloader
    workers = 3

    # Batch size during training
    batch_size = 64

    # The size of images
    image_size = 28

    # Number of channels in the training images. For gray images this is 1
    nc = 1

    # Size of z latent vector 
    nz = 100

    # Number of image classes (i.e. 0-9)
    nclass = 10

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
    log_dir = os.path.join(args.logdir, 'logs')    # For evaluation results
    ckpt_dir = os.path.join(args.logdir, 'checkpoints')    # For trained model weights
    result_dir = os.path.join(args.logdir, 'results')    # For testing results
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)

    if args.mode=='train':
        train_dataloader=data_loader(args.dataroot,batch_size,workers)
        show_example(train_dataloader)

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
        plt.savefig('cGAN_loss.png')
    
    if args.mode=='test':
        # load gernerator
        netG = Generator()
        netG.load_state_dict(torch.load(os.path.join(ckpt_dir, 'netG_epoch_{}.pth'.format(num_epochs))))
        netG.eval()
        netG.to(device)

        ###############################################################################
        # TODO: Store the generated 10*10 grid images
        # Implementation 1-3
        ###############################################################################

        # Generate 100 images and make a grid to save them.
        n_output = 100
        fixed_noise = torch.randn(100, nz, device=device)
        fixed_labels = torch.randint(0, nclass, (100,)).to(device)
        imgs_sample = netG(fixed_noise, fixed_labels)
        img_samples = (imgs_sample.view(-1, 1, image_size, image_size).data + 1) / 2.0
        filename = os.path.join(result_dir, 'result.jpg')

        # Show the images in notebook.
        plt.rcParams["font.family"]="serif"
        grid_img = torchvision.utils.make_grid(img_samples, nrow=10)
        plt.figure(figsize=(10, 10))
        plt.imshow(grid_img.cpu().permute([1, 2, 0]))
        # plt.show()
        plt.savefig(filename)

        ###############################################################################
        # TODO: Generate images with specified condition labels
        # Implementation 1-4
        ###############################################################################
        n_output = 10
        fixed_noise = torch.randn(10, nz, device=device)
        fixed_labels=list(range(0,10))
        fixed_labels=torch.tensor(fixed_labels,dtype=torch.long,device=device)
        imgs_sample = netG(fixed_noise, fixed_labels)
        img_samples = (imgs_sample.view(-1, 1, image_size, image_size).data + 1) / 2.0
        filename = os.path.join(result_dir, 'result_condition.jpg')

        plt.rcParams["font.family"]="serif"
        grid_img = torchvision.utils.make_grid(img_samples, nrow=5)
        plt.figure(figsize=(5, 2))
        plt.imshow(grid_img.cpu().permute([1, 2, 0]))
        # plt.show()
        plt.savefig(filename)
