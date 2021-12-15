# Self-attention GAN implementation by Christian Cosgrove
# Based on the paper by Zhang et al.
# https://arxiv.org/abs/1805.08318

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
from torchvision import datasets, transforms
from torch.autograd import Variable

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from spectral import SpectralNorm
from cond_bn import ConditionalBatchNorm2d
from self_attn import SelfAttention, SelfAttentionPost
from tqdm import tqdm

num_classes = 10

batch_size_mult = 10

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--lr_gen', type=float, default=1e-4)
parser.add_argument('--lr_disc', type=float, default=2e-4)
parser.add_argument('--loss', type=str, default='hinge')
parser.add_argument('--load', type=str)
parser.add_argument('--model', type=str)
parser.add_argument('--dataroot',type=str,default='/eva_data/zchin/mnist_dataset',help='dir to save the data')
parser.add_argument('--epochs',type=int,default=200,help='num of epoch')
parser.add_argument('--logdir',type=str,default='/eva_data/zchin/imvfx_hw2/sagan')
args = parser.parse_args()


channels = 1
leak = 0.1
num_classes = 10
w_g=3


class Generator(nn.Module):
    def __init__(self, z_dim):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        num_classes=10
        channels=1

        self.conv1 = SpectralNorm(nn.ConvTranspose2d(z_dim, 512, 4, stride=1))
        self.bn1 = ConditionalBatchNorm2d(512, num_classes)
        self.conv2 = SpectralNorm(nn.ConvTranspose2d(512, 256, 4, stride=2, padding=(1,1)))
        self.bn2 = ConditionalBatchNorm2d(256, num_classes)
        self.conv3 = SpectralNorm(nn.ConvTranspose2d(256, 128, 4, stride=2, padding=(2,2)))
        self.bn3 = ConditionalBatchNorm2d(128, num_classes)
        self.conv4 = SpectralNorm(nn.ConvTranspose2d(128, 64, 4, stride=2, padding=(1,1)))
        self.bn4 = ConditionalBatchNorm2d(64, num_classes)
        self.conv5 = SpectralNorm(nn.Conv2d(64, channels, 3, stride=1, padding=(1,1)))

    def forward(self, z, label):

        x = z.view(-1, self.z_dim, 1, 1)
        x = self.conv1(x)
        x = self.bn1(x, label)
        x = nn.ReLU()(x)
        x = self.conv2(x)
        x = self.bn2(x, label)
        x = nn.ReLU()(x)
        x = self.conv3(x)
        x = self.bn3(x, label)
        x = nn.ReLU()(x)
        x = self.conv4(x)
        x = self.bn4(x, label)
        x = nn.ReLU()(x)
        x = self.conv5(x)
        x = nn.Tanh()(x)

        return x


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.conv1 = SpectralNorm(nn.Conv2d(channels, 64, 3, stride=1, padding=(2,2)))

        self.conv2 = SpectralNorm(nn.Conv2d(64, 64, 4, stride=2, padding=(1,1)))
        self.conv3 = SpectralNorm(nn.Conv2d(64, 128, 3, stride=1, padding=(1,1)))
        self.conv4 = SpectralNorm(nn.Conv2d(128, 128, 4, stride=2, padding=(1,1)))

        self.conv5 = SpectralNorm(nn.Conv2d(128, 256, 3, stride=1, padding=(1,1)))
        self.conv6 = SpectralNorm(nn.Conv2d(256, 256, 4, stride=2, padding=(1,1)))

        self.attention_size = 32
        self.att = SelfAttention(256, self.attention_size)
        self.att_post = SelfAttentionPost(256, self.attention_size)

        self.conv7 = SpectralNorm(nn.Conv2d(256, 512, 3, stride=1, padding=(1,1)))

        self.embed = SpectralNorm(nn.Linear(num_classes, w_g * w_g * 512))


        self.fc = SpectralNorm(nn.Linear(w_g * w_g * 512, 1))

    def forward(self, x, c):
        # print('x shape', x.size())
        m = x
        m = nn.LeakyReLU(leak)(self.conv1(m))
        m = nn.LeakyReLU(leak)(self.conv2(m))
        m = nn.LeakyReLU(leak)(self.conv3(m))
        m = nn.LeakyReLU(leak)(self.conv4(m))
        m = nn.LeakyReLU(leak)(self.conv5(m))
        m = nn.LeakyReLU(leak)(self.conv6(m))

        self.attention_output = self.att(m)

        m = self.att_post(m, self.attention_output)

        m = nn.LeakyReLU(leak)(self.conv7(m))
        # print('m size', m.size())
        # m = m.view(-1, w_g * w_g * 512)
        m=m.reshape(-1, w_g * w_g * 512)


        return self.fc(m) + torch.bmm(m.view(-1, 1, w_g * w_g * 512), self.embed(c).view(-1, w_g * w_g * 512, 1))


channels = 1
width = 28

loader = torch.utils.data.DataLoader(
    datasets.MNIST(args.dataroot, train=True, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))])),
        batch_size=args.batch_size*batch_size_mult, shuffle=True, num_workers=1, pin_memory=True)


Z_dim = 128
#number of updates to discriminator for every update to generator 
disc_iters = 1

# discriminator = torch.nn.DataParallel(Discriminator()).cuda() # TODO: try out multi-gpu training
# if args.model == 'resnet':
#     discriminator = model_resnet.Discriminator().cuda()
#     generator = model_resnet.Generator(Z_dim).cuda()
# else:

discriminator = Discriminator().cuda()
generator = Generator(Z_dim).cuda()

# if args.load is not None:
#     cp_disc = torch.load(os.path.join(args.checkpoint_dir, 'disc_{}'.format(args.load)))
#     cp_gen = torch.load(os.path.join(args.checkpoint_dir, 'gen_{}'.format(args.load)))
#     discriminator.load_state_dict(cp_disc)
#     generator.load_state_dict(cp_gen)
#     print('Loaded checkpoint (epoch {})'.format(args.load))   


# because the spectral normalization module creates parameters that don't require gradients (u and v), we don't want to 
# optimize these using sgd. We only let the optimizer operate on parameters that _do_ require gradients
# TODO: replace Parameters with buffers, which aren't returned from .parameters() method.
optim_disc = optim.Adam(filter(lambda p: p.requires_grad, discriminator.parameters()), lr=args.lr_disc, betas=(0.0,0.9))
optim_gen  = optim.Adam(filter(lambda p: p.requires_grad, generator.parameters()), lr=args.lr_gen, betas=(0.0,0.9))

# use an exponentially decaying learning rate
scheduler_d = optim.lr_scheduler.ExponentialLR(optim_disc, gamma=0.999)
scheduler_g = optim.lr_scheduler.ExponentialLR(optim_gen, gamma=0.999)

def train(epoch):
    pbar=tqdm(loader)
    loss_D=[]
    loss_G=[]
    for batch_idx, (data, target) in enumerate(pbar):
        if data.size()[0] != args.batch_size*batch_size_mult:
            continue
        data, target = data.cuda(), target.cuda()

        rand_class, rand_c_onehot = make_rand_class()
        samples = data[(target == rand_class).nonzero()].squeeze()
        bsize = samples.size(0)
        data_selected = samples.repeat((args.batch_size // bsize + 1, 1,1,1,1)).view(-1, channels, width, width)[:args.batch_size]

        # update discriminator
        for _ in range(disc_iters):
            z = torch.randn(args.batch_size, Z_dim).cuda()

            optim_disc.zero_grad()
            optim_gen.zero_grad()

            disc_loss = (nn.ReLU()(1.0 - discriminator(data_selected, rand_c_onehot))).mean() + (nn.ReLU()(1.0 + discriminator(generator(z, rand_c_onehot[0]), rand_c_onehot))).mean()

            disc_loss.backward()
            optim_disc.step()

        z = Variable(torch.randn(args.batch_size, Z_dim).cuda())
        rand_class, rand_c_onehot = make_rand_class()
        # update generator
        optim_disc.zero_grad()
        optim_gen.zero_grad()

        gen_loss = -discriminator(generator(z, rand_c_onehot[0]), rand_c_onehot).mean()
        gen_loss.backward()
        optim_gen.step()

        # if batch_idx % 100 == 99:
            # print('disc loss', disc_loss.data[0], 'gen loss', gen_loss.data[0])
            # print('disc loss', disc_loss.item(), 'gen loss', gen_loss.item())
        pbar.set_description(f'[Epoch: {epoch}][disc_loss: {disc_loss.item():.4f}][gen_loss: {gen_loss.item():.4f}]')
        loss_D.append(disc_loss.item())
        loss_G.append(gen_loss.item())
    scheduler_d.step()
    scheduler_g.step()
    return loss_D,loss_G

fixed_z = torch.randn(args.batch_size, Z_dim).cuda()
fixed_labels = torch.randint(0, 10, (100,)).cuda()


def make_rand_class():
    rand_class = np.random.randint(num_classes)
    rand_c_onehot = torch.FloatTensor(args.batch_size, num_classes).cuda()
    rand_c_onehot.zero_()
    rand_c_onehot[:, rand_class] = 1
    return (rand_class, rand_c_onehot)

def evaluate(epoch):
    for fixed_class in range(num_classes):

        fixed_c_onehot = torch.FloatTensor(args.batch_size, num_classes).cuda()
        fixed_c_onehot.zero_()
        fixed_c_onehot[:, fixed_class] = 1
        # print(fixed_c_onehot[0])

        samples = generator(fixed_z, fixed_c_onehot[0]).expand(-1, 3, -1, -1).cpu().detach().numpy()[:64]
        fig = plt.figure(figsize=(8, 8))
        gs = gridspec.GridSpec(8, 8)
        gs.update(wspace=0.05, hspace=0.05)

        for i, sample in enumerate(samples):
            ax = plt.subplot(gs[i])
            plt.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            plt.imshow(sample.transpose((1,2,0)) * 0.5 + 0.5)

        if not os.path.isdir(f'{args.logdir}/logs'):
            os.makedirs(f'{args.logdir}/logs')

        plt.savefig('{}/logs/{}_{}.png'.format(args.logdir,str(epoch).zfill(3),str(fixed_class).zfill(2)), bbox_inches='tight')
        plt.close(fig)


ckpt_dir=os.path.join(args.logdir,'checkpoints')
os.makedirs(ckpt_dir, exist_ok=True)

loss_D=[]
loss_G=[]

for epoch in range(args.epochs+1):
    tmp_D_loss,tmp_G_loss=train(epoch)
    loss_D.extend(tmp_D_loss)
    loss_G.extend(tmp_G_loss)
    if epoch % 5 == 0:
        evaluate(epoch)
        torch.save(discriminator.state_dict(), os.path.join(ckpt_dir, 'disc_{}'.format(epoch)))
        torch.save(generator.state_dict(), os.path.join(ckpt_dir, 'gen_{}'.format(epoch)))

# draw loss curve
plt.rcParams["font.family"]="serif"
plt.figure()
plt.plot(loss_D,label='Discriminator loss')
plt.plot(loss_G,label='Generator loss')
plt.xlabel('iterations')
plt.ylabel('loss')
plt.legend()
plt.savefig('sagan_loss.png')