import os
import random
import numpy as np
import torch.nn as nn
import torchsummary
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import matplotlib.pyplot as plt  # グラフ作成用ライブラリー

import pandas as pd
from Dataset import Dataset

# Initial setting
workers = 2
batch_size = 49
nz = 100
nch_g = 64
nch_d = 64
n_epoch = 50
lr = 0.0002
beta1 = 0.5
outf = './result_cgan'
display_interval = 100
plt.rcParams['figure.figsize'] = 10, 6  # グラフの大きさ指定

n_class = 149

try:
    os.makedirs(outf)
except OSError:
    pass

random.seed(59)
np.random.seed(59)
torch.manual_seed(59)

# 重みとバイアスを正規分布に
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:            
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:        
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:    
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class Discriminator(nn.Module):
    def __init__(self, nch=3, nch_d=64):
        super(Discriminator, self).__init__()
        self.layers = nn.ModuleDict(
            {
            'layer0': nn.Sequential(  # (3, 64, 64) -> (64, 32, 32)
                nn.Conv2d(nch, nch_d, 4, 2, 1),
                nn.LeakyReLU(negative_slope=0.2)    
                ),
            'layer1': nn.Sequential(  # (64, 32, 32) -> (128, 16, 16)
                nn.Conv2d(nch_d, nch_d * 2, 4, 2, 1),
                nn.BatchNorm2d(nch_d * 2),
                nn.LeakyReLU(negative_slope=0.2)
                ),
            'layer2': nn.Sequential(  # (128, 16, 16) -> (256, 8, 8)
                nn.Conv2d(nch_d * 2, nch_d * 4, 4, 2, 1),
                nn.BatchNorm2d(nch_d * 4),
                nn.LeakyReLU(negative_slope=0.2)
                ),
            'layer3': nn.Sequential(  # (256, 8, 8) -> (512, 4, 4)
                nn.Conv2d(nch_d * 4, nch_d * 8, 4, 2, 1),
                nn.BatchNorm2d(nch_d * 8),
                nn.LeakyReLU(negative_slope=0.2)
                ),
            'layer4':
                nn.Conv2d(nch_d * 8, 1, 4, 1, 0)  # (512, 4, 4) -> (1, 1, 1)
            }
        )

    def forward(self, x):
        for layer in self.layers.values():  
            x = layer(x)
        return x.squeeze()    


class Generator(nn.Module):
    def __init__(self, nz=100, nch_g=64, nch=3):
        super(Generator, self).__init__()
        self.layers = nn.ModuleDict(
            {
            'layer0': nn.Sequential(  # (100, 1, 1) -> (512, 4, 4)
                nn.ConvTranspose2d(nz, nch_g * 8, 4, 1, 0),     
                nn.BatchNorm2d(nch_g * 8),                      
                nn.ReLU()                                       
                ),
            'layer1': nn.Sequential(  # (512, 4, 4) -> (256, 8, 8)
                nn.ConvTranspose2d(nch_g * 8, nch_g * 4, 4, 2, 1),
                nn.BatchNorm2d(nch_g * 4),
                nn.ReLU()
                ),
            'layer2': nn.Sequential(  # (256, 8, 8) -> (128, 16, 16)
                nn.ConvTranspose2d(nch_g * 4, nch_g * 2, 4, 2, 1),
                nn.BatchNorm2d(nch_g * 2),
                nn.ReLU()
                ),
            'layer3': nn.Sequential(  # (128, 16, 16) -> (64, 32, 32)
                nn.ConvTranspose2d(nch_g * 2, nch_g, 4, 2, 1),
                nn.BatchNorm2d(nch_g),
                nn.ReLU()
                ),
            'layer4': nn.Sequential(   # (64, 32, 32) -> (3, 64, 64)
                nn.ConvTranspose2d(nch_g, nch, 4, 2, 1),
                nn.Tanh()
                )
            }
        )

    def forward(self, z):
        for layer in self.layers.values():
            z = layer(z)
        return z



# def onehot_encode(label, device, n_class=7):  
#     # ラベルをOne-Hoe形式に変換
#     eye = torch.eye(n_class, device=device)
#     # ランダムベクトルあるいは画像と連結するために(B, c_class, 1, 1)のTensorにして戻す
#     return eye[label].view(-1, n_class, 1, 1)

def concat_image_label(image, label, device, n_class=7):
    # 画像とラベルを連結する
    N, B, H, W = image.shape    # 画像Tensorの大きさを取得
    label = label.expand(N, n_class, H, W)  # ラベルを画像サイズに拡大
    print(f'label(after expand): {label.shape}')
    # print(B,H,W) #3,256,256
    return torch.cat((image, label), dim=1)    # 画像とラベルをチャネル方向（dim=1）で連結


def concat_noise_label(noise, label, device):
    # ランダムベクトルとラベルを連結する
    print(noise.shape)
    print(label.shape)
    return torch.cat((noise, label), dim=0)  # ランダムベクトルとラベルを連結


# def main():
#     df_test = pd.read_csv('/home/ytakeda/fashion_images/keep_aspect_resized/test_images.csv')
#     df_train = pd.read_csv('/home/ytakeda/fashion_images/keep_aspect_resized/train_images.csv', encoding="shift-jis") # trainだけなぜかShift-JIS
#     df = pd.concat([df_test, df_train], ignore_index=True)
#     dataset = Dataset(df, transform=transforms.Compose([
#                           transforms.RandomResizedCrop(64, scale=(0.9, 1.0), ratio=(1., 1.)),
#                           transforms.RandomHorizontalFlip(),
#                           transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.05),
#                           transforms.ToTensor(),
#                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#                           ])
#                         )
#     dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
#                                          shuffle=True, num_workers=int(workers))
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     print(concat_image_label(image, label, device, n_class))

#     exit()


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    # device = 'cpu'


    # データ
    df_test = pd.read_csv('/home/ytakeda/fashion_images/keep_aspect_resized/test_images.csv')
    df_train = pd.read_csv('/home/ytakeda/fashion_images/keep_aspect_resized/train_images.csv', encoding="shift-jis") # trainだけなぜかShift-JIS
    df = pd.concat([df_test, df_train], ignore_index=True)
    dataset = Dataset(df, transform=transforms.Compose([
                          transforms.RandomResizedCrop(64, scale=(0.9, 1.0), ratio=(1., 1.)),
                          transforms.RandomHorizontalFlip(),
                          transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.05),
                          transforms.ToTensor(),
                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                          ])
                        )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=int(workers))


    # モデル
    netD = Discriminator(nch=3+n_class, nch_d=nch_d).to(device)
    netD.apply(weights_init)
    # print(3+n_class)
    # print(nch_d)
    # torchsummary.summary(netD, (3+n_class, nch_d, nch_d), device=device.type)
    print(netD) #print()なら出力できるが、summary()だとエラーになる

    netG = Generator(nz=nz+n_class, nch_g=nch_g).to(device)
    netG.apply(weights_init)
    # torchsummary.summary(netG, (nz+n_class, 1, 1))
    print(netG)


    # 最適化
    criterion = nn.MSELoss()  #BCELoss()?
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999), weight_decay=1e-5)
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999), weight_decay=1e-5)


    # ノイズ
    fixed_noise = torch.randn(batch_size, nz, 1, 1, device=device)

    fixed_label = np.zeros((batch_size, nz))
    fixed_label[:, 0:nz] = [i for i in range(nz)]
    fixed_label = fixed_label[..., np.newaxis, np.newaxis]

    fixed_label = torch.tensor(fixed_label, dtype=torch.long, device=device)

    fixed_noise_label = concat_noise_label(fixed_noise, fixed_label, device)
    Loss_D_list, Loss_G_list = [], []  # グラフ作成用リスト初期化


    # 訓練
    for epoch in range(n_epoch):
        print(f'Epoch {epoch+1}/{n_epoch}')
        for itr, data in enumerate(dataloader):



            real_image = torch.FloatTensor(data[0]).to(device)  # 本物画像
            # print(f'image: {real_image.shape}')
            tag = np.zeros((batch_size, n_class))
            tag[:, 0:n_class] = data[1]
            real_label = torch.LongTensor(tag).to(device)  # 本物画像のラベル
            print(f'label: {real_label.shape}')
            real_label = real_label.view(batch_size, n_class, 1, 1).to(device)
            print(f'label(after view): {real_label.shape}')
            real_image_label = concat_image_label(real_image, real_label, device, n_class)   # 本物画像とラベルを連結



            sample_size = real_image.size(0)  # 画像枚数
            noise = torch.randn(sample_size, nz, 1, 1, device=device)  # ランダムベクトル生成（ノイズ）
            # noise = torch.randn(sample_size, device=device)  # ランダムベクトル生成（ノイズ）
            # fake_label = torch.randint(256, (sample_size,), dtype=torch.long, device=device)  # 偽物画像のラベル
            fake_label = torch.randint(low=1, high=2, size=(sample_size, nz, 1, 1), dtype=torch.long, device=device)

            fake_noise_label = concat_noise_label(noise, fake_label, device)  # ランダムベクトルとラベルを連結

            real_target = torch.full((sample_size,), 1., device=device)  
            fake_target = torch.full((sample_size,), 0., device=device)  

            #----------  Update Discriminator  -----------
            netD.zero_grad()

            # output = netD(real_image_label.expand(256,1,256,256))
            output = netD(real_image_label.to(device))
            errD_real = criterion(output, real_target)
            D_x = output.mean().item()

            fake_image = netG(fake_noise_label)  # Generatorが生成した偽物画像
            fake_image_label = concat_image_label(fake_image, fake_label, device)   # 偽物画像とラベルを連結

            output = netD(fake_image_label.detach())
            errD_fake = criterion(output, fake_target)
            D_G_z1 = output.mean().item()

            errD = errD_real + errD_fake
            errD.backward()
            optimizerD.step()

            #----------  Update Generator  -------------
            netG.zero_grad()
        
            output = netD(fake_image_label)
            errG = criterion(output, real_target)  
            errG.backward()
            D_G_z2 = output.mean().item()
        
            optimizerG.step()

            if itr % display_interval == 0:
                print('[{}/{}][{}/{}] Loss_D: {:.3f} Loss_G: {:.3f} D(x): {:.3f} D(G(z)): {:.3f}/{:.3f}'
                      .format(epoch + 1, n_epoch,
                              itr + 1, len(dataloader),
                              errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
            
                Loss_D_list.append(errD.item())  # Loss_Dデータ蓄積 (グラフ用)
                Loss_G_list.append(errG.item())  # Loss_Gデータ蓄積 (グラフ用)
            
            if epoch == 0 and itr == 0:
                vutils.save_image(real_image, '{}/real_samples.png'.format(outf),
                                  normalize=True, nrow=7)

        # --------- save fake image  ----------
        fake_image = netG(fixed_noise_label)   
        vutils.save_image(fake_image.detach(), '{}/fake_samples_epoch_{:03d}.png'.format(outf, epoch + 1),
                         normalize=True, nrow=7)

        # ---------  save model  ----------
        if (epoch + 1) % 10 == 0:  
            torch.save(netG.state_dict(), '{}/netG_epoch_{}.pth'.format(outf, epoch + 1))
            torch.save(netD.state_dict(), '{}/netD_epoch_{}.pth'.format(outf, epoch + 1))


    # グラフ作成
    plt.figure()    
    plt.plot(range(len(Loss_D_list)), Loss_D_list, color='blue', linestyle='-', label='Loss_D')
    plt.plot(range(len(Loss_G_list)), Loss_G_list, color='red', linestyle='-', label='Loss_G')
    plt.legend()
    plt.xlabel('iter (*100)')
    plt.ylabel('loss')
    plt.title('Loss_D and Loss_G')
    plt.grid()
    plt.savefig('Loss_graph.png')          



if __name__ == '__main__':
    main()
