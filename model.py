import torch 
import torch.nn as nn
import torch.nn.functional as F

"""
First, i will build a discriminator class , this process for y -> x , this process using convolution and leaky relu 
"""
class Discriminator(nn.Module):
    def __init__(self, conv_dim = 32):
        super(Discriminator,self).__init__()
        self.first_layer = nn.Sequential(
            nn.Conv2d(3,conv_dim,kernel_size=4,stride=2,padding=1),
            nn.LeakyReLU(negative_slope=0.2,inplace=True),
        )
        self.second_layer = nn.Sequential(
            nn.Conv2d(conv_dim,conv_dim *2 , kernel_size=4,stride=2,padding=1),
            nn.InstanceNorm2d(conv_dim * 2),
            nn.LeakyReLU(0.2,inplace=True),
        )
        self.thrid_layer = nn.Sequential(
            nn.Conv2d(conv_dim *2, conv_dim * 4,kernel_size=4,stride=2,padding=1),
            nn.InstanceNorm2d(conv_dim * 4),
            nn.LeakyReLU(0.2,True),
        )
        self.fourth_layer = nn.Sequential(
            nn.Conv2d(conv_dim *4, conv_dim * 8,kernel_size=4,padding=1),
            nn.InstanceNorm2d(conv_dim * 8),
            nn.LeakyReLU(0.2,True),
        )
        self.final_layer = nn.Sequential(nn.Conv2d(conv_dim * 8, 1, 4,padding=1))
    def forward(self,x):
        output = self.first_layer(x)
        output = self.second_layer(output)
        output = self.thrid_layer(output)
        output = self.fourth_layer(output)
        output = self.final_layer(output)
        output = F.avg_pool2d(output, output.size()[2:]) # average calculation for matrix and then flattening one vector into one 
        output = torch.flatten(output, 1)
        return output

#By using resnet for 
class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels)

        self.matching_conv = nn.Conv2d(in_channels, in_channels, 1, stride=1, padding=0)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + x
        out = self.relu(out)

        return out
"""
Using unet concept downsampling -> skip layer -> up sampling and convert into real size
"""
class Generator(nn.Module):
    def __init__(self, conv_dim=64, n_res_block=9):
        super(Generator, self).__init__()

        self.main = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(3, conv_dim, 7),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(conv_dim, conv_dim*2, 3, stride=2, padding=1),
            nn.InstanceNorm2d(conv_dim*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(conv_dim*2, conv_dim*4, 3, stride=2, padding=1),
            nn.InstanceNorm2d(conv_dim*4),
            nn.ReLU(inplace=True),
            ResidualBlock(conv_dim*4),
            ResidualBlock(conv_dim*4),
            ResidualBlock(conv_dim*4),
            ResidualBlock(conv_dim*4),
            ResidualBlock(conv_dim*4),
            ResidualBlock(conv_dim*4),
            ResidualBlock(conv_dim*4),
            ResidualBlock(conv_dim*4),
            ResidualBlock(conv_dim*4),
            nn.ConvTranspose2d(conv_dim*4, conv_dim*2, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(conv_dim*2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(conv_dim*2, conv_dim, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(conv_dim),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(3),
            nn.Conv2d(conv_dim, 3, 7),
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)
# # Testing 
# x = torch.randn((8,3,128,128))
# model =Generator()
# output = model(x)
# print(output.shape) #torch.Size([4, 1, 14, 14])




################################################################################################
#TRAINING & Testing
################################################################################################
