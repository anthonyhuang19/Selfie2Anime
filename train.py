import torch 
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from math import log10
from model import *
from utils import *
from dataset import *


import torch
from math import log10
from skimage.metrics import structural_similarity as ssim

# Assuming normalize_image is defined and works with PyTorch tensors

def calculate_psnr(img1, img2):
    if img1.min() < 0 or img1.max() > 1 or img2.min() < 0 or img2.max() > 1:
        img1 = normalize_image(img1)
        img2 = normalize_image(img2)
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100  # PSNR is infinite in this case

    # Compute PSNR using the formula
    return 20 * log10(1.0 / torch.sqrt(mse))  # Max pixel value assumed to be 1.0 (normalized images)



def calculate_ssim(img1, img2):
    if img1.min() < 0 or img1.max() > 1 or img2.min() < 0 or img2.max() > 1:
        img1 = normalize_image(img1)
        img2 = normalize_image(img2)

    # Detach tensors and convert them to numpy arrays for SSIM computation
    img1_np = img1.detach().cpu().numpy()
    img2_np = img2.detach().cpu().numpy()

    # Get the smaller side of the image dimensions to calculate win_size
    min_side = min(img1_np.shape[1], img1_np.shape[2])  # assuming [batch, channels, height, width]

    # Ensure win_size is odd and fits within the image dimensions
    win_size = min(7, min_side) if min_side % 2 == 1 else min(7, min_side - 1)

    # SSIM computation using skimage with the specified win_size
    return ssim(img1_np, img2_np, multichannel=True, data_range=1.0, win_size=win_size)




def normalize_image(img):
    return (img + 1) / 2  
def save_metrics(psnr, ssim, output_file='metrics.txt'):
    with open(output_file, 'a') as f:
        f.write(f"PSNR: {psnr}\n")
        f.write(f"SSIM: {ssim}\n")

def write_list_to_txt(file_name, data_list):
    with open(file_name, 'a') as f:
        for item in data_list:
            f.write(f"{item}\n")  # Write each item on a new line
    print(f"List has been written to {file_name}")

def plot_and_save_losses(losses, save_path='training_losses_plot.png'):
    fig, ax = plt.subplots(figsize=(12, 8))
    plt.plot(losses.T[0], label='Discriminator, X', alpha=0.5)
    plt.plot(losses.T[1], label='Discriminator, Y', alpha=0.5)
    plt.plot(losses.T[2], label='Generators', alpha=0.5)
    plt.title("Training Losses")
    plt.legend()
    plt.savefig(save_path)  # Save as a PNG file
    print(f"Plot saved as '{save_path}'")

class CycleGAN:
    def __init__(self, g_conv_dim=64, d_conv_dim=64, n_res_block=6):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.Generator_x2y = Generator(conv_dim=g_conv_dim, n_res_block=n_res_block).to(self.device)
        self.Generator_y2x = Generator(conv_dim=g_conv_dim, n_res_block=n_res_block).to(self.device)
        self.Discriminator_x = Discriminator(conv_dim=d_conv_dim).to(self.device)
        self.Discriminator_y = Discriminator(conv_dim=d_conv_dim).to(self.device)

        print("Initializing Model Generator and Discriminator")

    def load_model(self, filename):
        save_filename = os.path.splitext(os.path.basename(filename))[0] + '.pt'
        return torch.load(save_filename)

    def real_mse_loss_function(self, D_out):
        return torch.mean((D_out - 1) ** 2)

    def fake_mse_loss_function(self, D_out):
        return torch.mean((D_out) ** 2)

    def cycle_consistency_loss(self, real_img, reconstructed_img, lambda_weight):
        reconstr_loss = torch.mean(torch.abs(real_img - reconstructed_img))
        return lambda_weight * reconstr_loss
    
    def train_generator(self, optimizer, image_x, image_y):
        optimizer["g_optim"].zero_grad()
        fake_images_x = self.Generator_y2x(image_y)
        d_real_x = self.Discriminator_x(fake_images_x)
        g_YtoX_loss = self.real_mse_loss_function(d_real_x)
        reconstructed_y = self.Generator_x2y(fake_images_x)
        reconstructed_y_loss = self.cycle_consistency_loss(image_y, reconstructed_y, lambda_weight=10)
        fake_images_y = self.Generator_x2y(image_x)
        d_real_y = self.Discriminator_y(fake_images_y)
        g_XtoY_loss = self.real_mse_loss_function(d_real_y)
        reconstructed_x = self.Generator_y2x(fake_images_y)
        reconstructed_x_loss = self.cycle_consistency_loss(image_x, reconstructed_x, lambda_weight=10)
        g_total_loss = g_YtoX_loss + g_XtoY_loss + reconstructed_y_loss + reconstructed_x_loss
        g_total_loss.backward()
        optimizer["g_optim"].step()
        return g_total_loss.item(),fake_images_x,fake_images_y

    def train_discriminator(self, optimizers, real_images_x, real_images_y):
        optimizers["d_x_optim"].zero_grad()
        real_d_x = self.Discriminator_x(real_images_x)
        real_loss_x = self.real_mse_loss_function(real_d_x)
        fake_images_x = self.Generator_y2x(real_images_y)
        fake_d_x = self.Discriminator_x(fake_images_x)
        fake_loss_x = self.fake_mse_loss_function(fake_d_x)
        d_x_loss = real_loss_x + fake_loss_x
        d_x_loss.backward()
        optimizers["d_x_optim"].step()

        optimizers["d_y_optim"].zero_grad()
        real_d_y = self.Discriminator_y(real_images_y)
        real_loss_y = self.real_mse_loss_function(real_d_y)
        fake_images_y = self.Generator_x2y(real_images_x)
        fake_d_y = self.Discriminator_y(fake_images_y)
        fake_loss_y = self.fake_mse_loss_function(fake_d_y)
        d_y_loss = real_loss_y + fake_loss_y
        d_y_loss.backward()
        optimizers["d_y_optim"].step()

        return d_x_loss.item(), d_y_loss.item() 
    
    def train(self, optimizers, data_loader_x, data_loader_y, print_every=10, sample_every=100):
        losses = []
        min_g_loss = np.Inf

        print(f'Running on {self.device}')

        calculate_psnr1 =[]
        calculate_psnr2 =[]
        calculate_ssim1 =[]
        calculate_ssim2 =[]
        for epoch in range(EPOCH):
            epoch_losses = []
            # Wrap both data_loader_x and data_loader_y with tqdm to track progress
            for batch_images_x, batch_images_y in zip(tqdm(data_loader_x, desc='Batch X', leave=False), tqdm(data_loader_y, desc='Batch Y', leave=False)):
                batch_images_x, batch_images_y = batch_images_x.to(self.device), batch_images_y.to(self.device)
                g_loss,fake_images_x,fake_images_y = self.train_generator(optimizers, batch_images_x, batch_images_y)
                d_x_loss, d_y_loss = self.train_discriminator(optimizers, batch_images_x, batch_images_y)
                epoch_losses.append((d_x_loss, d_y_loss, g_loss))

                psnr1 = calculate_psnr(fake_images_x,batch_images_x)
                psnr2 = calculate_psnr(fake_images_y,batch_images_y)
                ssim1 = calculate_ssim(fake_images_x,fake_images_y)
                ssim2 = calculate_ssim(fake_images_y,batch_images_y)
                calculate_psnr1.append(psnr1)
                calculate_psnr2.append(psnr2)
                calculate_ssim1.append(ssim1)
                calculate_ssim2.append(ssim2)
            # Print and save models if needed
            if epoch % print_every == 0:
                d_x_loss, d_y_loss, g_loss = np.mean(epoch_losses, axis=0)
                losses.append((d_x_loss, d_y_loss, g_loss))
                print(f'Epoch [{epoch:5d}/{EPOCH:5d}] | d_X_loss: {d_x_loss:6.4f} | d_Y_loss: {d_y_loss:6.4f} | g_loss: {g_loss:6.4f}')
                write_list_to_txt('output/log/output.txt', losses)
            if g_loss < min_g_loss:
                min_g_loss = g_loss
                # Save the model parameters in .pth format
                torch.save(self.Generator_x2y.state_dict(), "G_X2Y.pth")
                torch.save(self.Generator_y2x.state_dict(), "G_Y2X.pth")
                torch.save(self.Discriminator_x.state_dict(), "D_X.pth")
                torch.save(self.Discriminator_y.state_dict(), "D_Y.pth")
                print("Models Saved as .pth")
       
            avg_psnr1 = sum(calculate_psnr1)/len(calculate_psnr1)
            avg_ssim1 = sum(calculate_ssim1)/len(calculate_ssim1)

            avg_psnr2 = sum(calculate_psnr2)/len(calculate_psnr2)
            avg_ssim2 = sum(calculate_ssim2)/len(calculate_ssim2)

          
            print(f"\nAverage PSNR: {avg_psnr1}")
            print(f"Average SSIM: {avg_ssim1}")
            print(f"\nAverage PSNR: {avg_psnr2}")
            print(f"Average SSIM: {avg_ssim2}")
        return losses

def main():
    #Training dataset 
    print('Processing Dataset')
    x_dataset = Dataset('trainA')
    y_dataset = Dataset('trainB')
    x_data = DataLoader(x_dataset, batch_size=Batch_size, shuffle=False)  
    y_data = DataLoader(y_dataset, batch_size=Batch_size, shuffle=False)  
    print('Ending of Processing Dataset')
    #Model
    model = CycleGAN()
    parameter = list(model.Generator_x2y.parameters()) + list(model.Generator_y2x.parameters())
    optimizers = {
        "g_optim": optim.Adam(parameter, Learning_Rate, [Beta1, Beta2]),
        "d_x_optim": optim.Adam(model.Discriminator_x.parameters(), Learning_Rate, [Beta1, Beta2]),
        "d_y_optim": optim.Adam(model.Discriminator_y.parameters(), Learning_Rate, [Beta1, Beta2])
    }

    losses = model.train(optimizers=optimizers,data_loader_x=x_data,data_loader_y=y_data)
    print("Finisihed training model")
    

if __name__ == "__main__":
    main()



