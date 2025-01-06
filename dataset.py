import torch
import os
from torchvision import transforms
from PIL import Image
from torch.utils.data import DataLoader

class Dataset(torch.utils.data.Dataset):
    def __init__(self, img_dir):
        image_directory = 'dataset/' + img_dir
        path_list = os.listdir(image_directory)
        abspath = os.path.abspath(image_directory)

        self.image_dir = image_directory
        self.image_list = sorted([os.path.join(abspath, path) for path in path_list])  # Sort image paths
        
        self.transform = transforms.Compose([
            transforms.Resize(128),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        
    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, idx):
        path = self.image_list[idx]
        img = Image.open(path).convert('RGB')
        img_tensor = self.transform(img)
        return img_tensor

def save_image(tensor, path, unnormalize=False):
    if unnormalize:
        tensor = tensor * 0.5 + 0.5

    tensor = tensor.mul(255).clamp(0, 255).byte()
    image = tensor.permute(1, 2, 0).cpu().numpy()
    image = Image.fromarray(image)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    image.save(path)

# def main():
#     x = Dataset('trainA')  
#     y = Dataset('trainB')
#     x_data = DataLoader(x, batch_size=1, shuffle=False)  # shuffle=False to maintain the order
#     y_data = DataLoader(y, batch_size=1, shuffle=False)  # shuffle=False to maintain the order

#     output_dir_x = 'x_data'
#     os.makedirs(output_dir_x, exist_ok=True)

#     output_dir_y = 'y_data'
#     os.makedirs(output_dir_y, exist_ok=True)

#     for i, (x_tensor, y_tensor) in enumerate(zip(x_data, y_data)):
#         print(f"Processing pair {i + 1}")
#         save_image(x_tensor[0], os.path.join(output_dir_x, f'output_image_{i+1}.png'), unnormalize=True)
#         save_image(y_tensor[0], os.path.join(output_dir_y, f'output_image_{i+1}.png'), unnormalize=True)

# if __name__ == "__main__":
#     main()
