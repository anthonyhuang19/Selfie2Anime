import torch
import numpy as np
import os
from PIL import Image
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from model import *
from train import *
from dataset import Dataset  # Replace with your actual dataset class

class CycleGANEvaluator:
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.scaler = GradScaler()  # Initialize the gradient scaler for mixed precision

    def load_models(self, generator_x2y_path, generator_y2x_path):
        """Load pre-trained generator models."""
        # Load state_dicts for the generators
        self.model.Generator_x2y.load_state_dict(torch.load(generator_x2y_path))
        self.model.Generator_y2x.load_state_dict(torch.load(generator_y2x_path))

        # Move model to the correct device (GPU or CPU)
        self.model.Generator_x2y.to(self.device)
        self.model.Generator_y2x.to(self.device)

    def evaluate(self, data_loader_x, data_loader_y, output_dir='generated_images'):
        os.makedirs(output_dir, exist_ok=True)  # Ensure the output directory exists

        # Open file to write the log of generated image filenames
        with open('generated_images_log.txt', 'w') as log_file:
            log_file.write("Generated images log:\n")

            for i, (image_x, image_y) in enumerate(zip(data_loader_x, data_loader_y)):
                image_x = image_x.to(self.device)
                image_y = image_y.to(self.device)

                with autocast():  # Use mixed precision
                    # Generate fake images from the generators
                    fake_y = self.model.Generator_x2y(image_x)
                    fake_x = self.model.Generator_y2x(image_y)

                    # Process each image in the batch individually
                    for j in range(fake_y.size(0)):  # Loop over each image in the batch
                        # Convert images from [-1, 1] to [0, 1]
                        fake_y_np = (fake_y[j].detach().cpu().numpy() + 1) / 2
                        fake_x_np = (fake_x[j].detach().cpu().numpy() + 1) / 2

                        # Save the generated images
                        fake_y_image = self._convert_to_image(fake_y_np)
                        fake_x_image = self._convert_to_image(fake_x_np)

                        fake_y_image.save(os.path.join(output_dir, f"fake_y_{i}_{j}.png"))
                        fake_x_image.save(os.path.join(output_dir, f"fake_x_{i}_{j}.png"))

                        # Log the image filenames
                        log_file.write(f"fake_y_{i}_{j}.png\n")
                        log_file.write(f"fake_x_{i}_{j}.png\n")

                # After processing each batch, clear memory
                torch.cuda.empty_cache()

    def _convert_to_image(self, img_np):
        """Convert numpy array to image."""
        if len(img_np.shape) == 2:  # Grayscale image (height, width)
            return Image.fromarray((img_np * 255).astype(np.uint8))
        elif len(img_np.shape) == 3 and img_np.shape[0] == 1:  # Single channel (height, width, 1)
            # If the image has one channel, remove the channel dimension
            return Image.fromarray((img_np.squeeze() * 255).astype(np.uint8))
        elif len(img_np.shape) == 3 and img_np.shape[0] == 3:  # RGB image (3, height, width)
            # Change the shape from (3, height, width) to (height, width, 3)
            img_np = np.transpose(img_np, (1, 2, 0))  # Transpose to (height, width, channels)
            return Image.fromarray((img_np * 255).astype(np.uint8))
        else:
            raise ValueError(f"Unexpected image shape: {img_np.shape}")

# Main function to evaluate the model
def main():
    # Specify the device ('cuda' for GPU or 'cpu' for CPU)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load pre-trained model and datasets
    model = CycleGAN()  # Assuming CycleGAN is the class for your model
    evaluator = CycleGANEvaluator(model, device)

    # Load generator model weights
    evaluator.load_models("model/G_X2Y.pth", "model/G_Y2X.pth")

    # Prepare test data loaders
    test_x_dataset = Dataset('testA')  # Replace with actual dataset class
    test_y_dataset = Dataset('testB')
    test_x_loader = DataLoader(test_x_dataset, batch_size=4, shuffle=False)
    test_y_loader = DataLoader(test_y_dataset, batch_size=4, shuffle=False)

    # Evaluate the model
    evaluator.evaluate(test_x_loader, test_y_loader, output_dir="generated_images")


if __name__ == "__main__":
    main()
