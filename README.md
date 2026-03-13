# Convolutional Autoencoder for Image Denoising

## AIM
To develop a convolutional autoencoder for image denoising application.

## Problem Statement and Dataset
The MNIST dataset consists of 28x28 grayscale images of handwritten digits (0–9), often used for image processing tasks. The goal of this experiment is image denoising using autoencoders, a neural network designed to learn efficient representations. By introducing noise to images, the model is trained to reconstruct clean versions.

## DESIGN STEPS

## STEP 1:
Load MNIST dataset and convert to tensors.
### STEP 2:
Apply Gaussian noise to images for training.
### STEP 3:
Design encoder-decoder architecture for reconstruction.
### STEP 4:
Use MSE loss to measure reconstruction quality.
### STEP 5:
Train autoencoder using Adam optimizer efficiently.
### STEP 6:
Evaluate model on noisy and clean images.
### STEP 7:
Visualize results comparing original, noisy, denoised versions.
### STEP 8:
Improve performance by tuning hyperparameters carefully.

## PROGRAM
### Name: RANJIT R
### Register Number: 212224240131
```
class DenoisingAutoencoder(nn.Module):
    def __init__(self):
        super(DenoisingAutoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),  # [1,28,28] -> [32,14,14]
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # [32,14,14] -> [64,7,7]
            nn.ReLU(),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # [64,7,7] -> [32,14,14]
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),   # [32,14,14] -> [1,28,28]
            nn.Sigmoid()  # Output between 0 and 1
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
```
```
# Initialize model, loss function and optimizer
model = DenoisingAutoencoder().to(device)
criterion = nn.MSELoss()               # Mean Squared Error for reconstruction
optimizer = optim.Adam(model.parameters(), lr=0.001)
```
```
print("Name: RANJIT R\nReg no: 21222424031")
summary(model, input_size=(1, 28, 28))
```
## OUTPUT

### Model Summary

<img width="1274" height="521" alt="image" src="https://github.com/user-attachments/assets/de2e7037-959d-42fe-8125-806eb41cf928" />



### Original vs Noisy Vs Reconstructed Image

<img width="1388" height="704" alt="image" src="https://github.com/user-attachments/assets/081dfe39-bbba-46ee-9594-17aa503507e1" />


## RESULT
A convolutional autoencoder for image denoising application is developed successfully.
