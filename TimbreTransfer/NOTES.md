This contains the implementation of Timbre Transfer using 2 different methods:

- Denoising Diffusion Implicit Models
- VAE-GAN

## 1. Timbre Transfer using Image-to-Image Denoising Diffusion Implicit Models (DiffTransfer)

The code for this method is available in the folder `DiffTransfer`. The README file in the folder contains the details of the implementation.

> Problems: The dataset used in this model `StarNet` is not applicable for the exact use of VoiceForRobot that we are aiming for.
> In `StarNet` database and the implementation of the model, the training process using the pairs of audios (refers to 2 distinct timbre), to learning the style and reproduce it. However, these pairs of audios requires the same content and melody, which is not possible in our voice of robot sound.

## 2. Timbre Transfer using VAE-GAN

The code for this method is available in the folder `tt-vae-gan`. The README file in the folder contains the details of the implementation.

> Datasets: `Flickr8k` (for human speakers) and `URMP` (for musical instruments)
> Traininng Process:
>
> - The training procedure for the `VAE-GAN model` and the `Vocoder` (WaveNet) has been trained seperately. Therefore we have different pre-trained model for these 2 parts. See `models/pre-trained/tt-vae-gan` folder.

# Model Architecture for the Timbre Transfer using VAE-GAN for VoiceForRobot

The model architecture for the Timbre Transfer using VAE-GAN is as follows:

- Residual Block
- Encoder - Multiple Encoder Blocks
