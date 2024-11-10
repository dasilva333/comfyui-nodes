class ImageMochiLatentVideo:
    def __init__(self):
        self.device = comfy.model_management.get_torch_device()

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "latent": ("LATENT",),
                "length": ("INT", {"default":25, "min":7, "max": nodes.MAX_RESOLUTION, "step":6}),
            }
        }
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "generate"

    CATEGORY = "latent/mochi"

    def generate(self, latent, length):
        device = self.device

        # Extract the latent samples tensor from the input
        latent_samples = latent["samples"].to(device)

        # Check the shape of latent_samples
        shape = latent_samples.shape

        if len(shape) == 4:
            # Shape: [batch_size, channels, height, width]
            batch_size, channels, height, width = shape

            # The expected channels for video latent is 12
            if channels == 4:
                # Expand channels from 4 to 12 by repeating the latent along the channel dimension
                latent_samples = latent_samples.repeat(1, 3, 1, 1)  # Now shape: [batch_size, 12, height, width]
                channels = 12
            elif channels != 12:
                raise ValueError(f"Expected latent channels to be 4 or 12, but got {channels}")

            # Calculate the number of frames based on the provided length
            frames = ((length - 1) // 6) + 1

            # Initialize the latent video tensor with zeros
            latent_video = torch.zeros([batch_size, 12, frames, height, width], device=device)

            # Set the first frame to the input latent
            latent_video[:, :, 0, :, :] = latent_samples

            # The rest of the frames remain zeros

            # Optionally, add random noise to the rest of the frames to encourage variation
            # Uncomment the following line if desired:
            # latent_video[:, :, 1:, :, :] = torch.randn_like(latent_video[:, :, 1:, :, :])

            # Use latent_video as the new latent_samples
            latent_samples = latent_video

        elif len(shape) == 5:
            # Shape: [batch_size, channels, frames_in, height, width]
            batch_size, channels, frames_in, height, width = shape

            # The expected channels for video latent is 12
            if channels != 12:
                raise ValueError(f"Expected latent channels to be 12, but got {channels}")

            # Calculate the desired number of frames
            frames = ((length - 1) // 6) + 1

            # Initialize the latent video tensor with zeros
            latent_video = torch.zeros([batch_size, channels, frames, height, width], device=device)

            # Set the first frame to the first frame of the input latent video
            latent_video[:, :, 0, :, :] = latent_samples[:, :, 0, :, :]

            # The rest of the frames remain zeros

            # Use latent_video as the new latent_samples
            latent_samples = latent_video

        else:
            raise ValueError(f"Expected latent to have 4 or 5 dimensions, but got {len(shape)}")

        # Return the modified latent in the expected format
        return ({"samples": latent_samples},)