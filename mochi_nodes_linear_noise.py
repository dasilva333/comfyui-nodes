class ImageMochiLatentVideo:
    def __init__(self):
        self.device = comfy.model_management.get_torch_device()

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "latent": ("LATENT",),
                "length": ("INT", {"default": 25, "min": 7, "max": nodes.MAX_RESOLUTION, "step": 6}),
                "noise_start": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01}),
                "noise_end": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "generate"

    CATEGORY = "latent/mochi"

    def generate(self, latent, length, noise_start=0.1, noise_end=0.4):
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

            # Initialize latent_video by repeating the latent across frames
            latent_video = latent_samples.unsqueeze(2).repeat(1, 1, frames, 1, 1)  # Shape: [batch_size, 12, frames, height, width]

            # Generate random noise tensor
            noise = torch.randn_like(latent_video)

            # Create scaling factors for noise from noise_start to noise_end
            scaling_factors = torch.linspace(noise_start, noise_end, frames, device=device).view(1, 1, frames, 1, 1)

            # Apply scaled noise to latent_video
            latent_video = latent_video + noise * scaling_factors

        elif len(shape) == 5:
            # Shape: [batch_size, channels, frames_in, height, width]
            batch_size, channels, frames_in, height, width = shape

            # The expected channels for video latent is 12
            if channels != 12:
                raise ValueError(f"Expected latent channels to be 12, but got {channels}")

            # Calculate the desired number of frames
            frames = ((length - 1) // 6) + 1

            # Adjust the number of frames if necessary
            if frames_in != frames:
                if frames_in > frames:
                    latent_samples = latent_samples[:, :, :frames, :, :]
                else:
                    # Repeat the last frame to reach the desired number of frames
                    last_frame = latent_samples[:, :, -1:, :, :]
                    repeat_times = frames - frames_in
                    latent_samples = torch.cat([latent_samples, last_frame.repeat(1, 1, repeat_times, 1, 1)], dim=2)

            latent_video = latent_samples  # Use the input latent samples as the base

            # Generate random noise tensor
            noise = torch.randn_like(latent_video)

            # Create scaling factors for noise from noise_start to noise_end
            scaling_factors = torch.linspace(noise_start, noise_end, frames, device=device).view(1, 1, frames, 1, 1)

            # Apply scaled noise to latent_video
            latent_video = latent_video + noise * scaling_factors

        else:
            raise ValueError(f"Expected latent to have 4 or 5 dimensions, but got {len(shape)}")

        # Return the modified latent in the expected format
        return ({"samples": latent_video},)
