import math
import torch
from PIL import Image
import struct
import numpy as np
from comfy.cli_args import args, LatentPreviewMethod
from comfy.taesd.taesd import TAESD
import comfy.model_management
import folder_paths
import comfy.utils
import logging

MAX_PREVIEW_RESOLUTION = args.preview_size

# Compatibility handling for Pillow's resampling filters
try:
    ANTIALIAS = Image.Resampling.LANCZOS
except AttributeError:
    # For older Pillow versions that still support Image.ANTIALIAS
    ANTIALIAS = Image.ANTIALIAS

def preview_to_image(latent_image):
    """
    Converts a latent tensor to a PIL Image.

    Args:
        latent_image (torch.Tensor): The latent image tensor.

    Returns:
        PIL.Image.Image: The converted PIL Image.
    """
    latents_ubyte = (
        ((latent_image + 1.0) / 2.0)  # Change scale from -1..1 to 0..1
        .clamp(0, 1)
        .mul(0xFF)  # Scale to 0..255
        .to(
            device="cpu",
            dtype=torch.uint8,
            non_blocking=comfy.model_management.device_supports_non_blocking(latent_image.device)
        )
    )

    return Image.fromarray(latents_ubyte.numpy())

class LatentPreviewer:
    """
    Base class for latent image previewers.
    """
    def decode_latent_to_preview(self, x0):
        """
        Decodes latent tensors to preview images.

        Args:
            x0 (torch.Tensor): The latent tensors.

        Returns:
            PIL.Image.Image: The preview image.
        """
        pass

    def decode_latent_to_preview_image(self, preview_format, x0):
        """
        Decodes latent tensors and prepares the preview image.

        Args:
            preview_format (str): The image format ('JPEG' or 'PNG').
            x0 (torch.Tensor): The latent tensors.

        Returns:
            tuple: A tuple containing the format, the preview image, and the maximum resolution.
        """
        preview_image = self.decode_latent_to_preview(x0)
        if preview_image is not None:
            # Optionally resize the grid image to fit within MAX_PREVIEW_RESOLUTION
            preview_image.thumbnail((MAX_PREVIEW_RESOLUTION, MAX_PREVIEW_RESOLUTION), ANTIALIAS)
        return (preview_format, preview_image, MAX_PREVIEW_RESOLUTION)

    def create_image_grid(self, images, max_columns=4):
        """
        Creates a grid image from a list of PIL Images.

        Args:
            images (list of PIL.Image.Image): The list of images to include in the grid.
            max_columns (int, optional): Maximum number of columns in the grid. Defaults to 4.

        Returns:
            PIL.Image.Image: The composited grid image.
        """
        if not images:
            logging.warning("No images provided for grid creation.")
            return None

        batch_size = len(images)
        grid_columns = min(math.ceil(math.sqrt(batch_size)), max_columns)
        grid_rows = math.ceil(batch_size / grid_columns)

        # Resize images to have the same dimensions
        min_width = min(img.size[0] for img in images)
        min_height = min(img.size[1] for img in images)
        resized_images = [img.resize((min_width, min_height), ANTIALIAS) for img in images]

        grid_width = min_width * grid_columns
        grid_height = min_height * grid_rows
        grid_img = Image.new('RGB', (grid_width, grid_height))

        for idx, img in enumerate(resized_images):
            row = idx // grid_columns
            col = idx % grid_columns
            grid_img.paste(img, (col * min_width, row * min_height))

        return grid_img

class TAESDPreviewerImpl(LatentPreviewer):
    """
    Previewer implementation using TAESD decoding.
    """
    def __init__(self, taesd):
        """
        Initializes the TAESDPreviewerImpl.

        Args:
            taesd (TAESD): The TAESD decoder instance.
        """
        self.taesd = taesd

    def decode_latent_to_preview(self, x0):
        """
        Decodes all latent tensors in the batch and creates a grid image.

        Args:
            x0 (torch.Tensor): The latent tensors.

        Returns:
            PIL.Image.Image: The composited grid image.
        """
        if not hasattr(self.taesd, 'decode'):
            logging.error("TAESD instance does not have a 'decode' method.")
            return None

        # Decode all latents in the batch
        try:
            decoded_images = self.taesd.decode(x0)  # Assuming taesd.decode can handle batch inputs
        except Exception as e:
            logging.error(f"Error decoding latents with TAESD: {e}")
            return None

        # Convert decoded tensors to PIL Images
        pil_images = []
        for img_tensor in decoded_images:
            if img_tensor is None:
                logging.warning("Decoded image tensor is None.")
                continue
            img = img_tensor.movedim(0, 2)  # Move channel dimension to last
            pil_image = preview_to_image(img)
            pil_images.append(pil_image)

        # Create a grid from the PIL Images
        grid_image = self.create_image_grid(pil_images)

        return grid_image

class Latent2RGBPreviewer(LatentPreviewer):
    """
    Previewer implementation using Latent2RGB decoding.
    """
    def __init__(self, latent_rgb_factors, latent_rgb_factors_bias=None):
        """
        Initializes the Latent2RGBPreviewer.

        Args:
            latent_rgb_factors (list or np.ndarray): The RGB factors for decoding.
            latent_rgb_factors_bias (list or np.ndarray, optional): The bias for RGB decoding. Defaults to None.
        """
        self.latent_rgb_factors = torch.tensor(latent_rgb_factors, device="cpu").transpose(0, 1)
        self.latent_rgb_factors_bias = None
        if latent_rgb_factors_bias is not None:
            self.latent_rgb_factors_bias = torch.tensor(latent_rgb_factors_bias, device="cpu")

    def decode_latent_to_preview(self, x0):
        """
        Decodes all latent tensors in the batch using Latent2RGB and creates a grid image.

        Args:
            x0 (torch.Tensor): The latent tensors.

        Returns:
            PIL.Image.Image: The composited grid image.
        """
        # Ensure factors are on the correct device and dtype
        self.latent_rgb_factors = self.latent_rgb_factors.to(dtype=x0.dtype, device=x0.device)
        if self.latent_rgb_factors_bias is not None:
            self.latent_rgb_factors_bias = self.latent_rgb_factors_bias.to(dtype=x0.dtype, device=x0.device)

        # Decode all latent images in the batch
        latent_images = []
        for latent in x0:
            try:
                latent_image = torch.nn.functional.linear(
                    latent.permute(1, 2, 0),
                    self.latent_rgb_factors,
                    bias=self.latent_rgb_factors_bias
                )
                latent_images.append(latent_image)
            except Exception as e:
                logging.error(f"Error decoding latent with Latent2RGB: {e}")
                continue

        # Convert all latent images to PIL Images
        pil_images = []
        for img_tensor in latent_images:
            if img_tensor is None:
                logging.warning("Decoded latent image tensor is None.")
                continue
            pil_image = preview_to_image(img_tensor)
            pil_images.append(pil_image)

        # Create a grid from the PIL Images
        grid_image = self.create_image_grid(pil_images)

        return grid_image

def get_previewer(device, latent_format):
    """
    Retrieves the appropriate previewer based on the configuration.

    Args:
        device (torch.device): The device to load models on.
        latent_format (object): The latent format configuration.

    Returns:
        LatentPreviewer or None: The selected previewer instance or None if no previewer is selected.
    """
    previewer = None
    method = args.preview_method
    if method != LatentPreviewMethod.NoPreviews:
        taesd_decoder_path = None
        if latent_format.taesd_decoder_name is not None:
            taesd_decoder_path = next(
                (fn for fn in folder_paths.get_filename_list("vae_approx")
                 if fn.startswith(latent_format.taesd_decoder_name)),
                ""
            )
            taesd_decoder_path = folder_paths.get_full_path("vae_approx", taesd_decoder_path)

        if method == LatentPreviewMethod.Auto:
            method = LatentPreviewMethod.Latent2RGB

        if method == LatentPreviewMethod.TAESD:
            if taesd_decoder_path:
                try:
                    taesd = TAESD(None, taesd_decoder_path, latent_channels=latent_format.latent_channels).to(device)
                    previewer = TAESDPreviewerImpl(taesd)
                except Exception as e:
                    logging.error(f"Failed to initialize TAESDPreviewerImpl: {e}")
            else:
                logging.warning(
                    "Warning: TAESD previews enabled, but could not find models/vae_approx/{}".format(
                        latent_format.taesd_decoder_name
                    )
                )

        if previewer is None:
            if latent_format.latent_rgb_factors is not None:
                previewer = Latent2RGBPreviewer(
                    latent_format.latent_rgb_factors,
                    latent_format.latent_rgb_factors_bias
                )
            else:
                logging.warning("Latent2RGB factors not found in latent_format.")

    return previewer

def prepare_callback(model, steps, x0_output_dict=None):
    """
    Prepares the callback function for rendering progress and previews.

    Args:
        model (object): The model being used for rendering.
        steps (int): The total number of steps in the rendering process.
        x0_output_dict (dict, optional): Dictionary to store x0 outputs. Defaults to None.

    Returns:
        function: The callback function to be used during rendering.
    """
    preview_format = "JPEG"
    if preview_format not in ["JPEG", "PNG"]:
        logging.warning(f"Unsupported preview format '{preview_format}'. Falling back to 'JPEG'.")
        preview_format = "JPEG"

    previewer = get_previewer(model.load_device, model.model.latent_format)

    pbar = comfy.utils.ProgressBar(steps)

    def callback(step, x0, x, total_steps):
        """
        The actual callback function that gets called during each rendering step.

        Args:
            step (int): The current step number.
            x0 (torch.Tensor): The latent tensors.
            x (torch.Tensor): Additional tensors (unused).
            total_steps (int): The total number of steps.
        """
        if x0_output_dict is not None:
            x0_output_dict["x0"] = x0

        preview_bytes = None
        if previewer:
            try:
                # Directly assign the tuple returned by decode_latent_to_preview_image
                preview_bytes = previewer.decode_latent_to_preview_image(preview_format, x0)
            except Exception as e:
                logging.error(f"Error generating preview image: {e}")

        pbar.update_absolute(step + 1, total_steps, preview_bytes)

    return callback
