# Enhanced Latent Video Nodes for ComfyUI

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Classes Overview](#classes-overview)
  - [EmptyMochiLatentVideo](#emptymochilatentvideo)
  - [ImageMochiLatentVideo](#imagemochilatentvideo)
- [Compatibility](#compatibility)
- [How It Works](#how-it-works)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)
- [Learn More](#learn-more)

## Introduction

Welcome to the **Enhanced Latent Video Nodes** for [ComfyUI](https://github.com/comfyui/ComfyUI)! This enhancement introduces new latent video processing capabilities to ComfyUI, enabling users to generate and manipulate latent videos with ease. Whether you're a developer looking to extend ComfyUI's functionality or a creative user aiming to explore video generation, these nodes provide powerful tools to enhance your workflow.

## Features

- **Latent Video Generation**: Create latent videos from existing latent tensors with customizable frame lengths.
- **Noise Scaling**: Apply controlled noise scaling across video frames to introduce variation and dynamics.
- **Flexible Latent Handling**: Supports both 4D and 5D latent tensors, accommodating various input formats.
- **Batch Processing**: Handle multiple latent samples simultaneously with customizable batch sizes.
- **Easy Integration**: Seamlessly integrate with existing ComfyUI workflows by adding new nodes.

## Installation

Follow these steps to integrate the Enhanced Latent Video Nodes into your ComfyUI setup:

1. **Backup Existing Files**

   Before making any changes, ensure you back up the original `nodes_mochi.py` to prevent data loss.

   ```bash
   cd path/to/comfy/comfy_extras/
   cp nodes_mochi.py nodes_mochi_backup.py
   ```

2. **Download the Enhanced Nodes**

   Clone this repository or download the modified `nodes_mochi.py` file directly.

   ```bash
   git clone https://github.com/yourusername/enhanced-latent-video-nodes.git
   ```

3. **Replace the Original File**

   Copy the modified `nodes_mochi.py` into the ComfyUI extras directory, replacing the existing file.

   ```bash
   cp enhanced-latent-video-nodes/nodes_mochi.py path/to/comfy/comfy_extras/
   ```

4. **Verify Dependencies**

   Ensure that all required Python packages are installed. The Enhanced Latent Video Nodes rely on `torch` and `comfy` modules. If you haven't already, install or upgrade them:

   ```bash
   pip install --upgrade torch comfyui
   ```

5. **Restart ComfyUI**

   After replacing the file, restart ComfyUI to load the new nodes.

   ```bash
   python path/to/comfy/main.py
   ```

## Usage

Once installed, the new latent video nodes are available within ComfyUI's node library under the **latent/mochi** category. Here's how to utilize them:

1. **Add Nodes to Your Workflow**

   - **EmptyMochiLatentVideo**: Use this node to generate a blank latent video tensor with specified dimensions and frame length.
   - **ImageMochiLatentVideo**: Use this node to convert existing latent tensors into video format with customizable noise scaling.

2. **Configure Node Parameters**

   - **EmptyMochiLatentVideo**
     - **Width**: Width of the latent video (in pixels).
     - **Height**: Height of the latent video (in pixels).
     - **Length**: Number of frames in the video.
     - **Batch Size**: Number of latent samples to generate simultaneously.

   - **ImageMochiLatentVideo**
     - **Latent**: Input latent tensor to be converted into video format.
     - **Length**: Desired number of frames in the output video.
     - **Noise Start**: Initial noise scaling factor.
     - **Noise End**: Final noise scaling factor.

3. **Connect Nodes Appropriately**

   Integrate these nodes into your existing ComfyUI workflow by connecting them to relevant input and output nodes. For example, connect `EmptyMochiLatentVideo` to a rendering node to generate new videos or use `ImageMochiLatentVideo` to modify existing latent data.

4. **Generate and Preview Videos**

   After configuring, run your workflow to generate latent videos. Use preview nodes to visualize the results within ComfyUI.

## Classes Overview

### EmptyMochiLatentVideo

#### Description

The `EmptyMochiLatentVideo` node generates a blank latent video tensor filled with zeros. This is useful for initializing latent videos with specific dimensions and frame counts before applying transformations or rendering.

#### Parameters

- **Width** (`INT`): Width of the latent video in pixels. *Default: 848, Min: 16, Max: nodes.MAX_RESOLUTION, Step: 16*
- **Height** (`INT`): Height of the latent video in pixels. *Default: 480, Min: 16, Max: nodes.MAX_RESOLUTION, Step: 16*
- **Length** (`INT`): Number of frames in the video. *Default: 25, Min: 7, Max: nodes.MAX_RESOLUTION, Step: 6*
- **Batch Size** (`INT`): Number of latent samples to generate simultaneously. *Default: 1, Min: 1, Max: 4096*

#### Example Usage

```python
latent_video = EmptyMochiLatentVideo.generate(width=1024, height=768, length=30, batch_size=2)
```

#### Returns

- **LATENT**: Generated latent video tensor with shape `[batch_size, 12, frames, height//8, width//8]`.

### ImageMochiLatentVideo

#### Description

The `ImageMochiLatentVideo` node converts existing latent tensors into video format by adding a frames dimension and applying noise scaling across frames. This introduces variation and dynamics into the latent video, enhancing the rendering process.

#### Parameters

- **Latent** (`LATENT`): Input latent tensor to be converted into video format.
- **Length** (`INT`): Desired number of frames in the output video. *Default: 25, Min: 7, Max: nodes.MAX_RESOLUTION, Step: 6*
- **Noise Start** (`FLOAT`): Initial noise scaling factor applied to the first frame. *Default: 0.1, Min: 0.0, Max: 1.0, Step: 0.01*
- **Noise End** (`FLOAT`): Final noise scaling factor applied to the last frame. *Default: 0.4, Min: 0.0, Max: 1.0, Step: 0.01*

#### Example Usage

```python
modified_latent = ImageMochiLatentVideo.generate(latent=existing_latent, length=30, noise_start=0.2, noise_end=0.5)
```

#### Returns

- **LATENT**: Modified latent video tensor with shape `[batch_size, 12, frames, height//8, width//8]`.

## Compatibility

The Enhanced Latent Video Nodes are compatible with:

- **ComfyUI Versions**: Tested with ComfyUI v1.0 and above.
- **Python Versions**: Python 3.8 and later.
- **PyTorch Versions**: PyTorch 1.7 and above.

Ensure your environment meets these requirements to avoid any compatibility issues.

## How It Works

### EmptyMochiLatentVideo

1. **Initialization**: Determines the device (CPU or GPU) for tensor operations.
2. **Latent Tensor Generation**: Creates a latent tensor filled with zeros based on the specified width, height, length, and batch size.
3. **Output Formatting**: Returns the generated tensor in the expected format for ComfyUI nodes.

### ImageMochiLatentVideo

1. **Initialization**: Determines the device for tensor operations.
2. **Latent Tensor Processing**:
   - **Shape Handling**: Supports both 4D `[batch_size, channels, height, width]` and 5D `[batch_size, channels, frames, height, width]` latent tensors.
   - **Channel Adjustment**: Ensures the latent tensor has 12 channels, expanding from 4 if necessary.
   - **Frame Calculation**: Determines the number of frames based on the specified length.
   - **Noise Scaling**: Applies linearly increasing noise scaling factors from `noise_start` to `noise_end` across frames.
3. **Output Formatting**: Returns the modified latent tensor in the expected format for ComfyUI nodes.

## Troubleshooting

If you encounter issues after installing the Enhanced Latent Video Nodes, consider the following steps:

1. **Check Dependencies**

   Ensure that all required Python packages are installed and up-to-date.

   ```bash
   pip install --upgrade torch comfyui
   ```

2. **Verify Node Integration**

   - Ensure that the `nodes_mochi.py` file has been correctly replaced in the `comfy_extras` directory.
   - Restart ComfyUI after making changes to load the new nodes.

3. **Inspect Logs for Errors**

   Review the ComfyUI logs to identify any error messages related to the new nodes.

4. **Validate Input Shapes**

   Ensure that the latent tensors you provide match the expected dimensions:
   
   - For `EmptyMochiLatentVideo`: `[batch_size, 12, frames, height//8, width//8]`
   - For `ImageMochiLatentVideo`: Compatible 4D or 5D tensors with 12 channels.

5. **Restore Backup**

   If issues persist, revert to the original `nodes_mochi.py` using your backup.

   ```bash
   cp nodes_mochi_backup.py path/to/comfy/comfy_extras/nodes_mochi.py
   ```

6. **Seek Community Support**

   Engage with the [ComfyUI Community](https://github.com/comfyui/ComfyUI/discussions) for further assistance.

## Contributing

Contributions are welcome! If you'd like to improve the Enhanced Latent Video Nodes, follow these steps:

1. **Fork the Repository**

   Click the "Fork" button at the top-right corner of this page.

2. **Create a Feature Branch**

   ```bash
   git checkout -b feature/improve-video-nodes
   ```

3. **Commit Your Changes**

   ```bash
   git commit -m "Enhance noise scaling in ImageMochiLatentVideo"
   ```

4. **Push to Your Fork**

   ```bash
   git push origin feature/improve-video-nodes
   ```

5. **Submit a Pull Request**

   Open a pull request detailing your changes and their benefits.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments

- **ComfyUI Team**: For creating an amazing and extensible user interface.
- **PyTorch Community**: For the powerful tensor computation library.
- **Community Contributors**: For feedback and support in enhancing these nodes.

## Learn More

For more detailed information, visit our [project blog](https://blog.comfy.org/mochi-1/).

---
*Feel free to customize this README further to better fit your project's specifics and branding.*
```
