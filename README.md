# comfyui-nodes
collection of various custom work I've done for ComfyUI

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Compatibility](#compatibility)
- [How It Works](#how-it-works)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Introduction

Welcome to the **Enhanced Latent Previewer** for [ComfyUI](https://github.com/comfyui/ComfyUI)! This modified version of the built-in `latent_preview.py` file introduces advanced functionalities to improve the previewing experience during image rendering tasks. Whether you're a developer looking to extend ComfyUI's capabilities or a power user seeking better visualization tools, this enhanced previewer is designed to meet your needs.

## Features

- **Grid Previews**: Display all images in a batch as a cohesive grid, providing a comprehensive overview of rendered results.
- **Resampling Filter Compatibility**: Automatically adapts to different versions of the Pillow library, ensuring smooth image resizing without compatibility issues.
- **Robust Error Handling**: Comprehensive logging and error management to prevent crashes and provide clear diagnostic messages.
- **Performance Optimizations**: Efficient decoding and grid creation processes to handle larger batch sizes with minimal lag.
- **Customizable Grid Layouts**: Adjust the maximum number of columns to suit your visualization preferences.
- **Support for Multiple Preview Formats**: Generate previews in both JPEG and PNG formats based on user configuration.

## Installation

Follow these steps to replace the built-in `latent_preview.py` with the Enhanced Latent Previewer:

1. **Backup Existing File**

   Before making any changes, ensure you back up the original `latent_preview.py` to prevent data loss.

   ```bash
   cd path/to/ComfyUI
   cp latent_preview.py latent_preview_backup.py
