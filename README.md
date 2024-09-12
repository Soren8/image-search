# Image Search Application

This application allows you to search through a collection of images using natural language queries. It uses deep learning models to generate descriptions for images and perform similarity searches.

## Requirements

- Python 3.8+
- NVIDIA GPU with CUDA support
- NVIDIA drivers and CUDA toolkit installed

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/image-search.git
   cd image-search
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 transformers sentence-transformers pillow scikit-learn numpy
   ```

   Note: This command installs PyTorch with CUDA 11.8 support. If you have a different CUDA version, replace `cu118` with your version (e.g., `cu117` for CUDA 11.7).

## Usage

1. Place your images in a directory.

2. Run the script:
   ```
   python image-search.py --image_dir /path/to/your/image/directory
   ```

3. Follow the prompts to search for images using natural language queries.

## Note

Ensure that your NVIDIA GPU is properly set up with the latest drivers and CUDA toolkit before running the application.
