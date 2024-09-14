# Image Search Application

This application allows you to search through a collection of images using natural language queries. It uses deep learning models to generate descriptions for images and perform similarity searches.

## Requirements

- Python 3.8+
- NVIDIA GPU with CUDA support
- NVIDIA drivers and CUDA toolkit installed

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/Soren8/image-search.git
   cd image-search
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

4. Create a `config.yml` file in the project root directory with the following content:
   ```yaml
   # Path to Chrome executable
   chrome_path: "C:/Program Files/Google/Chrome/Application/chrome.exe"

   # Path to the directory containing images
   image_path: "%USERPROFILE%/path/to/your/images"
   ```
   Adjust the paths according to your system configuration.

## Usage

1. Ensure your images are in the directory specified in `config.yml`.

2. Run the script:
   ```
   python main.py
   ```

3. The script will first preprocess all images in the specified directory, generating descriptions and creating an index. This may take some time depending on the number of images and your hardware.

4. Once preprocessing is complete, you can enter search queries. The application will return the most relevant images based on your query.

## Notes

- The first run may take longer as it downloads the necessary AI models and preprocesses the images.
- GPU acceleration is highly recommended for faster processing, especially for large image collections.
- Ensure you have sufficient disk space for the AI models and the generated index.

## Troubleshooting

If you encounter any issues:
- Ensure all paths in `config.yml` are correct and accessible.
- Check that you have the necessary permissions to read/write in the specified directories.
- Make sure you have a compatible CUDA installation if using GPU acceleration.
- Make sure you have the CUDA-enabled version of PyTorch installed.

## Acknowledgements

This project uses the following open-source libraries:
- BLIP-2 for image captioning
- Sentence Transformers for text embeddings
- PyTorch for deep learning computations
- scikit-learn for nearest neighbor search
