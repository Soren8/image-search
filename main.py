import eel
import os
import base64
import subprocess
import json
from image_search_module import search_images, preprocess_images
import yaml

# Set web files folder
eel.init('web')

# Load configuration
with open('config.yml', 'r') as config_file:
    config = yaml.safe_load(config_file)

CHROME_PATH = os.path.expandvars(os.path.expanduser(config['chrome_path']))

@eel.expose
def python_search_images(query):
    results = search_images(query)
    # Instead of converting file paths, we'll create data URLs for images
    for result in results:
        with open(result['path'], 'rb') as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
            result['data_url'] = f"data:image/jpeg;base64,{encoded_image}"
        # We'll keep the original path for reference, but won't use it directly in the frontend
        result['original_path'] = result['path']
        del result['path']
    return results

@eel.expose
def python_preprocess_images():
    preprocess_images()
    return "Preprocessing complete"

@eel.expose
def open_file_location(file_path):
    if os.path.exists(file_path):
        if os.name == 'nt':  # Windows
            subprocess.Popen(f'explorer /select,"{file_path}"')
        elif os.name == 'posix':  # macOS and Linux
            subprocess.Popen(['open', '-R', file_path])
    else:
        print(f"File not found: {file_path}")

# Start the app
if __name__ == '__main__':
    eel.browsers.set_path('chrome', CHROME_PATH)
    eel.start('index.html', size=(1280, 1400), mode='chrome')