import eel
import os
import base64
import subprocess
import json
from image_search_module import search_images, preprocess_images

# Set web files folder
eel.init('web')

chromium_path = r"C:\Program Files (x86)\Chromium\chrome.exe"

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
def open_file(path):
    # The path is now directly usable
    if os.path.exists(path):
        if os.name == 'nt':  # For Windows
            os.startfile(path)
        elif os.name == 'posix':  # For macOS and Linux
            subprocess.call(('xdg-open', path))
    else:
        print(f"File not found: {path}")

# Start the app
if __name__ == '__main__':
    eel.browsers.set_path('chrome', chromium_path)
    eel.start('index.html', size=(1280, 1400), mode='chrome')