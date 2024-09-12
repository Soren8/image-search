
import os
import torch
from transformers import AutoModelForVision2Seq, AutoProcessor, AutoTokenizer
from sentence_transformers import SentenceTransformer
import sentence_transformers.models as models
from sklearn.neighbors import NearestNeighbors
import numpy as np
import pickle
import json
import argparse
from PIL import Image
import warnings
from flask import Flask, render_template, request, send_from_directory

app = Flask(__name__)

# Directory containing images
parser = argparse.ArgumentParser(description='Meme Search')
parser.add_argument('--image_dir', type=str, default='', help='Path to the directory containing images')
args = parser.parse_args()

IMAGE_DIR = args.image_dir if args.image_dir else '.'
INDEX_FILE = 'image_index.pkl'
META_FILE = 'image_metadata.json'

# Ensure you're using GPU if available
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Torch version: {torch.__version__}")
print(f"Torch CUDA version: {torch.version.cuda}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Current device: {device}")

# Load models, suppress warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    print("Loading description model...")
    description_model = AutoModelForVision2Seq.from_pretrained("microsoft/git-base-coco").to(device)
    print("Loading processor...")
    processor = AutoProcessor.from_pretrained("microsoft/git-base-coco")
    print("Loading embedding model...")
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def save_index(index, image_paths, descriptions):
    with open(INDEX_FILE, 'wb') as f:
        pickle.dump({
            'index': index,
            'image_paths': image_paths,
            'descriptions': descriptions
        }, f)

def load_index():
    if os.path.exists(INDEX_FILE):
        with open(INDEX_FILE, 'rb') as f:
            data = pickle.load(f)
            return data['index'], data['image_paths'], data['descriptions']
    return None, [], []

def preprocess_images():
    global index, image_paths, descriptions
    
    index, image_paths, descriptions = load_index()
    if index is not None:
        print("Index loaded from file.")
    else:
        print("Starting image preprocessing...")
        for img_name in os.listdir(IMAGE_DIR):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                print(f"Preprocessing image: {img_name}")
                img_path = os.path.join(IMAGE_DIR, img_name)
                image = Image.open(img_path)
                
                # Generate description
                inputs = processor(images=image, return_tensors="pt").to(device)
                outputs = description_model.generate(**inputs)
                description = processor.decode(outputs[0], skip_special_tokens=True)
                print(description)
                
                # Store for indexing
                image_paths.append(img_path)
                descriptions.append(description)
        
        # Create embeddings
        embeddings = embedding_model.encode(descriptions)
        
        # Setup nearest neighbors for quick search
        index = NearestNeighbors(n_neighbors=5, algorithm='ball_tree').fit(embeddings)
        
        # Save the index
        save_index(index, image_paths, descriptions)
        print("New index created and saved.")

    # Print out descriptions/tags for each image
    print("\nImage Descriptions/Tags:")
    for path, desc in zip(image_paths, descriptions):
        print(f"{os.path.basename(path)}: {desc}")

def search_images(query):
    if index is None:
        print("Please preprocess the images first or ensure the index file exists.")
        return []
    
    # Convert query to embedding
    query_embedding = embedding_model.encode([query])
    
    # Determine the maximum number of neighbors we can search for
    max_neighbors = min(5, len(image_paths))  # Use the smaller of 5 or the number of images
    
    if max_neighbors == 0:
        print("No images have been indexed.")
        return []

    # Find the nearest neighbors
    distances, indices = index.kneighbors(query_embedding, n_neighbors=max_neighbors)
    
    results = []
    for i in range(len(indices[0])):  # Iterate over the actual number of returned indices
        idx = indices[0][i]
        results.append({
            'path': image_paths[idx],
            'description': descriptions[idx],
            'distance': distances[0][i]
        })
    return results

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        query = request.form['query']
        results = search_images(query)
        return render_template('results.html', results=results)
    return render_template('index.html')

@app.route('/images/<path:filename>')
def serve_image(filename):
    return send_from_directory(IMAGE_DIR, filename)

# Add this new route to serve images using their full path
@app.route('/image_file/<path:filepath>')
def serve_image_file(filepath):
    directory, filename = os.path.split(filepath)
    return send_from_directory(directory, filename)

# Usage
if __name__ == "__main__":
    preprocess_images()
    
    # Run Flask app
    app.run(debug=True)

    while True:
        query = input("Enter your search query or 'exit' to quit: ")
        if query.lower() == 'exit':
            break
        results = search_images(query)
        for result in results:
            print(f"Path: {result['path']}, Similarity: {1 - result['distance']:.4f}")