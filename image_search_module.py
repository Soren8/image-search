import os
import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
import numpy as np
import pickle
import yaml
from PIL import Image
import warnings
import gc

# Load configuration
with open('config.yml', 'r') as config_file:
    config = yaml.safe_load(config_file)

IMAGE_DIR = os.path.expandvars(os.path.expanduser(config['image_path']))
INDEX_FILE = os.path.join(IMAGE_DIR, 'image_index.pkl')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Current device: {device}")

# Global variables for index, image paths, and descriptions
index = None
image_paths = []
descriptions = []

def load_models():
    print("Loading models...")
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    description_model = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-opt-2.7b", 
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    ).to(device)
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    return processor, description_model, embedding_model

def unload_models(processor, description_model, embedding_model):
    print("Unloading models...")
    del processor
    del description_model
    del embedding_model
    torch.cuda.empty_cache()
    gc.collect()

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
    existing_images = set(image_paths)
    
    current_images = set()
    new_images = []
    
    for img_name in os.listdir(IMAGE_DIR):
        if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp')):
            img_path = os.path.join(IMAGE_DIR, img_name)
            current_images.add(img_path)
            if img_path not in existing_images:
                new_images.append(img_path)
    
    removed_images = existing_images - current_images
    
    if not new_images and not removed_images:
        print("No changes in the image directory. Using existing index.")
        return
    
    if removed_images:
        print(f"Removing {len(removed_images)} images from the index that are no longer in the directory.")
        image_paths = [path for path in image_paths if path not in removed_images]
        descriptions = [desc for path, desc in zip(image_paths, descriptions) if path not in removed_images]
    
    if new_images:
        print(f"Found {len(new_images)} new images. Processing...")
        processor, description_model, embedding_model = load_models()
        
        try:
            for img_path in new_images:
                print(f"Preprocessing image: {os.path.basename(img_path)}")
                image = Image.open(img_path)
                
                # Generate description
                inputs = processor(images=image, return_tensors="pt").to(device)
                outputs = description_model.generate(**inputs, max_length=200)
                description = processor.decode(outputs[0], skip_special_tokens=True)
                print(description)
                
                # Store for indexing
                image_paths.append(img_path)
                descriptions.append(description)
            
            # Create embeddings for all images
            all_embeddings = embedding_model.encode(descriptions)
            
            # Setup nearest neighbors for quick search
            index = NearestNeighbors(n_neighbors=5, algorithm='ball_tree').fit(all_embeddings)
        finally:
            unload_models(processor, description_model, embedding_model)
            del inputs, outputs, all_embeddings
            torch.cuda.empty_cache()
            gc.collect()
        
        # Save the updated index
        save_index(index, image_paths, descriptions)
        print(f"Index updated and saved at: {INDEX_FILE}")
    
    # Print out descriptions/tags for new images
    if new_images:
        print("\nNew Image Descriptions/Tags:")
        for path, desc in zip(image_paths[-len(new_images):], descriptions[-len(new_images):]):
            print(f"{os.path.basename(path)}: {desc}")

def search_images(query):
    global index, image_paths, descriptions
    
    if index is None:
        print("Please preprocess the images first or ensure the index file exists.")
        return []
    
    # Load only the embedding model for search
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Convert query to embedding
    query_embedding = embedding_model.encode([query])
    
    # Find the nearest neighbors (use a large number to get all results)
    distances, indices = index.kneighbors(query_embedding, n_neighbors=len(image_paths))
    
    results = []
    for i in range(len(indices[0])):
        idx = indices[0][i]
        similarity = 1 - distances[0][i]  # Convert distance to similarity
        results.append({
            'path': image_paths[idx],
            'description': descriptions[idx],
            'similarity': float(similarity)  # Convert numpy float to Python float
        })
    
    # Sort results by similarity in descending order
    results.sort(key=lambda x: x['similarity'], reverse=True)
    
    # Unload the embedding model
    del embedding_model
    torch.cuda.empty_cache()
    gc.collect()
    
    # Filter results based on similarity score
    positive_results = [r for r in results if r['similarity'] > 0]
    if positive_results:
        return positive_results
    else:
        return results[:5]  # Return top 5 if no positive scores

if __name__ == "__main__":
    preprocess_images()
    
    while True:
        query = input("Enter your search query or 'exit' to quit: ")
        if query.lower() == 'exit':
            break
        results = search_images(query)
        for result in results:
            print(f"Path: {result['path']}, Similarity: {result['similarity']:.4f}")