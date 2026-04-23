import os
import numpy as np
import torch
import h5py
import json
from tensorflow.keras import utils

#from PIL import Image
#from transformers import LlavaForConditionalGeneration, AutoProcessor

#from datasets import load_dataset

# project name
PROJECT_NAME = os.path.basename(os.getcwd())
# directories
SCRATCH_DIR = os.path.join("/anvil/scratch/x-mwhite2", PROJECT_NAME)
MODEL_PATH = os.path.join(SCRATCH_DIR, "models", "llava-7b")
DATASET_PATH = os.path.join(SCRATCH_DIR, "datasets", "galaxy_sample15.h5")

with h5py.File(DATASET_PATH, 'r') as F:
    images = np.array(F['images'])
    labels = np.array(F['ans'])

# device
'''device = "cuda" if torch.cuda.is_available() else "cpu"
print("using device:", device)''' 

# load model and processor
print("loading model and processor from:", MODEL_PATH)
model = LlavaForConditionalGeneration.from_pretrained(MODEL_PATH).to(device)
processor = AutoProcessor.from_pretrained(MODEL_PATH)

print(labels[0])




## select sample of 15 from images and labels
	## 3 samples of barred spirals
	## 3 unbarred tight spirals
	## 3 unbarred loose spirals
	## 3 Edge-on without bulge
	## 3 Edge-on with bulge
	## define as image_subset and label_subset

## display each image 

num_iter = 5

for i in range(len(image_subset)):
	 
    print("caption:", label_subset[i])
    image = Image.open(image_subset[i].convert("RGB"))

    for j in range(len(num_iter)):
        if j == 0:
            ## classify galaxy, also describe and explain galaxy
            
            prompts = []
            
            prompt0 = "Describe what you see in this image."  
            
            prompt1 = "Is the galaxy in this image viewed edge on or not?" 
            
            #prompt2 = "Does the galaxy appear to have a central bulge?"
            
            prompt2 = "Is there a bar feature at the center of the galaxy?"
            
            prompt3 = "Does this galaxy have a spiral arm pattern?" 
            
            prompt4 = "Is the central galaxy in this image more red or blue in color? What does this indicate about the galaxy's age and rate of star formation as a result?"             
            prompt5 =  "Please classify the galaxy in this image into one of the following: 'barred spiral', 'unbarred tight spiral', 'unbarred loose spiral', 'edge-on without bulge', or 'edge-on with bulge'"       

            question = "<image>\n"+prompt

            inputs = processor(text=question, images=image, return_tensors="pt").to(device)
            print("generating caption...")
            output = model.generate(**inputs, max_new_tokens=50)
            response = processor.decode(output[0], skip_special_tokens=True)
            
            # print output
            print("question:", question)
            print("model response:", response)

        else:
            revision_cue = "<image>\n"+f"Here is the previous output describing the image: {response} \
                        Critique this output and print an improved classification and summary \
                        of the image. Here is the previous prompt: {prompt}"

            inputs = processor(text=revision_cue, images=image, return_tensors="pt").to(device)
            print("generating caption...")
            output = model.generate(**inputs, max_new_tokens=50)
            response = processor.decode(output[0], skip_special_tokens=True)

            # print output
            print("question:", question)
            print("model response:", response)
