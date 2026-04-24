import os
import numpy as np
import torch
import h5py
import json
from tensorflow.keras import utils

from PIL import Image
from transformers import LlavaForConditionalGeneration, AutoProcessor

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

    response_list = []
    for j in range(len(num_iter)):
        if j == 0:
            ## classify galaxy, also describe and explain galaxy
            
            prompt = []
            
            prompt.append("Describe what you see in this image.")
            
            prompt.append("Is the galaxy in this image viewed edge on or not?")
            
            #prompt.append("Does the galaxy appear to have a central bulge?")
            
            prompt.append("Is there a bar feature at the center of the galaxy?")
            
            prompt.append("Does this galaxy have a spiral arm pattern?")
            
            prompt.append("Is the central galaxy in this image more red or blue in color? What does this indicate about the galaxy's age and rate of star formation as a result?")

            prompt.append("Please classify the galaxy in this image into one of the following: 'barred spiral', 'unbarred tight spiral', 'unbarred loose spiral', 'edge-on without bulge', or 'edge-on with bulge'")

            responses = []
            for p in prompt:
                question = "<image>\n"+p

                inputs = processor(text=question, images=image, return_tensors="pt").to(device)
                print("generating caption...")
                output = model.generate(**inputs, max_new_tokens=50)
                response = processor.decode(output[0], skip_special_tokens=True)
                responses.append(response)

                # print output
                print("question:", question)
                print("model response:", response)
            
            response_list.append(responses)

        else:
            responses = []
            for k, p in enumerate(prompt):
                revision_cue = "<image>\n"+f"Here is the previous prompt for the attached image: {p} \n the corresponding output was: {responses[j-1][k]} \n Critique this output and provide an improved response if possible"
                print("model response:", response)
                inputs = processor(text=revision_cue, images=image, return_tensors="pt").to(device)
                print("generating revision...")
                output = model.generate(**inputs, max_new_tokens=50)
                response = processor.decode(output[0], skip_special_tokens=True)
                responses.append(response)
            response_list.append(responses)

with open("results.txt", "w") as f:
    for i,q in enumerate(questions):
        f.write(f"Question {i+1}: {q}")
        for j,r in enumerate(responses[:,i]):
            f.write(f"Revision {j}: {r}")
        f.write("\n\n")

