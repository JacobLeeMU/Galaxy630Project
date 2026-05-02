import os
import numpy as np
import time
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
SCRATCH_DIR = os.path.join("/anvil/scratch/x-mwhite2/llava_inference", PROJECT_NAME)
MODEL_PATH = os.path.join(SCRATCH_DIR, "models", "llava-7b")
DATASET_PATH = os.path.join(SCRATCH_DIR, "datasets", "galaxy_sample15.h5")

with h5py.File(DATASET_PATH, 'r') as F:
    images = np.array(F['images'])
    labels = np.array(F['ans'])

# device
device = "cuda" if torch.cuda.is_available() else "cpu"
print("using device:", device) 

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

num_iter = 3

f = open("results.txt", "w")

for i in range(len(images)):
	 
    print("caption:", labels[i])
    image_obj = Image.fromarray(images[i])
    image = image_obj.convert("RGB")

    print(f'\n image {i+1}\n')
    f.write(f'image {i+1}\n')
    
    prompt = []
    
    prompt.append("Describe what you see in this image.\nASSISTANT: ")
    prompt.append("Is the galaxy in this image viewed edge on or not?\nASSISTANT: ")
    prompt.append("Is there a bar feature at the center of the galaxy?\nASSISTANT: ")
    prompt.append("Does this galaxy have a spiral arm pattern?\nASSISTANT: ")
    prompt.append("Is the central galaxy in this image more red or blue in color? What does this indicate about the galaxy's age and rate of star formation as a result? List the steps in your reasoning that led you to this conclusion.\nASSISTANT: ")
    prompt.append("Please classify the galaxy in this image into one of the following: 'barred spiral', 'unbarred tight spiral', 'unbarred loose spiral', 'edge-on without bulge', or 'edge-on with bulge', and list the steps in your reasoning that led you to this conclusion.\nASSISTANT: ")

    response_list = []
    for j in range(num_iter):
        if j == 0:
            ## classify galaxy, also describe and explain galaxy
            responses = []
            for p in prompt:
                question = "USER: <image>\n"+p

                inputs = processor(text=question, images=image, return_tensors="pt").to(device)
                input_len = inputs["input_ids"].shape[-1]
                print("generating caption...")
                output = model.generate(**inputs, max_new_tokens=512)
                response = processor.decode(output[0][input_len:], skip_special_tokens=True)
                #response = processor.decode(output[0], skip_special_tokens=True)
                responses.append(response)

                # print output
                print("question:", question)
                print("model response:", response)
                time.sleep(1)

            response_list.append(responses)

        else:
            responses = []
            for k, p in enumerate(prompt):
                prev_response = response_list[j-1][k]
                revision_cue = "USER: <image>\n"+f"Here is the previous prompt for the attached image: {p}; the previous output was: {prev_response}. Please provide an improved version of the previous output, modified to improve accuracy and clarity\nASSISTANT: "
                inputs = processor(text=revision_cue, images=image, return_tensors="pt").to(device)
                input_len = inputs["input_ids"].shape[-1] #Gets the response length
                print("generating revision...")
                output = model.generate(**inputs, max_new_tokens=512)
                #response = processor.decode(output[0], skip_special_tokens=True)
                response = processor.decode(output[0][input_len:], skip_special_tokens=True) #Slices off everything before input_len
                print("model response:", response)
                responses.append(response)
                time.sleep(1) 
            response_list.append(responses)

    for i,q in enumerate(prompt):
        f.write(f"Question {i+1}: {q}\n")
        for j,r in enumerate([row[i] for row in response_list]):
            f.write(f"Revision {j}: {r}\n")
        f.write("\n\n")
f.close()
