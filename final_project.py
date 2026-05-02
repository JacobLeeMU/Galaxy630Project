#Authors: Jacob Lee and Ellie White
#Course: CS630 Machine Learning
#Final project LLaVA prompting code using a subset of the Galaxy10 DECaLS dataset https://astronn.readthedocs.io/en/latest/galaxy10.html, produces results.txt

import os
import numpy as np
import time
import torch
import h5py
import json
from tensorflow.keras import utils

from PIL import Image
from transformers import LlavaForConditionalGeneration, AutoProcessor

#Project name
PROJECT_NAME = os.path.basename(os.getcwd())
#Directories
SCRATCH_DIR = "/anvil/scratch/x-mwhite2/llava_inference"
MODEL_PATH = os.path.join(SCRATCH_DIR, "models", "llava-7b")
DATASET_PATH = os.path.join(SCRATCH_DIR, PROJECT_NAME,"galaxy_sample15.h5")

#Opens our data file, importing into images and labels (The model does not get passed the labels):
## The datafile is a subset of Galaxy10 DECaLS with 15 images and labels:
	## 3 samples of barred spirals
	## 3 unbarred tight spirals
	## 3 unbarred loose spirals
	## 3 Edge-on without bulge
	## 3 Edge-on with bulge

## define how many images from the subset to use
## (this should be set to 15, unless using a smaller
## number for short tests)
img_num = 15

## retrieve the images and labels, described above
with h5py.File(DATASET_PATH, 'r') as F:
    images = np.array(F['images'][:img_num])
    labels = np.array(F['ans'][:img_num])

#Device
device = "cuda" if torch.cuda.is_available() else "cpu"
print("using device:", device) 

#Load model and processor
print("loading model and processor from:", MODEL_PATH)
model = LlavaForConditionalGeneration.from_pretrained(MODEL_PATH).to(device)
processor = AutoProcessor.from_pretrained(MODEL_PATH)


num_iter = 3

#Open results file for writing
f = open("results.txt", "w")

#Iterates over each image
for i in range(len(images)):
    print(f'\n\n\n\n\n----------Image---------- {i+1}\n')
    f.write(f'\n\n\n\n\n----------Image---------- {i+1}\n')
    image_obj = Image.fromarray(images[i])
    image = image_obj.convert("RGB")
	
    print("Correct label according to dataset:", labels[i])
    f.write(f'Correct label according to dataset: {labels[i]}\n')
    

    #We append all of our prompts to a list
    prompt = []
    
    prompt.append("Describe what you see in this image.")
    prompt.append("Is the galaxy in this image viewed edge on or not?")
    prompt.append("Is there a bar feature at the center of the galaxy?")
    prompt.append("Does this galaxy have a spiral arm pattern?")
    prompt.append("Is the central galaxy in this image more red or blue in color? What does this indicate about the galaxy's age and rate of star formation as a result? List the steps in your reasoning that led you to this conclusion.")
    prompt.append("Please classify the galaxy in this image into one of the following: 'barred spiral', 'unbarred tight spiral', 'unbarred loose spiral', 'edge-on without bulge', or 'edge-on with bulge', and list the steps in your reasoning that led you to this conclusion.")

    response_list = [] #This is a list of lists for our responses. Each list is the set of responses for iteration j of responses
    
    for j in range(num_iter):
        #Initial responses
        if j == 0:
            responses = [] #This iteration's responses

            #Iterate over each prompt
            for p in prompt:
                question = "USER: <image>\n"+p+"\nASSISTANT: "

                inputs = processor(text=question, images=image, return_tensors="pt").to(device)
                input_len = inputs["input_ids"].shape[-1] #Gets the response length
                print("generating caption...")
                output = model.generate(**inputs, max_new_tokens=512)
                response = processor.decode(output[0][input_len:], skip_special_tokens=True) #Because LLaVa echoes back the input, we need to slice that off using [input_len:]
                responses.append(response)

                #Print output
                print("question:", question)
                print("model response:", response)
                
            response_list.append(responses)
        
        #Revisions
        else:
            responses = []
            for k, p in enumerate(prompt):
                prev_response = response_list[j-1][k] #The previous response for this prompt
                revision_cue = "USER: <image>\n"+f"Here is the previous prompt for the attached image: {p}; the previous output was: {prev_response}. Please provide (1) a very brief summary of inaccuracies or problems with the previous output, and (2) an updated version of the previous output, modified to improve accuracy and clarity\nASSISTANT: "
                inputs = processor(text=revision_cue, images=image, return_tensors="pt").to(device)
                input_len = inputs["input_ids"].shape[-1] #Gets the response length
                print("generating revision...")
                output = model.generate(**inputs, max_new_tokens=512)
                response = processor.decode(output[0][input_len:], skip_special_tokens=True) #Because LLaVa echoes back the output, we need to slice that off using [input_len:]
                print("model response:", response)
                responses.append(response)
            response_list.append(responses)
    
    #Prints to the results file for this image
    for i,q in enumerate(prompt):
        f.write(f"Question {i+1}: {q}\n")
        for j,r in enumerate([row[i] for row in response_list]):
            f.write(f"Revision {j}: {r}\n\n")
        f.write("\n\n")

f.close() 
