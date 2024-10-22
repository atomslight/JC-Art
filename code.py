# Import the necessary libraries
!pip install diffusers transformers torch accelerate gradio moviepy opencv-python

from diffusers import StableDiffusionImg2ImgPipeline
import torch
import numpy as np
import cv2
import moviepy.editor as mpy
import gradio as gr
from PIL import Image

# Load the pre-trained Stable Diffusion image-to-image model
model_id = "runwayml/stable-diffusion-v1-5"  # Pretrained Stable Diffusion image-to-image model
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")  # Ensure the model uses GPU for faster generation

# Function to generate a single frame based on a prompt and an input image
def generate_frame(prompt, input_image, strength=0.75):
    # Convert the numpy image to PIL format as required by the pipeline
    input_image = Image.fromarray(input_image)
    
    # Generate an image using image-to-image generation
    image = pipe(prompt=prompt, image=input_image, strength=strength, guidance_scale=7.5).images[0]
    
    return np.array(image)  # Convert to numpy array for further processing

# Function to create a surrealism/cubism/impressionism-inspired video from an input image
def create_video(image, style_prompt, output_path='video.mp4'):
    # Prepare a list of prompts that alternate between surrealism, cubism, and impressionism
    #styles = ['surrealism', 'cubism', 'impressionism']
    prompts = [f"{style_prompt}, {i}% dreamlike painting" for i in range(10, 110, 10)]

    frames = []
    height, width = image.shape[:2]  # Get original image size

    for i, prompt in enumerate(prompts):
        print(f"Generating frame {i+1} with prompt: {prompt}")
        frame = generate_frame(prompt, image)  # Use image-to-image generation
        frame_resized = cv2.resize(frame, (width, height))  # Resize to match input image size
        frames.append(frame_resized)  # Collect frames

    # Convert the frames into a video using MoviePy
    video_clip = mpy.ImageSequenceClip(frames, fps=10)  # 10 frames per second
    video_clip.write_videofile(output_path, codec="libx264")  # Save the video
    return output_path

# Gradio interface function
def generate_video(image, style_prompt):
    output_video_path = 'video.mp4'
    # Create the surrealism, cubism, impressionism video using the provided image and style prompt
    video_path = create_video(np.array(image), style_prompt, output_path=output_video_path)
    return video_path  # Return the path to the generated video

# Gradio Interface Setup
iface = gr.Interface(
    fn=generate_video,
    inputs=[gr.Image(type="numpy"), "text"],
    outputs="video",
    title="Surrealism/Cubism/Impressionism Art Video Generator",
    description="Upload an image and provide a style prompt to generate a surrealism, cubism, impressionism art video."
)

# Launch the Gradio interface
iface.launch()
