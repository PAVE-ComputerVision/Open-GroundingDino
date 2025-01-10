import os
import subprocess

# Directory containing the images
image_dir = "/home/ubuntu/data/prob_avn/"
# Get a list of all image files in the directory
image_files = [f for f in os.listdir(image_dir) if f.endswith('.png') or f.endswith('.jpg')]

# Define the other arguments for the inference script
config_path = "./GroundingDINO_SwinT_OGC.py"
checkpoint_path = "/home/ubuntu/roisul/output/checkpoint0004.pth"
text_prompts = "car"
output_dir = "pred_imgs"

# Loop over all image files and run the inference script on each one
for image_file in image_files:
    image_path = os.path.join(image_dir, image_file)
    command = [
        "python", "inference_on_a_image_pave.py",
        "-c", config_path,
        "-p", checkpoint_path,
        "-i", image_path,
        "-t", text_prompts,
        "-o", output_dir
    ]
    subprocess.run(command)
