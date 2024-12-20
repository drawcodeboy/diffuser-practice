from diffusers import DiffusionPipeline
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--counts", type=int, default=10)

args = parser.parse_args()

model_id = "CompVis/stable-diffusion-v1-4"
pipeline = DiffusionPipeline.from_pretrained(model_id).to('cuda')

for idx in range(0, args.counts):
    prompt = "A photograph of an astronaut riding a horse"

    image = pipeline(prompt).images[0]

    image.save(f"figures/astronaut_riding_a_horse_{idx+1:02d}.png")
    print(type(image))