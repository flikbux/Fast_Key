from PIL import Image
import torch
from tqdm.auto import tqdm

from point_e.diffusion.configs import DIFFUSION_CONFIGS, diffusion_from_config
from point_e.diffusion.sampler import PointCloudSampler
from point_e.models.download import load_checkpoint
from point_e.models.configs import MODEL_CONFIGS, model_from_config
from point_e.util.plotting import plot_point_cloud

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Creating base model")
base_name = "base300M"  # Use base1B for better results
base_model = model_from_config(MODEL_CONFIGS[base_name], device)
base_model.eval()
base_diffusion = diffusion_from_config(DIFFUSION_CONFIGS[base_name])

print("Creating upsample model")
upsampler_model = model_from_config(MODEL_CONFIGS["upsample"], device)
upsampler_model.eval()
upsampler_diffusion = diffusion_from_config(DIFFUSION_CONFIGS["upsample"])

print("Downloading base checkpoint")
base_model.load_state_dict(load_checkpoint(base_name, device))

print("Downloading upsampler checkpoint")
upsampler_model.load_state_dict(load_checkpoint("upsample", device))

# Combine the image-to-point cloud and upsampler model
sampler = PointCloudSampler(
    device=device,
    models=[base_model, upsampler_model],
    diffusions=[base_diffusion, upsampler_diffusion],
    num_points=[1024, 4096 - 1024],
    aux_channels=["R", "G", "B"],
    guidance_scale=[3.0, 3.0],
)




# Load an image to condition on
img_path = "licorne.png" # Fill in your image path
img = Image.open(img_path)



# Produce a sample from the model (this takes around 3 minutes on base300M)
samples = None
for x in tqdm(
    sampler.sample_batch_progressive(batch_size=1, model_kwargs=dict(images=[img]))
):
    samples = x
pc = sampler.output_to_point_clouds(samples)[0]





import plotly.express as px

def rgb_to_hex(r: float, g: float, b: float) -> str:
    """Map a [0, 1] RGB to a [0, 255] RGB hex string"""
    return ("#{:02x}{:02x}{:02x}").format(int(r * 255), int(g * 255), int(b * 255))

x, y, z = pc.coords[:, 0], pc.coords[:, 1], pc.coords[:, 2]
colors = [
    rgb_to_hex(r, g, b)
    for r, g, b in zip(pc.channels["R"], pc.channels["G"], pc.channels["B"])
]  # Create a category per color
color_map = {hex: hex for hex in colors}  # Map a color to a category
fig = px.scatter_3d(x=x, y=y, z=z, color=colors, color_discrete_map=color_map)
fig.update_traces(showlegend=False)
fig.show()

