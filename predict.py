import os
from typing import List

import torch
from cog import BasePredictor, Input, Path
from diffusers import StableDiffusionPipeline

CACHE_DIR = "diffusers-cache"

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        print("Loading pipeline...")
        self.pipe = StableDiffusionPipeline.from_pretrained(
            "lambdalabs/sd-pokemon-diffusers",
            cache_dir=CACHE_DIR,
            torch_dtype=torch.float16,
            revision="6d2fb7c893aac58d79cc17e1b21ac0beeacc8338"
        ).to("cuda")

    @torch.inference_mode()
    @torch.cuda.amp.autocast()
    def predict(
        self,
        prompt: str = Input(description="Input prompt", default=""),
        num_outputs: int = Input(
            description="Number of images to output", choices=[1, 2, 3, 4], default=1
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps", ge=1, le=50, default=25
        ),
        guidance_scale: float = Input(
            description="Scale for classifier-free guidance", ge=1, le=20, default=7.5
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
    ) -> List[Path]:
        """Run a single prediction on the model"""
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        # Safety checker doesn't really work for Pokemon
        def null_safety(images, **kwargs):
            return images, False
        self.pipe.safety_checker = null_safety

        generator = torch.Generator("cuda").manual_seed(seed)
        output = self.pipe(
            prompt=[prompt] * num_outputs,
            guidance_scale=guidance_scale,
            generator=generator,
            num_inference_steps=num_inference_steps,
        )

        output_paths = []
        for i, sample in enumerate(output["sample"]):
            output_path = f"/tmp/out-{i}.png"
            sample.save(output_path)
            output_paths.append(Path(output_path))

        return output_paths
