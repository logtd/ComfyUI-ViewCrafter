import torch
from comfy.clip_vision import clip_preprocess

from comfy import model_management


class ModelFunctionWrapper(torch.nn.Module):
    def __init__(self, concat_latents, stride, img_embs) -> None:
        super().__init__()
        self.concat_latents = torch.cat([concat_latents]*2)
        self.stride = stride
        self.img_embs = img_embs

    def to(self, device):
        self.concat_latents = self.concat_latents.to(device)
        if self.stride is not None:
            self.stride = self.stride.to(device)
        self.img_embs = self.img_embs.to(device)
        return super().to(device)

    def forward(self, apply_model_func, kwargs):
        x, t, model_in_kwargs, cond_list =  kwargs['input'], kwargs['timestep'], kwargs['c'], kwargs['cond_or_uncond']
        c_crossattn = model_in_kwargs.pop("c_crossattn")
        stride = self.stride
        if self.stride is not None:
            stride = torch.cat([self.stride] * len(cond_list))
        img_embs = []
        for cond in cond_list:
            img_embed = self.img_embs[cond]
            if len(img_embed) == 1:
                img_embed = img_embed.repeat(len(x)//len(cond_list), 1, 1)
            img_embs.append(img_embed)
        img_embs = torch.cat(img_embs)
        
        return apply_model_func(
            x, 
            t=t,
            c_crossattn=c_crossattn,
            c_latents=self.concat_latents[:len(x)],
            img_emb=img_embs,
            fs=stride,
            **model_in_kwargs
        )


class ApplyViewCrafterNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", ),
                "clip_vision": ("CLIP_VISION", ),
                "latents": ("LATENT", ),
                "image_proj_model": ("IMAGE_PROJ_MODEL", ),
                "clip_image": ("IMAGE", ),
                "stride": ("INT", {"default": 10, "min": 0, "max": 30, "step": 1}, ),
                "scale_image": ("BOOLEAN", { "default": False })
            },
        }
        
    CATEGORY = "viewcrafter"
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)

    FUNCTION = "process_image_conditioning"

    def process_image_conditioning(
        self, 
        model, 
        clip_vision, 
        latents, 
        image_proj_model, 
        clip_image, 
        stride,
        scale_image=False,
    ):
        model = model.clone()
        if scale_image:
            clip_image = clip_image*2-1 
        encoded_image = clip_vision.encode_image(clip_image)['last_hidden_state']
        image_emb = image_proj_model(encoded_image)

        encoded_image_uncond  = clip_vision.encode_image(torch.zeros_like(clip_image))['last_hidden_state']
        image_emb_uncond = image_proj_model(encoded_image_uncond)

        latents = model.model.process_latent_in(latents['samples'])
            
        stride = torch.tensor([stride], dtype=torch.long, device=model_management.intermediate_device()) if stride > 0 else None
        
        model.set_model_unet_function_wrapper(ModelFunctionWrapper(
            concat_latents=latents,
            stride=stride,
            img_embs=torch.stack([image_emb, image_emb_uncond])
        ))
        
        return (model,)



class ScaleImagesNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE", ),
            },
        }
        
    CATEGORY = "viewcrafter"
    RETURN_TYPES = ("IMAGE",)

    FUNCTION = "scale_images"

    def scale_images(
        self, 
        images
    ):
  
        images = (images - .5) * 2
        return (images,)
