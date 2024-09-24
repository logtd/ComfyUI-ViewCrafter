import os

import torch
import comfy
import folder_paths

from comfy import model_base, model_management
from comfy_extras.nodes_model_advanced import ModelSamplingDiscrete, RescaleCFG
from ..modules.lvdm.modules.networks.openaimodel3d import UNetModel as DynamiCrafterUNetModel

from ..utils.model_utils import DynamiCrafterBase, DYNAMICRAFTER_CONFIG, \
    load_image_proj_dict, load_viewcrafter_dict, get_image_proj_model

from ..utils.utils import get_models_directory


MODEL_DIR= "viewcrafter"
MODEL_DIR_PATH = os.path.join(folder_paths.models_dir, MODEL_DIR)
os.makedirs(MODEL_DIR_PATH, exist_ok=True)


class LoadViewCrafterNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_path": (get_models_directory(os.listdir(MODEL_DIR_PATH)), ),
                "precision": (["fp16", "fp32"],),
                "zsnr": ("BOOLEAN", { "default": False }),
            },
        }
        
    CATEGORY = "viewcrafter"
    RETURN_TYPES = ("MODEL", "IMAGE_PROJ_MODEL", )
    RETURN_NAMES = ("model", "image_proj_model", )
    FUNCTION = "load_viewcrafter"
    
    def load_model_dicts(self, model_path, precision=torch.float32):
        model_state_dict = comfy.utils.load_torch_file(model_path)
        viewcrafter_dict = load_viewcrafter_dict(model_state_dict)
        image_proj_dict = load_image_proj_dict(model_state_dict)

        return viewcrafter_dict, image_proj_dict

    def get_prediction_type(self, is_eps: bool, model_config):
        if not is_eps and "image_cross_attention_scale_learnable" in model_config.unet_config.keys():
                model_config.unet_config["image_cross_attention_scale_learnable"] = False

        return model_base.ModelType.EPS if is_eps else model_base.ModelType.V_PREDICTION

    def handle_model_management(self, viewcrafter_dict: dict, model_config, precision):
        parameters = comfy.utils.calculate_parameters(viewcrafter_dict, "model.diffusion_model.")
        load_device = model_management.get_torch_device()

        manual_cast_dtype = model_management.unet_manual_cast(
            precision, 
            load_device, 
            model_config.supported_inference_dtypes
        )
        model_config.set_inference_dtype(precision, precision)
        inital_load_device = model_management.unet_inital_load_device(parameters, precision)

        return load_device, inital_load_device

    def check_leftover_keys(self, state_dict: dict):
        left_over = state_dict.keys()
        if len(left_over) > 0:
            print("left over keys:", left_over)

    def load_viewcrafter(self, model_path, precision, zsnr):
        model_path = os.path.join(MODEL_DIR_PATH, model_path)
        precision = torch.float32 if precision == 'fp32' else torch.float16
        
        config = {
            **DYNAMICRAFTER_CONFIG,
            'dtype': precision
        }
        
        if os.path.exists(model_path):
            viewcrafter_dict, image_proj_dict = self.load_model_dicts(model_path)
            model_config = DynamiCrafterBase(config)

            viewcrafter_dict, is_eps = model_config.process_dict_version(state_dict=viewcrafter_dict)

            MODEL_TYPE = self.get_prediction_type(is_eps, model_config)
            load_device, inital_load_device = self.handle_model_management(viewcrafter_dict, model_config, precision)


            model = model_base.BaseModel(
                model_config, 
                model_type=MODEL_TYPE, 
                device=inital_load_device, 
                unet_model=DynamiCrafterUNetModel
            )
            model.diffusion_model.dtype = precision
            model.to(precision)

            image_proj_model = get_image_proj_model(image_proj_dict)
            model.load_model_weights(viewcrafter_dict, "model.diffusion_model.")
            self.check_leftover_keys(viewcrafter_dict)

            model_patcher = comfy.model_patcher.ModelPatcher(
                model, 
                load_device=load_device, 
                offload_device=model_management.unet_offload_device(), 
            )

        model_patcher, = RescaleCFG().patch(model_patcher, 0.7)
        model_patcher, = ModelSamplingDiscrete().patch(model_patcher, "v_prediction", zsnr)
    
        return (model_patcher, image_proj_model, )
