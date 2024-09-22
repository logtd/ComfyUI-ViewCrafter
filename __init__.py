from .nodes.apply_viewcrafter_node import ApplyViewCrafterNode, ScaleImagesNode
from .nodes.load_viewcrafter_node import LoadViewCrafterNode


NODE_CLASS_MAPPINGS = {
    "ApplyViewCrafter": ApplyViewCrafterNode,
    "LoadViewCrafter": LoadViewCrafterNode,
    # "ScaleImages": ScaleImagesNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ApplyViewCrafter": "Apply ViewCrafter",    
    "LoadViewCrafter": "Load ViewCrafter",
    # "ScaleImages": "Scale Images"
}
