import node_helpers

class CLIPTextEncodeFlux:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "clip": ("CLIP", ),
            "clip_l": ("STRING", {"multiline": True, "dynamicPrompts": True}),
            "t5xxl": ("STRING", {"multiline": True, "dynamicPrompts": True}),
            "guidance": ("FLOAT", {"default": 3.5, "min": 0.0, "max": 100.0, "step": 0.1}),
            }}
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "encode"

    CATEGORY = "advanced/conditioning/flux"

    def encode(self, clip, clip_l, t5xxl, guidance):
        tokens = clip.tokenize(clip_l)
        tokens["t5xxl"] = clip.tokenize(t5xxl)["t5xxl"]

        return (clip.encode_from_tokens_scheduled(tokens, add_dict={"guidance": guidance}), )

class FluxGuidance:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "conditioning": ("CONDITIONING", ),
            "guidance": ("FLOAT", {"default": 3.5, "min": 0.0, "max": 100.0, "step": 0.1}),
            }}

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "append"

    CATEGORY = "advanced/conditioning/flux"

    def append(self, conditioning, guidance):
        c = node_helpers.conditioning_set_values(conditioning, {"guidance": guidance})
        return (c, )

class TranslateGuidance:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "guidance_method": ([
                    "cosine",
                    "inverted_cosine",
                    "sin",
                    "linear_increase",
                    "linear_decrease",
                    "random_noise",
                    "random_gaussian",
                    "random_extreme",
                    "ripsaw",
                    "bubble"
                ],),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply_transformer_options"

    CATEGORY = "advanced/model"

    def apply_transformer_options(self, model, guidance_method):
        # Clone the model to avoid modifying the original instance
        m = model.clone()

        # Save the original `forward_orig` if it exists and hasn't already been patched
        print("forward_orig has been patched")

        # Replace `forward_orig` with the custom implementation
        # def inline_custom_forward_orig(*args, **kwargs):
        #     print("Custom forward_orig has been called")
        #     return 0

        # m.model.forward_orig = inline_custom_forward_orig
        # model.add_object_patch("forward_orig", inline_custom_forward_orig)

        print("guidance_method:", guidance_method)
        print("model:", m.model_options)

        # Ensure model_options exists
        if not hasattr(m, "model_options") or not isinstance(m.model_options, dict):
            print("Model does not have 'model_options'. Initializing 'model_options'.")
            m.model_options = {}

        # Check if transformer_options exists and initialize or update it
        if "transformer_options" in m.model_options:
            print("Model already has 'transformer_options' key.")
            print("Current transformer_options:", m.model_options["transformer_options"])
        else:
            print("'transformer_options' key does not exist. Initializing it.")
            m.model_options["transformer_options"] = {}

        # Update transformer_options with the selected guidance method
        m.model_options["transformer_options"].update({"guidance_method": guidance_method})

        # Logging the final transformer_options
        print("Updated transformer_options:", m.model_options["transformer_options"])

        return (m,)


NODE_CLASS_MAPPINGS = {
    "CLIPTextEncodeFlux": CLIPTextEncodeFlux,
    "FluxGuidance": FluxGuidance,
    "TranslateGuidance": TranslateGuidance,
}
