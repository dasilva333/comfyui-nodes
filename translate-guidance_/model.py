#Original code can be found on: https://github.com/black-forest-labs/flux

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from einops import rearrange, repeat
import comfy.ldm.common_dit

import math

def inverted_cosine_bubble_ripsaw(timestep, guidance, device, adjust_normal_cfg=True):
    # Ensure timestep is [0,1].
    t = timestep.to(device, dtype=torch.float32).clamp(0.0, 1.0)

    # Original guidance as float on the correct device
    g = guidance.to(device, dtype=torch.float32)
    g2 = None
    transformed2 = None

    # If we have more than one channel and want to adjust normal cfg
    if adjust_normal_cfg and len(g.shape) > 1:
        g = g[:, 0]
        g2 = g[:, 1]

    # Generate random shifts (0 to 8 inclusive)
    # Cast guidance to int and shift, then cast back to float
    g_int = g.to(torch.int64)  # safer for bit-shift
    shift = torch.randint(low=1, high=8, size=g_int.shape, device=device, dtype=torch.int64)
    g_shifted = (g_int << shift).to(torch.float32)

    # Inverted cosine with the shifted guidance
    transformed = 4.5 - torch.cos(math.pi * (g_shifted + t) * 0.9995)

    if g2 is not None:
        g2_int = g2.to(torch.int64)
        shift2 = torch.randint(low=-8, high=-1, size=g2_int.shape, device=device, dtype=torch.int64)
        g2_shifted = (g2_int << shift2).to(torch.float32)
        transformed2 = 4.5 - torch.cos(math.pi * (g2_shifted + t) * 0.9995)

    if g2 is not None and transformed2 is not None:
        transformed = torch.stack([transformed, transformed2], dim=-1)

    # Interpolate between original guidance (g) and the new transformed value
    out = (1.0 - t) * g + (t * transformed)

    # Debug prints
    print("Random bit-shift bubble ripsaw transformation:")
    print("Original guidance:", g)
    print("Bit shifts used:", shift if g2 is None else (shift, shift2))
    print("Guidance after shifting:", g_shifted if g2 is None else (g_shifted, g2_shifted))
    print("Transformed guidance:", transformed)
    print("Linear progress (t):", t)
    print("Resulting guidance:", out)

    return out

def inverted_cosine_ripsaw(timestep, guidance, device, adjust_normal_cfg=True):
    """
    Similar to 'inverted_cosine' but with a random bit-shift ripsaw approach.

    We'll randomly choose an integer shift ∈ [0..8], cast guidance to int, left shift it,
    cast back to float, and use that in our inverted_cosine formula. The result then
    gets linearly blended with the original guidance based on the timestep 't'.

    Args:
      timestep : scalar or 1D tensor in [0,1], representing the normalized progress.
      guidance : scalar or 1D tensor containing the original guidance value(s).
      device   : torch device on which computations will run.
      adjust_normalcfg (bool) : if True and guidance is multi-dimensional, we
                                 adjust the first dimension(s) for dual-step transforms.

    Returns:
      out : The transformed guidance after random bit-shift peaks, mapped via
            an inverted cosine, and linearly interpolated against the original guidance.
    """

    # Ensure timestep is [0,1].
    t = timestep.to(device, dtype=torch.float32).clamp(0.0, 1.0)

    # Original guidance as float on the correct device
    g = guidance.to(device, dtype=torch.float32)
    g2 = None
    transformed2 = None

    # If we have more than one channel and want to adjust normal cfg
    if adjust_normal_cfg and len(g.shape) > 1:
        g = g[:, 0]
        g2 = g[:, 1]

    # Generate random shifts (0 to 8 inclusive)
    # Cast guidance to int and shift, then cast back to float
    g_int = g.to(torch.int64)  # safer for bit-shift
    shift = torch.randint(low=0, high=255, size=g_int.shape, device=device, dtype=torch.int64)
    g_shifted = (g_int << shift).to(torch.float32)

    # Inverted cosine with the shifted guidance
    transformed = 4.5 - torch.cos(math.pi * (g_shifted + t))

    if g2 is not None:
        g2_int = g2.to(torch.int64)
        shift2 = torch.randint(low=0, high=9, size=g2_int.shape, device=device, dtype=torch.int64)
        g2_shifted = (g2_int << shift2).to(torch.float32)
        transformed2 = 4.5 - torch.cos(math.pi * (g2_shifted + t) * 0.95)

    if g2 is not None and transformed2 is not None:
        transformed = torch.stack([transformed, transformed2], dim=-1)

    # Interpolate between original guidance (g) and the new transformed value
    out = (1.0 - t) * g + (t * transformed)

    # Debug prints
    print("Random bit-shift ripsaw transformation:")
    print("Original guidance:", g)
    print("Bit shifts used:", shift if g2 is None else (shift, shift2))
    print("Guidance after shifting:", g_shifted if g2 is None else (g_shifted, g2_shifted))
    print("Transformed guidance:", transformed)
    print("Linear progress (t):", t)
    print("Resulting guidance:", out)

    return out

def translate_guidance(timestep, guidance, device, method="inverted_cosine"):
    """
    Interpolate and transform guidance strength based on a normalized 'timestep' in [0,1].

    Arguments:
      timestep : scalar or 1D tensor in [0,1], representing the normalized progress.
        - 0 => beginning of inference
        - 1 => end of inference
      guidance : scalar or 1D tensor containing the original guidance value(s).
      device   : torch device on which computations will run.
      method   : Which transformation method to use on the guidance.

    The returned guidance = (1 - t)*original_guidance + t*transformed_guidance.
    The difference here is that 'transformed_guidance' itself may depend on `t`,
    so we incorporate `t` both in the transformation and in the final interpolation.

    Possible examples of 'method':
      1) "cosine"          : cos(pi*(g + t))
      2) "inverted_cosine" : 3.5 - cos(pi*(g + t))
      3) "sin"             : sin(pi*(g + t))
      4) "linear_increase" : g + t*1.5
      5) "linear_decrease" : g - t*1.5
      6) "random_noise"    : add uniform noise once, then keep it consistent
      7) "random_gaussian" : add normal(0, std=0.3) noise once
      8) "random_extreme"  : add uniform(-2, 2) noise once

    You can of course tailor these transformations to your training/inference scheme.
    """
    # Ensure timestep is [0,1].
    t = timestep.to(device, dtype=torch.float32).clamp_(0.0, 1.0)

    # Original guidance as float on the correct device
    g = guidance.to(device, dtype=torch.float32)

    # 1) Decide how the *transformed* guidance is computed, including dependence on t
    if method == "cosine":
        # Incorporate t so that the transform changes over time
        # e.g. cos(pi*(g + t)) yields a changing transformation
        transformed = torch.cos(math.pi * (g + t))

    elif method == "inverted_cosine":
        # Shift by 3.5, but also incorporate t so it evolves over time
        # If you use '3.5 - cos(pi*g)', it may remain constant if g is constant.
        # So add t to ensure it shifts each step:
        transformed = 5.0 - torch.cos(math.pi * (g + t))

    elif method == "sin":
        # Use g + t inside the sine for changing transformations
        transformed = torch.sin(math.pi * (g + t))

    elif method == "linear_increase":
        # Let the transformation itself incorporate t. For example:
        transformed = g + (1.5 * t)

    elif method == "linear_decrease":
        transformed = g - (1.5 * t)

    elif method == "random_noise":
        # If you truly want random noise each call, apply it here
        # But be aware that if you keep calling it, you'll get new noise every time
        noise = torch.empty_like(g).uniform_(-0.75, 0.75)
        # Optionally incorporate t in the final scaled noise
        transformed = g + (noise * t)

    elif method == "random_gaussian":
        # Same idea, but Gaussian
        noise = torch.randn_like(g) * 0.3
        transformed = g + (noise * t)

    elif method == "random_extreme":
        noise = torch.empty_like(g).uniform_(-2.0, 2.0)
        transformed = g + (noise * t)

    elif method == "ripsaw":    
        transformed = inverted_cosine_ripsaw(timestep, guidance, device)

    elif method == "bubble":
        transformed = inverted_cosine_bubble_ripsaw(timestep, guidance, device)

    else:
        # Fallback if method is invalid or not recognized
        transformed = g

    # 2) Interpolate between the original guidance (g) and the new transformed value
    #    The factor t in the interpolation means:
    #      - When t=0, you stick to the original g.
    #      - When t=1, you fully adopt 'transformed'.
    #    If the transform also uses t, you get a more pronounced shift over time.
    print("Original guidance:", g)
    print("Transformed guidance:", transformed)
    print("Linear progress (t):", t)
    print("Adjusted Transform:", t * transformed)
    print("Timestep Adjusted:", (1.0 - t) * g)
    print("Resulting guidance:", (1.0 - t) * g + (t * transformed))
    out = (1.0 - t) * g + (t * transformed)

    # You can log for debugging:
    # print("Original guidance:", g)
    # print("Transformed guidance:", transformed)
    # print("Linear progress (t):", t)
    # print("Resulting guidance:", out)

    return out

from .layers import (
    DoubleStreamBlock,
    EmbedND,
    LastLayer,
    MLPEmbedder,
    SingleStreamBlock,
    timestep_embedding,
)

@dataclass
class FluxParams:
    in_channels: int
    out_channels: int
    vec_in_dim: int
    context_in_dim: int
    hidden_size: int
    mlp_ratio: float
    num_heads: int
    depth: int
    depth_single_blocks: int
    axes_dim: list
    theta: int
    patch_size: int
    qkv_bias: bool
    guidance_embed: bool


class Flux(nn.Module):
    """
    Transformer model for flow matching on sequences.
    """

    def __init__(self, image_model=None, final_layer=True, dtype=None, device=None, operations=None, **kwargs):
        super().__init__()
        self.dtype = dtype
        params = FluxParams(**kwargs)
        self.params = params
        self.patch_size = params.patch_size
        self.in_channels = params.in_channels * params.patch_size * params.patch_size
        self.out_channels = params.out_channels * params.patch_size * params.patch_size
        if params.hidden_size % params.num_heads != 0:
            raise ValueError(
                f"Hidden size {params.hidden_size} must be divisible by num_heads {params.num_heads}"
            )
        pe_dim = params.hidden_size // params.num_heads
        if sum(params.axes_dim) != pe_dim:
            raise ValueError(f"Got {params.axes_dim} but expected positional dim {pe_dim}")
        self.hidden_size = params.hidden_size
        self.num_heads = params.num_heads
        self.pe_embedder = EmbedND(dim=pe_dim, theta=params.theta, axes_dim=params.axes_dim)
        self.img_in = operations.Linear(self.in_channels, self.hidden_size, bias=True, dtype=dtype, device=device)
        self.time_in = MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size, dtype=dtype, device=device, operations=operations)
        self.vector_in = MLPEmbedder(params.vec_in_dim, self.hidden_size, dtype=dtype, device=device, operations=operations)
        self.guidance_in = (
            MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size, dtype=dtype, device=device, operations=operations) if params.guidance_embed else nn.Identity()
        )
        self.txt_in = operations.Linear(params.context_in_dim, self.hidden_size, dtype=dtype, device=device)

        self.double_blocks = nn.ModuleList(
            [
                DoubleStreamBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=params.mlp_ratio,
                    qkv_bias=params.qkv_bias,
                    dtype=dtype, device=device, operations=operations
                )
                for _ in range(params.depth)
            ]
        )

        self.single_blocks = nn.ModuleList(
            [
                SingleStreamBlock(self.hidden_size, self.num_heads, mlp_ratio=params.mlp_ratio, dtype=dtype, device=device, operations=operations)
                for _ in range(params.depth_single_blocks)
            ]
        )

        if final_layer:
            self.final_layer = LastLayer(self.hidden_size, 1, self.out_channels, dtype=dtype, device=device, operations=operations)

    def forward_orig(
        self,
        img: Tensor,
        img_ids: Tensor,
        txt: Tensor,
        txt_ids: Tensor,
        timesteps: Tensor,
        y: Tensor,
        guidance: Tensor = None,
        control = None,
        transformer_options={},
        attn_mask: Tensor = None,
    ) -> Tensor:
        patches_replace = transformer_options.get("patches_replace", {})
        if img.ndim != 3 or txt.ndim != 3:
            raise ValueError("Input img and txt tensors must have 3 dimensions.")

        print("Transformer Options:", transformer_options)

        # Running on sequences img
        img = self.img_in(img)
        vec = self.time_in(timestep_embedding(timesteps, 256).to(img.dtype))

        if self.params.guidance_embed:
            if guidance is None:
                raise ValueError("Didn't get guidance strength for guidance distilled model.")

            # Check if guidance_method is set in transformer_options
            guidance_method = transformer_options.get("guidance_method", None)
            if guidance_method:
                print("Using guidance method:", guidance_method, "for guidance:", guidance)
                method_guidance = translate_guidance(timesteps, guidance, img.device, guidance_method)
                vec = vec + self.guidance_in(timestep_embedding(method_guidance, 256).to(img.dtype))
            else:
                vec = vec + self.guidance_in(timestep_embedding(guidance, 256).to(img.dtype))

        vec = vec + self.vector_in(y[:,:self.params.vec_in_dim])
        txt = self.txt_in(txt)

        ids = torch.cat((txt_ids, img_ids), dim=1)
        pe = self.pe_embedder(ids)

        blocks_replace = patches_replace.get("dit", {})
        for i, block in enumerate(self.double_blocks):
            if ("double_block", i) in blocks_replace:
                def block_wrap(args):
                    out = {}
                    out["img"], out["txt"] = block(img=args["img"],
                                                   txt=args["txt"],
                                                   vec=args["vec"],
                                                   pe=args["pe"],
                                                   attn_mask=args.get("attn_mask"))
                    return out

                out = blocks_replace[("double_block", i)]({"img": img,
                                                           "txt": txt,
                                                           "vec": vec,
                                                           "pe": pe,
                                                           "attn_mask": attn_mask},
                                                          {"original_block": block_wrap})
                txt = out["txt"]
                img = out["img"]
            else:
                img, txt = block(img=img,
                                 txt=txt,
                                 vec=vec,
                                 pe=pe,
                                 attn_mask=attn_mask)

            if control is not None: # Controlnet
                control_i = control.get("input")
                if i < len(control_i):
                    add = control_i[i]
                    if add is not None:
                        img += add

        img = torch.cat((txt, img), 1)

        for i, block in enumerate(self.single_blocks):
            if ("single_block", i) in blocks_replace:
                def block_wrap(args):
                    out = {}
                    out["img"] = block(args["img"],
                                       vec=args["vec"],
                                       pe=args["pe"],
                                       attn_mask=args.get("attn_mask"))
                    return out

                out = blocks_replace[("single_block", i)]({"img": img,
                                                           "vec": vec,
                                                           "pe": pe,
                                                           "attn_mask": attn_mask}, 
                                                          {"original_block": block_wrap})
                img = out["img"]
            else:
                img = block(img, vec=vec, pe=pe, attn_mask=attn_mask)

            if control is not None: # Controlnet
                control_o = control.get("output")
                if i < len(control_o):
                    add = control_o[i]
                    if add is not None:
                        img[:, txt.shape[1] :, ...] += add

        img = img[:, txt.shape[1] :, ...]

        img = self.final_layer(img, vec)  # (N, T, patch_size ** 2 * out_channels)
        return img

    def forward(self, x, timestep, context, y, guidance, control=None, transformer_options={}, **kwargs):
        bs, c, h, w = x.shape
        patch_size = self.patch_size
        x = comfy.ldm.common_dit.pad_to_patch_size(x, (patch_size, patch_size))

        img = rearrange(x, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch_size, pw=patch_size)

        h_len = ((h + (patch_size // 2)) // patch_size)
        w_len = ((w + (patch_size // 2)) // patch_size)
        img_ids = torch.zeros((h_len, w_len, 3), device=x.device, dtype=x.dtype)
        img_ids[:, :, 1] = img_ids[:, :, 1] + torch.linspace(0, h_len - 1, steps=h_len, device=x.device, dtype=x.dtype).unsqueeze(1)
        img_ids[:, :, 2] = img_ids[:, :, 2] + torch.linspace(0, w_len - 1, steps=w_len, device=x.device, dtype=x.dtype).unsqueeze(0)
        img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)

        txt_ids = torch.zeros((bs, context.shape[1], 3), device=x.device, dtype=x.dtype)
        out = self.forward_orig(img, img_ids, context, txt_ids, timestep, y, guidance, control, transformer_options, attn_mask=kwargs.get("attention_mask", None))
        return rearrange(out, "b (h w) (c ph pw) -> b c (h ph) (w pw)", h=h_len, w=w_len, ph=2, pw=2)[:,:,:h,:w]
