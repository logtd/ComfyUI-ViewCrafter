import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat
from functools import partial
from ..common import (
    checkpoint,
    exists,
    default,
)
from ..basics import zero_module
import comfy.ops
ops = comfy.ops.disable_weight_init
from comfy import model_management
from comfy.ldm.modules.attention import optimized_attention, optimized_attention_masked

if model_management.xformers_enabled():
    import xformers
    import xformers.ops
    XFORMERS_IS_AVAILBLE = True
else:
    XFORMERS_IS_AVAILBLE = False

class RelativePosition(nn.Module):
    """ https://github.com/evelinehong/Transformer_Relative_Position_PyTorch/blob/master/relative_position.py """

    def __init__(self, num_units, max_relative_position):
        super().__init__()
        self.num_units = num_units
        self.max_relative_position = max_relative_position
        self.embeddings_table = nn.Parameter(torch.Tensor(max_relative_position * 2 + 1, num_units))
        nn.init.xavier_uniform_(self.embeddings_table)

    def forward(self, length_q, length_k):
        device = self.embeddings_table.device
        range_vec_q = torch.arange(length_q, device=device)
        range_vec_k = torch.arange(length_k, device=device)
        distance_mat = range_vec_k[None, :] - range_vec_q[:, None]
        distance_mat_clipped = torch.clamp(distance_mat, -self.max_relative_position, self.max_relative_position)
        final_mat = distance_mat_clipped + self.max_relative_position
        final_mat = final_mat.long()
        embeddings = self.embeddings_table[final_mat]
        return embeddings


# TODO Add native Comfy optimized attention.
class CrossAttention(nn.Module):

    def __init__(
                self, 
                query_dim, 
                context_dim=None, 
                heads=8, 
                dim_head=64, 
                dropout=0., 
                relative_position=False, 
                temporal_length=None, 
                video_length=None, 
                image_cross_attention=False, 
                image_cross_attention_scale=1.0, 
                image_cross_attention_scale_learnable=False, 
                text_context_len=77,
                device=None,
                dtype=None,
                operations=ops
            ):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        self.scale = dim_head**-0.5
        self.heads = heads
        self.dim_head = dim_head
        
        self.to_q = operations.Linear(query_dim, inner_dim, bias=False, device=device, dtype=dtype)
        self.to_k = operations.Linear(context_dim, inner_dim, bias=False, device=device, dtype=dtype) 
        self.to_v = operations.Linear(context_dim, inner_dim, bias=False, device=device, dtype=dtype)

        self.to_out = nn.Sequential(
            operations.Linear(inner_dim, query_dim, device=device, dtype=dtype), 
            nn.Dropout(dropout)
        )
        
        self.relative_position = relative_position
        if self.relative_position:
            assert(temporal_length is not None)
            self.relative_position_k = RelativePosition(num_units=dim_head, max_relative_position=temporal_length)
            self.relative_position_v = RelativePosition(num_units=dim_head, max_relative_position=temporal_length)
        else:
            ## only used for spatial attention, while NOT for temporal attention
            if XFORMERS_IS_AVAILBLE and temporal_length is None:
                self.forward = self.efficient_forward
            else:
                self.forward = self.comfy_efficient_forward

        self.image_cross_attention = image_cross_attention
        self.image_cross_attention_scale = image_cross_attention_scale
        self.text_context_len = text_context_len
        self.image_cross_attention_scale_learnable = image_cross_attention_scale_learnable
        if self.image_cross_attention:
            self.to_k_ip = operations.Linear(context_dim, inner_dim, bias=False, device=device, dtype=dtype)
            self.to_v_ip = operations.Linear(context_dim, inner_dim, bias=False, device=device, dtype=dtype)
            if image_cross_attention_scale_learnable:
                self.register_parameter('alpha', nn.Parameter(torch.tensor(0.)) )

    def comfy_efficient_forward(self, x, context=None, mask=None, *args, **kwargs):
        spatial_self_attn = (context is None)
        k_ip, v_ip, out_ip = None, None, None

        h = self.heads
        q = self.to_q(x)
        context = default(context, x)

        if self.image_cross_attention and not spatial_self_attn:
            context, context_image = context[:,:self.text_context_len,:], context[:,self.text_context_len:,:]
            k = self.to_k(context)
            v = self.to_v(context)
            k_ip = self.to_k_ip(context_image)
            v_ip = self.to_v_ip(context_image)
        else:
            if not spatial_self_attn:
                context = context[:,:self.text_context_len,:]
            k = self.to_k(context)
            v = self.to_v(context)

        out = optimized_attention(q, k, v, h)

        ## for image cross-attention
        if k_ip is not None:
            q = rearrange(q, 'b n (h d) -> (b h) n d', h=h)
            k_ip, v_ip = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (k_ip, v_ip))
            sim_ip =  torch.einsum('b i d, b j d -> b i j', q, k_ip) * self.scale
            del k_ip
            sim_ip = sim_ip.softmax(dim=-1)
            out_ip = torch.einsum('b i j, b j d -> b i d', sim_ip, v_ip)
            out_ip = rearrange(out_ip, '(b h) n d -> b n (h d)', h=h)

        if out_ip is not None:
            if self.image_cross_attention_scale_learnable:
                out = out + self.image_cross_attention_scale * out_ip * (torch.tanh(self.alpha)+1)
            else:
                out = out + self.image_cross_attention_scale * out_ip

        return self.to_out(out)
            
    def forward(self, x, context=None, mask=None):
        spatial_self_attn = (context is None)
        k_ip, v_ip, out_ip = None, None, None

        h = self.heads
        q = self.to_q(x)
        context = default(context, x)

        if self.image_cross_attention and not spatial_self_attn:
            context, context_image = context[:,:self.text_context_len,:], context[:,self.text_context_len:,:]
            k = self.to_k(context)
            v = self.to_v(context)
            k_ip = self.to_k_ip(context_image)
            v_ip = self.to_v_ip(context_image)
        else:

            # Assumed Spatial Attention (b c h w)
            if not spatial_self_attn:
                context = context[:,:self.text_context_len,:]
            k = self.to_k(context)
            v = self.to_v(context)


        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        sim = torch.einsum('b i d, b j d -> b i j', q, k) * self.scale
        if self.relative_position:
            len_q, len_k, len_v = q.shape[1], k.shape[1], v.shape[1]
            k2 = self.relative_position_k(len_q, len_k)
            sim2 = einsum('b t d, t s d -> b t s', q, k2) * self.scale # TODO check 
            sim += sim2
        del k

        if exists(mask):
            ## feasible for causal attention mask only
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b i j -> (b h) i j', h=h)
            sim.masked_fill_(~(mask>0.5), max_neg_value)

        # attention, what we cannot get enough of
        sim = sim.softmax(dim=-1)

        out = torch.einsum('b i j, b j d -> b i d', sim, v)
        if self.relative_position:
            v2 = self.relative_position_v(len_q, len_v)
            out2 = einsum('b t s, t s d -> b t d', sim, v2) # TODO check
            out += out2
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)


        ## for image cross-attention
        if k_ip is not None:
            k_ip, v_ip = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (k_ip, v_ip))
            sim_ip =  torch.einsum('b i d, b j d -> b i j', q, k_ip) * self.scale
            del k_ip
            sim_ip = sim_ip.softmax(dim=-1)
            out_ip = torch.einsum('b i j, b j d -> b i d', sim_ip, v_ip)
            out_ip = rearrange(out_ip, '(b h) n d -> b n (h d)', h=h)


        if out_ip is not None:
            if self.image_cross_attention_scale_learnable:
                out = out + self.image_cross_attention_scale * out_ip * (torch.tanh(self.alpha)+1)
            else:
                out = out + self.image_cross_attention_scale * out_ip
        
        return self.to_out(out)
    
    def efficient_forward(self, x, context=None, value=None, mask=None):
        spatial_self_attn = (context is None)
        k_ip, v_ip, out_ip = None, None, None

        q = self.to_q(x)
        context = default(context, x)

        if self.image_cross_attention and not spatial_self_attn:
            context, context_image = context[:,:self.text_context_len,:], context[:,self.text_context_len:,:]
            k = self.to_k(context)
            v = self.to_v(context)
            k_ip = self.to_k_ip(context_image)
            v_ip = self.to_v_ip(context_image)
        else:
            if not spatial_self_attn:
                context = context[:,:self.text_context_len,:]
            k = self.to_k(context)
            v = self.to_v(context)

        b, _, _ = q.shape
        q, k, v = map(
            lambda t: t.unsqueeze(3)
            .reshape(b, t.shape[1], self.heads, self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b * self.heads, t.shape[1], self.dim_head)
            .contiguous(),
            (q, k, v),
        )
        # actually compute the attention, what we cannot get enough of
        out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=None, op=None)
        
        ## for image cross-attention
        if k_ip is not None:
            k_ip, v_ip = map(
                lambda t: t.unsqueeze(3)
                .reshape(b, t.shape[1], self.heads, self.dim_head)
                .permute(0, 2, 1, 3)
                .reshape(b * self.heads, t.shape[1], self.dim_head)
                .contiguous(),
                (k_ip, v_ip),
            )
            out_ip = xformers.ops.memory_efficient_attention(q, k_ip, v_ip, attn_bias=None, op=None)
            out_ip = (
                out_ip.unsqueeze(0)
                .reshape(b, self.heads, out.shape[1], self.dim_head)
                .permute(0, 2, 1, 3)
                .reshape(b, out.shape[1], self.heads * self.dim_head)
            )

        if exists(mask):
            raise NotImplementedError
        out = (
            out.unsqueeze(0)
            .reshape(b, self.heads, out.shape[1], self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b, out.shape[1], self.heads * self.dim_head)
        )
        if out_ip is not None:
            if self.image_cross_attention_scale_learnable:
                out = out + self.image_cross_attention_scale * out_ip * (torch.tanh(self.alpha)+1)
            else:
                out = out + self.image_cross_attention_scale * out_ip
           
        return self.to_out(out)


class BasicTransformerBlock(nn.Module):

    def __init__(
        self, 
        dim, 
        n_heads, 
        d_head, 
        dropout=0., 
        context_dim=None, 
        gated_ff=True, 
        checkpoint=True,
        disable_self_attn=False, 
        attention_cls=None, 
        video_length=None, 
        inner_dim=None,
        image_cross_attention=False, 
        image_cross_attention_scale=1.0, 
        image_cross_attention_scale_learnable=False, 
        switch_temporal_ca_to_sa=False,
        text_context_len=77,
        ff_in=None,
        device=None,
        dtype=None,
        operations=ops
    ):
        super().__init__()
        attn_cls = CrossAttention if attention_cls is None else attention_cls

        self.ff_in = ff_in or inner_dim is not None
        if self.ff_in:
            self.norm_in = operations.LayerNorm(dim, dtype=dtype, device=device)
            self.ff_in = FeedForward(
                dim, 
                dim_out=inner_dim, 
                dropout=dropout, 
                glu=gated_ff, 
                dtype=dtype, 
                device=device, 
                operations=operations
            )
        if inner_dim is None:
            inner_dim = dim

        self.is_res = inner_dim == dim
        self.disable_self_attn = disable_self_attn
        self.attn1 = attn_cls(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout,
            context_dim=None, device=device, dtype=dtype)
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff, device=device, dtype=dtype)
        self.attn2 = attn_cls(
            query_dim=dim, 
            context_dim=context_dim, 
            heads=n_heads, 
            dim_head=d_head, 
            dropout=dropout, 
            video_length=video_length, 
            image_cross_attention=image_cross_attention, 
            image_cross_attention_scale=image_cross_attention_scale, 
            image_cross_attention_scale_learnable=image_cross_attention_scale_learnable,
            text_context_len=text_context_len,
            device=device,
            dtype=dtype
        )
        self.image_cross_attention = image_cross_attention

        self.norm1 = operations.LayerNorm(dim, device=device, dtype=dtype)
        self.norm2 = operations.LayerNorm(dim, device=device, dtype=dtype)
        self.norm3 = operations.LayerNorm(dim, device=device, dtype=dtype)

        self.n_heads = n_heads
        self.d_head = d_head
        self.checkpoint = checkpoint
        self.switch_temporal_ca_to_sa = switch_temporal_ca_to_sa

    def forward(self, x, context=None, mask=None, **kwargs):
        ## implementation tricks: because checkpointing doesn't support non-tensor (e.g. None or scalar) arguments
        input_tuple = (x,)      ## should not be (x), otherwise *input_tuple will decouple x into multiple arguments
        if context is not None:
            input_tuple = (x, context)
        if mask is not None:
            forward_mask = partial(self._forward, mask=mask)
            return checkpoint(forward_mask, (x,), self.parameters(), self.checkpoint)
        return checkpoint(self._forward, input_tuple, self.parameters(), self.checkpoint)


    def _forward(self, x, context=None, mask=None, transformer_options={}):
        extra_options = {}
        block = transformer_options.get("block", None)
        block_index = transformer_options.get("block_index", 0)
        transformer_patches = {}
        transformer_patches_replace = {}

        for k in transformer_options:
            if k == "patches":
                transformer_patches = transformer_options[k]
            elif k == "patches_replace":
                transformer_patches_replace = transformer_options[k]
            else:
                extra_options[k] = transformer_options[k]

        extra_options["n_heads"] = self.n_heads
        extra_options["dim_head"] = self.d_head

        if self.ff_in:
            x_skip = x
            x = self.ff_in(self.norm_in(x))
            if self.is_res:
                x += x_skip

        n = self.norm1(x)
        if self.disable_self_attn:
            context_attn1 = context
        else:
            context_attn1 = None
        value_attn1 = None

        if "attn1_patch" in transformer_patches:
            patch = transformer_patches["attn1_patch"]
            if context_attn1 is None:
                context_attn1 = n
            value_attn1 = context_attn1
            for p in patch:
                n, context_attn1, value_attn1 = p(n, context_attn1, value_attn1, extra_options)

        if block is not None:
            transformer_block = (block[0], block[1], block_index)
        else:
            transformer_block = None
        attn1_replace_patch = transformer_patches_replace.get("attn1", {})
        block_attn1 = transformer_block
        if block_attn1 not in attn1_replace_patch:
            block_attn1 = block

        if block_attn1 in attn1_replace_patch:
            if context_attn1 is None:
                context_attn1 = n
                value_attn1 = n
            n = self.attn1.to_q(n)
            context_attn1 = self.attn1.to_k(context_attn1)
            value_attn1 = self.attn1.to_v(value_attn1)
            n = attn1_replace_patch[block_attn1](n, context_attn1, value_attn1, extra_options)
            n = self.attn1.to_out(n)
        else:
            n = self.attn1(n, context=context_attn1)

        if "attn1_output_patch" in transformer_patches:
            patch = transformer_patches["attn1_output_patch"]
            for p in patch:
                n = p(n, extra_options)

        x += n
        if "middle_patch" in transformer_patches:
            patch = transformer_patches["middle_patch"]
            for p in patch:
                x = p(x, extra_options)

        if self.attn2 is not None:
            n = self.norm2(x)
            if self.switch_temporal_ca_to_sa:
                context_attn2 = n
            else:
                context_attn2 = context
            value_attn2 = None
            if "attn2_patch" in transformer_patches:
                patch = transformer_patches["attn2_patch"]
                value_attn2 = context_attn2
                for p in patch:
                    n, context_attn2, value_attn2 = p(n, context_attn2, value_attn2, extra_options)

            attn2_replace_patch = transformer_patches_replace.get("attn2", {})
            block_attn2 = transformer_block
            if block_attn2 not in attn2_replace_patch:
                block_attn2 = block

            if block_attn2 in attn2_replace_patch:
                if value_attn2 is None:
                    value_attn2 = context_attn2
                n = self.attn2.to_q(n)
                context_attn2 = self.attn2.to_k(context_attn2)
                value_attn2 = self.attn2.to_v(value_attn2)
                n = attn2_replace_patch[block_attn2](n, context_attn2, value_attn2, extra_options)
                n = self.attn2.to_out(n)
            else:
                n = self.attn2(n, context=context_attn2, value=value_attn2)

        if "attn2_output_patch" in transformer_patches:
            patch = transformer_patches["attn2_output_patch"]
            for p in patch:
                n = p(n, extra_options)

        x += n
        if self.is_res:
            x_skip = x
        x = self.ff(self.norm3(x))
        if self.is_res:
            x += x_skip

        return x


class SpatialTransformer(nn.Module):
    """
    Transformer block for image-like data in spatial axis.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    NEW: use_linear for more efficiency instead of the 1x1 convs
    """

    def __init__(
        self, 
        in_channels, 
        n_heads, 
        d_head, 
        depth=1, 
        dropout=0., 
        context_dim=None,
        use_checkpoint=True, 
        disable_self_attn=False, 
        use_linear=False, 
        video_length=None,
        image_cross_attention=False, 
        image_cross_attention_scale_learnable=False,
        device=None,
        dtype=None,
        operations=ops
    ):
        super().__init__()
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = operations.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True, device=device, dtype=dtype)
        if not use_linear:
            self.proj_in = operations.Conv2d(in_channels, inner_dim, kernel_size=1, stride=1, padding=0, device=device, dtype=dtype)
        else:
            self.proj_in = operations.Linear(in_channels, inner_dim, device=device, dtype=dtype)

        attention_cls = None
        self.transformer_blocks = nn.ModuleList([
            BasicTransformerBlock(
                inner_dim,
                n_heads,
                d_head,
                dropout=dropout,
                context_dim=context_dim,
                disable_self_attn=disable_self_attn,
                checkpoint=use_checkpoint,
                attention_cls=attention_cls,
                video_length=video_length,
                image_cross_attention=image_cross_attention,
                image_cross_attention_scale_learnable=image_cross_attention_scale_learnable,
                device=device,
                dtype=dtype
                ) for d in range(depth)
        ])
        if not use_linear:
            self.proj_out = zero_module(operations.Conv2d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0, device=device, dtype=dtype))
        else:
            self.proj_out = zero_module(operations.Linear(inner_dim, in_channels, device=device, dtype=dtype))
        self.use_linear = use_linear

    def forward(self, x, context=None, transformer_options={}, **kwargs):
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        if not self.use_linear:
            x = self.proj_in(x)
        x = rearrange(x, 'b c h w -> b (h w) c').contiguous()
        if self.use_linear:
            x = self.proj_in(x)
        for i, block in enumerate(self.transformer_blocks):
            transformer_options['block_index'] = i
            x = block(x, context=context, **kwargs)
        if self.use_linear:
            x = self.proj_out(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w).contiguous()
        if not self.use_linear:
            x = self.proj_out(x)
        return x + x_in
    
    
class TemporalTransformer(nn.Module):
    """
    Transformer block for image-like data in temporal axis.
    First, reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    """
    def __init__(
        self, 
        in_channels, 
        n_heads, 
        d_head, 
        depth=1, 
        dropout=0., 
        context_dim=None,
        use_checkpoint=True, 
        use_linear=False, 
        only_self_att=True, 
        causal_attention=False, 
        causal_block_size=1,
        relative_position=False, 
        temporal_length=None,
        device=None,
        dtype=None,
        operations=ops
    ):
        super().__init__()
        self.only_self_att = only_self_att
        self.relative_position = relative_position
        self.causal_attention = causal_attention
        self.causal_block_size = causal_block_size

        if only_self_att:
            context_dim = None
            
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = operations.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True, device=device, dtype=dtype)
        self.proj_in = nn.Conv1d(in_channels, inner_dim, kernel_size=1, stride=1, padding=0).to(device, dtype)
        if not use_linear:
            self.proj_in = nn.Conv1d(in_channels, inner_dim, kernel_size=1, stride=1, padding=0).to(device, dtype)
        else:
            self.proj_in = operations.Linear(in_channels, inner_dim, device=device, dtype=dtype)

        if relative_position:
            assert(temporal_length is not None)
            attention_cls = partial(CrossAttention, relative_position=True, temporal_length=temporal_length, device=device, dtype=dtype)
        else:
            attention_cls = partial(CrossAttention, temporal_length=temporal_length, device=device, dtype=dtype)
        if self.causal_attention:
            assert(temporal_length is not None)
            self.mask = torch.tril(torch.ones([1, temporal_length, temporal_length]))

        if self.only_self_att:
            context_dim = None
        self.transformer_blocks = nn.ModuleList([
            BasicTransformerBlock(
                inner_dim,
                n_heads,
                d_head,
                dropout=dropout,
                context_dim=context_dim,
                attention_cls=attention_cls,
                checkpoint=use_checkpoint,
                device=device,
                dtype=dtype
            ) for d in range(depth)
        ])
        if not use_linear:
            self.proj_out = zero_module(nn.Conv1d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0).to(device, dtype))
        else:
            self.proj_out = zero_module(operations.Linear(inner_dim, in_channels, device=device, dtype=dtype))
        self.use_linear = use_linear

    def forward(self, x, context=None):
        b, c, t, h, w = x.shape
        x_in = x
        x = self.norm(x)
        x = rearrange(x, 'b c t h w -> (b h w) c t').contiguous()
        if not self.use_linear:
            x = self.proj_in(x)
        x = rearrange(x, 'bhw c t -> bhw t c').contiguous()
        if self.use_linear:
            x = self.proj_in(x)

        temp_mask = None
        if self.causal_attention:
            # slice the from mask map
            temp_mask = self.mask[:,:t,:t].to(x.device)

        if temp_mask is not None:
            mask = temp_mask.to(x.device)
            mask = repeat(mask, 'l i j -> (l bhw) i j', bhw=b*h*w)
        else:
            mask = None

        if self.only_self_att:
            ## note: if no context is given, cross-attention defaults to self-attention
            for i, block in enumerate(self.transformer_blocks):
                x = block(x, mask=mask)
            x = rearrange(x, '(b hw) t c -> b hw t c', b=b).contiguous()
        else:
            x = rearrange(x, '(b hw) t c -> b hw t c', b=b).contiguous()
            context = rearrange(context, '(b t) l con -> b t l con', t=t).contiguous()
            for i, block in enumerate(self.transformer_blocks):
                # calculate each batch one by one (since number in shape could not greater then 65,535 for some package)
                for j in range(b):
                    context_j = repeat(
                        context[j],
                        't l con -> (t r) l con', r=(h * w) // t, t=t).contiguous()
                    ## note: causal mask will not applied in cross-attention case
                    x[j] = block(x[j], context=context_j)
        
        if self.use_linear:
            x = self.proj_out(x)
            x = rearrange(x, 'b (h w) t c -> b c t h w', h=h, w=w).contiguous()
        if not self.use_linear:
            x = rearrange(x, 'b hw t c -> (b hw) c t').contiguous()
            x = self.proj_out(x)
            x = rearrange(x, '(b h w) c t -> b c t h w', b=b, h=h, w=w).contiguous()

        return x + x_in
    

class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out, device=None, dtype=None, operations=ops):
        super().__init__()
        self.proj = operations.Linear(dim_in, dim_out * 2, device=device, dtype=dtype)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0., device=None, dtype=None, operations=ops):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(
            operations.Linear(dim, inner_dim, device=device, dtype=dtype),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim, device=device, dtype=dtype)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            operations.Linear(inner_dim, dim_out, device=device, dtype=dtype)
        )

    def forward(self, x):
        return self.net(x)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32, device=None, dtype=None, operations=ops):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = operations.Conv2d(dim, hidden_dim * 3, 1, bias = False, device=device, dtype=dtype)
        self.to_out = operations.Conv2d(hidden_dim, dim, 1, device=device, dtype=dtype)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)', heads = self.heads, qkv=3)
        k = k.softmax(dim=-1)  
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', heads=self.heads, h=h, w=w)
        return self.to_out(out)


class SpatialSelfAttention(nn.Module):
    def __init__(self, in_channels, device=None, dtype=None, operations=ops):
        super().__init__()
        self.in_channels = in_channels

        self.norm = operations.GroupNorm(
            num_groups=32, 
            num_channels=in_channels, 
            eps=1e-6, 
            affine=True, 
            device=device, 
            dtype=dtype
        )
        self.q = operations.Conv2d(
                                in_channels,
                                in_channels,
                                kernel_size=1,
                                stride=1,
                                padding=0,
                                device=device,
                                dtype=dtype
                            )
        self.k = operations.Conv2d(
                                in_channels,
                                in_channels,
                                kernel_size=1,
                                stride=1,
                                padding=0,
                                device=device,
                                dtype=dtype
                            )
        self.v = operations.Conv2d(
                                in_channels,
                                in_channels,
                                kernel_size=1,
                                stride=1,
                                padding=0,
                                device=device,
                                dtype=dtype
                            )
        self.proj_out = operations.Conv2d(
                                in_channels,
                                in_channels,
                                kernel_size=1,
                                stride=1,
                                padding=0,
                                device=device,
                                dtype=dtype
                            )

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b,c,h,w = q.shape
        q = rearrange(q, 'b c h w -> b (h w) c')
        k = rearrange(k, 'b c h w -> b c (h w)')
        w_ = torch.einsum('bij,bjk->bik', q, k)

        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = rearrange(v, 'b c h w -> b c (h w)')
        w_ = rearrange(w_, 'b i j -> b j i')
        h_ = torch.einsum('bij,bjk->bik', v, w_)
        h_ = rearrange(h_, 'b c (h w) -> b c h w', h=h)
        h_ = self.proj_out(h_)

        return x+h_
