# Extracted for the Mona-only KernelAdapter export, 2026-10-14.
"""Frozen Swin backbone with two complete Mona adapters per block."""
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from .swin_core import (PatchEmbedding, PatchMerging, MLP, DropPath,
                        window_partition, window_reverse, tokens_to_grid, grid_to_tokens)
from .mona import MonaAdapter


class Attention(nn.Module):
    def __init__(self, dim, heads, window, dropout=0., attention_dropout=0.):
        super().__init__()
        if dim%heads:
            raise ValueError("Channels must be divisible by heads")
        self.heads,self.window,self.scale=heads,window,(dim//heads)**-.5
        self.qkv = nn.Linear(dim,dim*3)
        self.projection = nn.Linear(dim,dim)
        self.attention_dropout=nn.Dropout(attention_dropout)
        self.projection_dropout=nn.Dropout(dropout)
        self.relative_position_bias_table = nn.Parameter(torch.zeros((2*window-1)**2,heads))
        coords=torch.stack(torch.meshgrid(torch.arange(window),torch.arange(window),indexing="ij")).flatten(1)
        relative=(coords[:,:,None]-coords[:,None,:]).permute(1,2,0)
        relative=relative+window-1
        index=relative[:,:,0]*(2*window-1)+relative[:,:,1]
        self.register_buffer("relative_position_index",index,persistent=False)
        nn.init.trunc_normal_(self.relative_position_bias_table,std=.02)

    def forward(self, x, mask):
        b,n,c=x.shape
        q,k,v=self.qkv(x).reshape(b,n,3,self.heads,c//self.heads).permute(2,0,3,1,4).unbind(0)
        logits=(q*self.scale)@k.transpose(-2,-1)
        bias=self.relative_position_bias_table[self.relative_position_index.flatten()].reshape(n,n,-1).permute(2,0,1)
        logits=logits+bias[None]
        if mask is not None:
            count=mask.shape[0]
            logits=(logits.reshape(-1,count,self.heads,n,n)+mask[None,:,None]).reshape(b,self.heads,n,n)
        logits=logits.to(q.dtype)
        out=(self.attention_dropout(logits.softmax(-1))@v).transpose(1,2).reshape(b,n,c)
        return self.projection_dropout(self.projection(out))


class AdaptedBlock(nn.Module):
    def __init__(self,dim,heads,window,shift,rate,adapter_factory,mlp_ratio,dropout,attention_dropout):
        super().__init__()
        self.window,self.shift=window,shift
        self.norm1,self.norm2=nn.LayerNorm(dim),nn.LayerNorm(dim)
        self.attn=Attention(dim,heads,window,dropout,attention_dropout)
        self.mlp=MLP(dim,int(dim*mlp_ratio),dropout)
        self.drop_path=DropPath(rate)
        self.adapter1,self.adapter2=adapter_factory(dim),adapter_factory(dim)

    def mask(self,h,w,device,dtype):
        if not self.shift:
            return None
        mask=torch.zeros((1,h,w,1),device=device,dtype=dtype)
        slices=(slice(0,-self.window),slice(-self.window,-self.shift),slice(-self.shift,None))
        for i,hs in enumerate(slices):
            for j,ws in enumerate(slices):
                mask[:,hs,ws,:]=i*3+j
        windows=window_partition(mask,self.window).squeeze(-1)
        differences=windows[:,None,:]-windows[:,:,None]
        return differences.masked_fill(differences!=0,-100.).masked_fill(differences==0,0.)

    def forward(self,x,hw):
        h,w=hw
        z=tokens_to_grid(self.norm1(x),hw)
        ph,pw=(-h)%self.window,(-w)%self.window
        z=F.pad(z,(0,0,0,pw,0,ph))
        if self.shift:
            z=torch.roll(z,(-self.shift,-self.shift),(1,2))
        attended=self.attn(window_partition(z,self.window),self.mask(h+ph,w+pw,x.device,z.dtype))
        attended=window_reverse(attended,self.window,h+ph,w+pw,x.shape[0])
        if self.shift:
            attended=torch.roll(attended,(self.shift,self.shift),(1,2))
        x=x+self.drop_path(grid_to_tokens(attended[:,:h,:w]))
        x=self.adapter1(x,hw)
        x=x+self.drop_path(self.mlp(self.norm2(x)))
        v=self.adapter2.encode(x,hw)
        return x+self.adapter2.decode(v)


class Stage(nn.Module):
    def __init__(self, blocks):
        super().__init__()
        self.blocks=nn.ModuleList(blocks)


class MonaSwin(nn.Module):
    def __init__(self, channels=(128,256,512,1024),depths=(2,2,18,2),
                 num_heads=(4,8,16,32),base="mona",base_width=64,
                 patch_size=4,window_size=7,mlp_ratio=4.,dropout=0.,
                 attention_dropout=0.,drop_path_rate=.3,with_cp=False,
                 pretrained=None,require_complete_base=True):
        super().__init__()
        if len(channels)!=4 or len(depths)!=4 or len(num_heads)!=4:
            raise ValueError("Exactly four stages required")
        if base != "mona":
            raise ValueError("This export supports only the Mona branch")
        self.channels,self.depths=tuple(channels),tuple(depths)
        self.base,self.base_width=base,base_width or 64
        self.with_cp,self.pretrained=with_cp,pretrained
        self.require_complete_base=require_complete_base
        self.patch_size=patch_size
        self.patch_embed=PatchEmbedding(3,channels[0],patch_size)
        factory=lambda c:MonaAdapter(c,self.base_width)
        rates=[drop_path_rate*i/max(1,sum(depths)-1) for i in range(sum(depths))]
        self.stages=nn.ModuleList()
        offset=0
        for s,(c,l,heads) in enumerate(zip(channels,depths,num_heads)):
            blocks=[AdaptedBlock(c,heads,window_size,0 if b%2==0 else window_size//2,
                                 rates[offset+b],factory,mlp_ratio,dropout,attention_dropout)
                    for b in range(l)]
            self.stages.append(Stage(blocks))
            offset+=l
        self.mergers=nn.ModuleList([PatchMerging(channels[s],channels[s+1]) for s in range(3)])
        self.output_norms=nn.ModuleList([nn.LayerNorm(c) for c in channels])
        self.apply_trainable_policy()

    def apply_trainable_policy(self):
        for p in self.parameters():
            p.requires_grad_(False)
        for stage in self.stages:
            for block in stage.blocks:
                for adapter in (block.adapter1,block.adapter2):
                    for p in adapter.parameters():
                        p.requires_grad_(True)

    def init_weights(self):
        if self.pretrained:
            from .checkpoint import load_pretrained
            self.pretrained_audit=load_pretrained(self,self.pretrained,self.require_complete_base)
        self.apply_trainable_policy()

    def forward(self,images,valid_hw=None):
        x,hw=self.patch_embed(images)
        features=[]
        if valid_hw is not None:
            sizes=torch.as_tensor(valid_hw,device=images.device)
            if sizes.shape!=(images.shape[0],2) or (sizes<=0).any() or (sizes[:,0]>images.shape[-2]).any() or (sizes[:,1]>images.shape[-1]).any():
                raise ValueError("valid_hw must contain each image's unpadded positive H,W")
        for s,stage in enumerate(self.stages):
            for b,block in enumerate(stage.blocks):
                if self.with_cp and self.training:
                    x=checkpoint(block,x,hw,use_reentrant=False)
                else:
                    x=block(x,hw)
            features.append(tokens_to_grid(self.output_norms[s](x),hw).permute(0,3,1,2).contiguous())
            if s<3:
                x,hw=self.mergers[s](x,hw)
        return tuple(features)
