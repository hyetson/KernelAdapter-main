# Extracted for the Mona-only KernelAdapter export, 2026-10-14.
"""Microsoft-Swin initialization and backbone PEFT-delta semantics."""
import re
import torch
def unwrap(payload):
    if isinstance(payload,dict):
        for key in ("state_dict","model","model_state"):
            if isinstance(payload.get(key),dict):
                return payload[key]
        if all(isinstance(v,torch.Tensor) for v in payload.values()):
            return payload
    raise ValueError("Expected a tensor state dictionary")
def map_key(key):
    while any(key.startswith(p) for p in ("module.","model.","backbone.")):
        key=key.split(".",1)[1]
    key=key.replace("patch_embed.proj.","patch_embed.projection.")
    key=re.sub(r"^layers\.(\d+)\.blocks\.",r"stages.\1.blocks.",key)
    key=re.sub(r"^layers\.(\d+)\.downsample\.",r"mergers.\1.",key)
    key=re.sub(r"^norm(\d+)\.",r"output_norms.\1.",key)
    key=re.sub(r"^norm\.",r"output_norms.3.",key)
    return key.replace(".attn.proj.",".attn.projection.")
def load_pretrained(model,path,require_complete=True):
    source=unwrap(torch.load(path,map_location="cpu",weights_only=False))
    target=model.state_dict()
    from .discarded import DiscardedSubspaceAdapter
    adapter_ids={id(p) for m in model.modules() if isinstance(m,DiscardedSubspaceAdapter) for p in m.parameters()}
    trainable={k for k,p in model.named_parameters() if p.requires_grad or id(p) in adapter_ids}
    mapped,ignored,unexpected={},[],[]
    for original,value in source.items():
        key=map_key(original)
        if key in target and key not in trainable:
            if value.shape!=target[key].shape:
                raise ValueError(f"Pretrained shape mismatch: {original}")
            if key in mapped:
                raise ValueError(f"Multiple source tensors map to {key}")
            mapped[key]=value
        elif key.startswith("head.") or key.endswith(("relative_position_index","attn_mask")):
            ignored.append(original)
        else:
            unexpected.append(original)
    allowed={k for k in target if k.startswith(("output_norms.0.","output_norms.1.","output_norms.2."))}
    missing=set(target)-set(mapped)
    missing_base=missing-trainable-allowed
    if unexpected or (require_complete and missing_base):
        raise ValueError(f"Incompatible pure Swin checkpoint. missing_base={sorted(missing_base)}, unexpected={unexpected}")
    model.load_state_dict(mapped,strict=False)
    return {"loaded":sorted(mapped),"missing":sorted(missing),"ignored":ignored}
def _delta_names(model):
    from .discarded import DiscardedSubspaceAdapter
    ids={id(p) for m in model.modules() if isinstance(m,DiscardedSubspaceAdapter) for p in m.parameters()}
    return {k for k,p in model.named_parameters() if p.requires_grad or id(p) in ids}
def save_delta(model,path):
    names=_delta_names(model)
    state=model.state_dict()
    torch.save({"format":"kerneladapter_backbone_delta_v1","signature":model.kernel_signature,"state_dict":{
        k:state[k].detach().cpu() for k in names}},path)
def load_delta(model,path):
    payload=torch.load(path,map_location="cpu",weights_only=True)
    if payload.get("format")!="kerneladapter_backbone_delta_v1" or payload.get("signature")!=model.kernel_signature:
        raise ValueError("Backbone delta format/architecture differs")
    expected=_delta_names(model)
    source=payload["state_dict"]
    if set(source)!=expected:
        raise ValueError("Delta architecture/parameter keys differ")
    model.load_state_dict(source,strict=False)
