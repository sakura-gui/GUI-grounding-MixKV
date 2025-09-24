from importlib.metadata import version
import transformers


from sparsemm.qwen_model import qwen_flash_attn2_forward_AdaKV, qwen_flash_attn2_forward_MixSparseMM, qwen_flash_attn2_forward_PyramidKV, qwen_flash_attn2_forward_SnapKV, \
                                qwen_flash_attn2_forward_SparseMM, qwen_flash_attn2_forward_Mask
from sparsemm.qwen_model import prepare_inputs_for_generation_qwen, adakv_qwen_forward

def replace_qwen(method):
    if method == 'snapkv':
        print("Using SnapKV!")
        transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLFlashAttention2.forward = qwen_flash_attn2_forward_SnapKV

    elif method == 'pyramidkv':
        print("Using PyramidKV!")
        transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLFlashAttention2.forward = qwen_flash_attn2_forward_PyramidKV
    
    if method == "adakv":
        print("Using AdaKV!")
        transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLModel.forward = adakv_qwen_forward
        transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLFlashAttention2.forward = qwen_flash_attn2_forward_AdaKV

    elif method == "sparsemm":
        print("Using SparseMM!")
        transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLModel.forward = adakv_qwen_forward
        transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLFlashAttention2.forward = qwen_flash_attn2_forward_SparseMM

    elif method == 'mask':
        print("Mask Head")
        transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLFlashAttention2.forward = qwen_flash_attn2_forward_Mask
    
    elif method == "mixsparsemm":
        print("Using MixSparseMM!")
        transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLModel.forward = adakv_qwen_forward
        transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLFlashAttention2.forward = qwen_flash_attn2_forward_MixSparseMM
    #if method not in ["fullkv"]:
    #    transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLForConditionalGeneration.prepare_inputs_for_generation = prepare_inputs_for_generation_qwen

