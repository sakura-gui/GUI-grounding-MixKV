#!/bin/bash
set -e
ratios=(0.1)
methods=("adakv" )
select_methods=("attn" "headwisemixkv" )
budgets=(   64 128 256 512 1024 ) 
mask_ratio=0.1 # only used for "mask" / "mask_random" 
model=("ui_venus_ground_7b")
for budget in ${budgets[@]}; do
    for ratio in ${ratios[@]}; do
        for method in ${methods[@]}; do
            for select_method in ${select_methods[@]}; do
                export METHOD=${method}
                export BUDGET=${budget}
                export RATIO=${ratio}
                export MASK_RATIO=${mask_ratio}
                export SELECT_METHOD=${select_method}
                python models/grounding/eval_screenspot_pro.py  \
                    --model_type ${model}  \
                    --screenspot_imgs "/data/u202315217/data/screenspot-v2/images"  \
                    --screenspot_test "/data/u202315217/data/screenspot-v2/annotations"  \
                    --model_name_or_path "/data/u202315217/qwen2.5vl-7b/" \
                    --task "all" \
                    --language "en" \
                    --gt_type "positive" \
                    --log_path "qwen2_5_vl_7b/ss2_${method}_${budget}_${select_method}.json" \
                    --inst_style "instruction"
            done
        done
    done
done




#models=("ui_venus_ground_7b") 
#for model in "${models[@]}"
#do
#    python models/grounding/eval_screenspot_pro.py  \
#        --model_type ${model}  \
#        --screenspot_imgs "/data/guixiyan-20250909/screenspot-pro/images/"  \
#       --screenspot_test "/data/guixiyan-20250909/screenspot-pro/annotations/"  \
#       --model_name_or_path "/data/guixiyan-20250909/qwen2.5vl-7b/" \
#        --task "all" \
#        --language "en" \
#        --gt_type "positive" \
#        --log_path "qwen2_5_vl_7b/qwen_2_5_vl_7b_pro.json" \
#        --inst_style "instruction"

#done



