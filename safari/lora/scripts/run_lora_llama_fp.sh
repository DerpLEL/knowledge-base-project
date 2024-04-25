model_name_or_path=/kfdata03/kf_grp/hrwang/LLM/models/chinese/belle-llama-7b-2m

CUDA_VISIBLE_DEVICES=2 python lora/train_belle.py \
    --model_type belle-llama \
    --model_name_or_path ${model_name_or_path} \
    --llama \
    --use_lora True \
    --lora_config configs/lora_config_llama.json \
    --gradient_checkpointing True \
    --output_dir ./belle-llama-fp16 \