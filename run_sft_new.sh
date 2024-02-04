# lr=1e-4
# lora_rank=64
# lora_alpha=128
# lora_trainable="q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj"
# modules_to_save="embed_tokens,lm_head"
# lora_dropout=0.05
# pretrained_model=/home/oem/Desktop/Hengyu/biglanguage/Chinese-LLaMA-Alpaca-2/model/chinese-llama-2-7b-hf
# chinese_tokenizer_path=/home/oem/Desktop/Hengyu/biglanguage/Chinese-LLaMA-Alpaca-2/model/chinese-llama-2-7b-hf
# dataset_dir=/home/oem/Desktop/Hengyu/biglanguage/Chinese-LLaMA-Alpaca-2/dataset
# per_device_train_batch_size=1
# per_device_eval_batch_size=1
# gradient_accumulation_steps=8
# max_seq_length=512
# output_dir=/home/oem/Desktop/Hengyu/biglanguage/outputdir
# validation_file=/home/oem/Desktop/Hengyu/biglanguage/Chinese-LLaMA-Alpaca-2/validation_data.json
# deepspeed_config_file=ds_zero2_no_offload.json

export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"

torchrun --nnodes 1 --nproc_per_node 1 run_clm_sft_with_peft.py \
    --deepspeed ds_zero2_no_offload.json \
    --model_name_or_path /home/oem/Desktop/Hengyu/biglanguage/Chinese-LLaMA-Alpaca-2/model/chinese-llama-2-7b-hf \
    --tokenizer_name_or_path /home/oem/Desktop/Hengyu/biglanguage/Chinese-LLaMA-Alpaca-2/model/chinese-llama-2-7b-hf \
    --dataset_dir /home/oem/Desktop/Hengyu/biglanguage/Chinese-LLaMA-Alpaca-2/dataset4\
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --do_train \
    --do_eval \
    --seed $RANDOM \
    --fp16 \
    --num_train_epochs 3 \
    --lr_scheduler_type cosine \
    --learning_rate 1e-4 \
    --warmup_ratio 0.04 \
    --weight_decay 0 \
    --logging_strategy steps \
    --logging_steps 10 \
    --save_strategy steps \
    --save_total_limit 3 \
    --evaluation_strategy steps \
    --eval_steps 100 \
    --save_steps 200 \
    --gradient_accumulation_steps 8 \
    --preprocessing_num_workers 8 \
    --max_seq_length 512 \
    --output_dir /home/oem/Desktop/Hengyu/biglanguage/outputdir4 \
    --overwrite_output_dir \
    --ddp_timeout 30000 \
    --logging_first_step True \
    --lora_rank 48 \
    --lora_alpha 64 \
    --trainable "q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj" \
    --lora_dropout 0.05\
    --torch_dtype float16 \
    --validation_file /home/oem/Desktop/Hengyu/biglanguage/Chinese-LLaMA-Alpaca-2/validation_data.json \
    --load_in_kbits 8 \
    --save_safetensors False \
    --gradient_checkpointing \
    --ddp_find_unused_parameters False 