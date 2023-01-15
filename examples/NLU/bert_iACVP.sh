export num_gpus=1
export CUBLAS_WORKSPACE_CONFIG=":16:8" # https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
export PYTHONHASHSEED=0
export output_dir="./iACVP/r8_test"

python \
examples/text-classification/run_iACVP.py \
--model_name_or_path Rostlab/prot_bert \
--do_train \
--do_eval \
--per_device_train_batch_size 16 \
--num_train_epochs 40 \
--output_dir $output_dir/model \
--overwrite_output_dir \
--logging_steps 10 \
--logging_dir $output_dir/log \
--logging_strategy steps \
--evaluation_strategy epoch \
--apply_lora \
--lora_r 8 \
--lora_alpha 16 \
--seed 0 \
--lr_scheduler_type constant \
--save_total_limit 2 \
--load_best_model_at_end True \
--report_to tensorboard
# --save_strategy no \
# --weight_decay 0.1 \
# --learning_rate 4e-4 \
# --warmup_ratio 0.06 \
# --gradient_accumulation_steps 64 \
# --per_device_eval_batch_size 1 \

exit 1
