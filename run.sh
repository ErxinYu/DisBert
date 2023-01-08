#Train topic model, take mrpc task as an example:
# python3 topic_model/GSM_run.py --taskname mrpc --n_topic 30 --num_epochs 500


# Joint train topic model and VQ-VAE model:
# export CUDA_VISIBLE_DEVICES=0
# python run_double_sentences.py \
#    --pretrain_vq 1\
#    --topic_num 30\
#    --task_name mrpc \
#    --home_dir /home/pc/Desktop/DisBert/

#Train and test DisBert:
# for i in 1 2 3 4 5 6 7 8 9 10
# do
   export CUDA_VISIBLE_DEVICES=7
   python run_double_sentences.py \
      --load_bert_model True \
      --model_name_or_path /home/yex/temp2/DisBert/ckpt/mrpc_bert_0.8895348837209301 \
      --max_length 128 \
      --per_device_train_batch_size 32\
      --topic_num 30\
      --pretrain_vq_model mrpc_30\
      --task_name mrpc \
      --home_dir /home/yex/temp2/DisBert/

# done