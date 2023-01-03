#Train topic model, take mrpc task as an example:
# python3 topic_model/GSM_run.py --taskname mrpc --n_topic 30 --num_epochs 500


# Joint train topic model and VQ-VAE model:
python run_double_sentences.py \
   --pretrain_vq 1\
   --topic_num 30\
   --task_name mrpc \
   --home_dir /home/pc/Desktop/DisBert

#Train and test DisBert:
# python run_double_sentences.py \
#    --model_name_or_path bert-base-uncased \
#    --max_length 128 \
#    --per_device_train_batch_size 32\
#    --topic_num 30\
#    --pretrain_vq_model dis-cls_${topic}_epoch_9\
#    --task_name mrpc \
#    --home_dir /home/pc/Desktop/DisBert