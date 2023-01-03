# export CUDA_VISIBLE_DEVICES=0

# python3 topic_model/GSM_run.py --taskname mrpc --n_topic 80 --num_epochs 500


# 0--->2
# 1---->3
# 2---->0
# 3--->1
# #roberta-base   bert-base-uncased # ##dis cls dis-cls avg theta  zq

# for i in 1 2 3 4 5 6 7 8 
# do
#     task=mrpc
#     topic=30
#     model=bert-base-uncased
#     mode=dis
#     python run_double_sentences.py \
#         --model_name_or_path $model \
#         --max_length 128 \
#         --per_device_train_batch_size 32\
#         --learning_rate 2e-5 \
#         --num_train_epochs 10\
#         --output2file 0\
#         --pretrain_vq 0\
#         --topic_num $topic\
#         --pretrain_vq_model dis-cls_${topic}_epoch_9\
#         --task_name $task \
#         --mode $mode\
#         --task_desc dis-cls\
#         --train_file /home/yex/DisBert/data/sts_data/$task/train.json \
#         --validation_file  /home/yex/DisBert/data/sts_data/$task/test.json \
#         --test_file  /home/yex/DisBert/data/sts_data/$task/test.json 
# done


for i in 1 2 3 4 5 6 7 8 9 10
do
    export CUDA_VISIBLE_DEVICES=7
    task=mrpc
    topic=30
    model=roberta-base
    python run_double_sentences.py \
        --model_name_or_path $model \
        --max_length 128 \
        --per_device_train_batch_size 32\
        --learning_rate 2e-5 \
        --num_train_epochs 10\
        --pretrain_vq 0\
        --topic_num $topic\
        --pretrain_vq_model dis-cls_${topic}_epoch_9\
        --task_name $task \
        --home_dir /home/yex/DisBert/
done


# export CUDA_VISIBLE_DEVICES=5
# task=mrpc
# topic=30
# model=bert-base-uncased
# mode=dis
# python run_double_sentences.py \
#     --model_name_or_path $model \
#     --max_length 128 \
#     --per_device_train_batch_size 32\
#     --learning_rate 5e-5 \
#     --num_train_epochs 10\
#     --pretrain_vq 0\
#     --topic_num $topic\
#     --pretrain_vq_model dis-cls_${topic}_epoch_9\
#     --task_name $task \
#     --mode $mode\
#     --task_desc dis-cls-new\
#     --home_dir /home/yex/DisBert/