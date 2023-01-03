##Overview

The code for "Learning Semantic Textual Similarity via Topic-informed Discrete Latent Variables".

## Requirements

``` 
python=3.7
pytorch-cuda=11.6
```

## Usage

1. Train topic model 

   ``` 
   python3 topic_model/GSM_run.py --taskname mrpc --n_topic 30 --num_epochs 500
   ```

2. Joint train topic model and VQ-VAE model

   ```
   	 python run_double_sentences.py \
           --model_name_or_path $bert-base-uncased \
           --pretrain_vq 1\
           --topic_num 30\
           --task_name mrpc \
           --home_dir /home/XXX/DisBert/
   ```

3. Train and test DisBert

   ```
   	 python run_double_sentences.py \
           --model_name_or_path $bert-base-uncased \
           --max_length 128 \
           --per_device_train_batch_size 32\
           --learning_rate 5e-5 \
           --num_train_epochs 10\
           --pretrain_vq 0\
           --topic_num 30\
           --pretrain_vq_model dis-cls_${topic}_epoch_9\
           --task_name mrpc \
           --home_dir /home/XXX/DisBert/
   ```

## Citation

```
@inproceedings{yu2022DisBert,
  title={Learning Semantic Textual Similarity via Topic-informed Discrete Latent Variables},
  author={Erxin Yu, Lan Du, Yuan Jin, Zhepei Wei, Yi Chang},
  booktitle={Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing},
  pages={4937–4948},
  year={2022}
}
```



