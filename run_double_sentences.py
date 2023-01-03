import argparse
from io import StringIO
import logging
import math
import os
import random
from pathlib import Path

import datasets
from datasets import load_dataset, load_metric, list_datasets
import torch
torch.autograd.set_detect_anomaly(True)
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import sys
import transformers
from accelerate import Accelerator
from huggingface_hub import Repository

from transformers.optimization import AdamW,get_scheduler
from transformers.data.data_collator import DataCollatorWithPadding,default_data_collator
from transformers.trainer_utils import set_seed,SchedulerType
sys.path.append("./transformers/models")
from transformers.models.bert.tokenization_bert import BertTokenizer
from transformers.models.bert.configuration_bert import BertConfig
from transformers.models.bert.modeling_bert import BertForSequenceClassification
from transformers.models.roberta.tokenization_roberta import RobertaTokenizer
from transformers.models.roberta.configuration_roberta import RobertaConfig
from transformers.models.roberta.modeling_roberta import RobertaForSequenceClassification
from transformers.utils.logging import enable_propagation,set_verbosity_error
from transformers.utils.versions import require_version
from  topic_model.VAE import VAE
from topic_model.utils import evaluate_topic_quality, smooth_curve,calc_topic_diversity
from topic_model.dataset import topicDataset
import numpy as np
import matplotlib.pyplot as plt
from dvq_model.task import get_task, sentence2tenorIndex
import time
from torch.nn.utils.clip_grad import clip_grad_norm_


os.environ["TOKENIZERS_PARALLELISM"] = "false"
logger = logging.getLogger(__name__)
set_verbosity_error()
parser = argparse.ArgumentParser(description="Finetune a transformers model on a text classification task")
parser.add_argument(
    "--task_name",
    type=str,
    default=None,
    help="The name of all task inculing (task_name)",
)
parser.add_argument(
    "--home_dir", type=str, default=None
)
parser.add_argument(
    "--output_dir", type=str, default=None
)
parser.add_argument(
    "--data_dir", type=str, default=None
)
parser.add_argument(
    "--max_length",
    type=int,
    default=128,
    help=(
        "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
        " sequences shorter will be padded if `--pad_to_max_lengh` is passed."
    ),
)
parser.add_argument(
    "--pad_to_max_length",
    action="store_true",
    help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
)
parser.add_argument(
    "--model_name_or_path",
    type=str,
    default=bert-base-uncased,
    help="Path to pretrained model or model identifier from huggingface.co/models.",
    required=True,
)
parser.add_argument(
    "--use_slow_tokenizer",
    action="store_true",
    help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
)
parser.add_argument(
    "--per_device_train_batch_size",
    type=int,
    default=8,
    help="Batch size (per device) for the training dataloader.",
)
parser.add_argument(
    "--per_device_eval_batch_size",
    type=int,
    default=32,
    help="Batch size (per device) for the evaluation dataloader.",
)
parser.add_argument(
    "--learning_rate",
    type=float,
    default=5e-5,
    help="Initial learning rate (after the potential warmup period) to use.",
)
parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
parser.add_argument("--num_train_epochs", type=int, default=10, help="Total number of training epochs to perform.")
parser.add_argument(
    "--max_train_steps",
    type=int,
    default=None,
    help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
)
parser.add_argument("--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler." )
parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
parser.add_argument("--task_desc", type=str, help="task description.")
parser.add_argument("--pretrain_vq", type=int, defalt=0, help="if pretrain_vq.")
parser.add_argument("--pretrain_vq_model", type=str, help="pretrain_vq_model.")
parser.add_argument("--topic_num", type=int, help="")
args = parser.parse_args()
args.output_dir = args.home_dir + "ckpt/"
args.data_dir = args.home_dir + "data/" + args.task_name + "/"
accelerator = Accelerator()
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.ERROR, 
) 


# If passed along, set the training seed now.
if args.seed is not None:
    seed = set_seed(args.seed)
accelerator.wait_for_everyone()

def load_DVQ():
    from dvq_model.config import handle_arguments, params_from_file
    from dvq_model.core_module import TransformerQuantizerEncoder, TransformerEncoderDecoder
    import torch.optim as optim
    
    # 1.config 
    cl_arguments = ['-c',f'{args.home_dir}/dvq_model/base.conf', '-o', f'expname={args.task_name},task={args.task_name},quantizer.K={args.topic_num},root_dir={args.home_dir+"ckpt"},data_dir={args.home_dir+"data"}']
    cl_args = handle_arguments(cl_arguments)
    config = params_from_file(cl_args.config_file, cl_args.overrides)
    # 2.vacab
    task, vocab, indexers = get_task(config)#
    config.vocab_size = vocab.get_vocab_size("tokens")
    tokenIndex = sentence2tenorIndex("hello,there", vocab, indexers)
    # 3. model
    vq_encoder = TransformerQuantizerEncoder(config)
    vae = TransformerEncoderDecoder(config, vq_encoder)
    if not args.pretrain_vq:
        ckpt_path = args.output_dir + "vae_ckpt/" + f"{args.task_name}/"+ args.pretrain_vq_model
        checkpoint = torch.load(ckpt_path)
        vae.load_state_dict(checkpoint["model"], strict=False)
    optimizer = optim.Adam(vae.parameters(), lr=config.pretrain.lr)
    return task, vocab, indexers, vae, optimizer, config

    
def load_topicModel(voc_size):
    ###topic model
    gsm = VAE(voc_size=voc_size, n_topic=args.topic_num)
    gsm_optimizer = torch.optim.Adam(gsm.parameters(),lr=1e-3)
    ckpt = f"{args.home_dir}/ckpt/topic_ckpt/{args.task_name}_{args.topic_num}.ckpt"
    checkpoint=torch.load(ckpt)
    gsm.load_state_dict(checkpoint["net"])
    return gsm,gsm_optimizer





def get_token_word_position_map(epoch, batch_bert_inputIds, bert_tokenizer, topicDataset, batch_dvq_inputIds_sen1, batch_dvq_inputIds_sen2, dvq_vocab):
        word_tokens_map = {}
        batch_bertWord2Token_position = []
        batch_topic_mask = []
        batch_bert2dvq_position = [] 
        batch_bert_words = []
        batch_dvq_words_sen1 = []
        batch_dvq_words_sen2 = []
        batch_dvq2topic_ids_sen1 = [] 
        batch_dvq2topic_ids_sen2 = []
        batch_bert_inputIds = batch_bert_inputIds.cpu().tolist()
        batch_dvq_inputIds_sen1 = batch_dvq_inputIds_sen1.cpu().tolist()
        batch_dvq_inputIds_sen2 = batch_dvq_inputIds_sen2.cpu().tolist()

        for i, bert_inputIds in enumerate(batch_bert_inputIds):
            bertWord2Token_positions = [] 
            tokens = []
            bert_words = []
            for j, token_id in enumerate(bert_inputIds):
                token = bert_tokenizer.convert_ids_to_tokens(token_id)
                tokens.append(token)
                if token.startswith("##"):
                    bertWord2Token_positions[-1].append(j)
                    bert_words[-1] = bert_words[-1] + token[2:]
                else:
                    bertWord2Token_positions.append([j])
                    bert_words.append(token)
            batch_bertWord2Token_position.append(bertWord2Token_positions) 
            batch_bert_words.append(bert_words)  
            
            # for topic_model in dvq model
            topic_mask = []
            for j, word in enumerate(bert_words):
                bertWord2Token_position= bertWord2Token_positions[j]
                if word in topicDataset.dictionary.token2id:
                    for k in bertWord2Token_position:
                        topic_mask.append(1)
                        if epoch == 0:
                            token_id = bert_inputIds[j]
                            token = bert_tokenizer.convert_ids_to_tokens(token_id)
                            if word in word_tokens_map:
                                word_tokens_map[word].add(token)
                            else:
                                word_tokens_map[word] = {token}
                else:
                    for k in bertWord2Token_position:
                        topic_mask.append(0)    
            batch_topic_mask.append(topic_mask)    


            bert2dvq_position = {}
            dvq_ids_sen1 = batch_dvq_inputIds_sen1[i]
            dvq_words_sen1 = []
            dvq_ids_sen2 = batch_dvq_inputIds_sen2[i]
            dvq_words_sen2 = []
            for dvq_id in dvq_ids_sen1:
                dvq_words_sen1.append(dvq_vocab.get_token_from_index(dvq_id, 'tokens'))
            for dvq_id in dvq_ids_sen2:
                dvq_words_sen2.append(dvq_vocab.get_token_from_index(dvq_id, 'tokens'))
            dvq_words_sen = dvq_words_sen1
            for j, bert_word in enumerate(bert_words):
                bertWord2Token_position = bertWord2Token_positions[j] #[1, 2, 3, 4]
                if bert_word == "[SEP]":
                    dvq_words_sen = dvq_words_sen2
                    bert2dvq_position[bertWord2Token_position[0]]=-1  #sep的位置对应的是-1
                    continue
                if bert_word in dvq_words_sen: 
                    dvq_position = dvq_words_sen.index(bert_word) 
                    for k in bertWord2Token_position:
                        bert2dvq_position[k] = dvq_position
                else:
                    for k in bertWord2Token_position:
                        bert2dvq_position[k] = 0
                 
            batch_bert2dvq_position.append(bert2dvq_position)
            batch_dvq_words_sen1.append(dvq_words_sen1)
            batch_dvq_words_sen2.append(dvq_words_sen2)

            dvq2topic_ids_sen1 = []
            for word in dvq_words_sen1:
                if word in topicDataset.dictionary.token2id:
                    topic_id = topicDataset.dictionary.token2id[word]
                    dvq2topic_ids_sen1.append(topic_id)
                else:
                    dvq2topic_ids_sen1.append(0)
            dvq2topic_ids_sen1 = torch.Tensor(dvq2topic_ids_sen1)
            batch_dvq2topic_ids_sen1.append(dvq2topic_ids_sen1)

            dvq2topic_ids_sen2 = []
            for word in dvq_words_sen1:
                if word in topicDataset.dictionary.token2id:
                    topic_id = topicDataset.dictionary.token2id[word]
                    dvq2topic_ids_sen2.append(topic_id)
                else:
                    dvq2topic_ids_sen2.append(0)
            dvq2topic_ids_sen2 = torch.Tensor(dvq2topic_ids_sen2)
            batch_dvq2topic_ids_sen2.append(dvq2topic_ids_sen2)
        batch_dvq2topic_ids_sen1 = torch.stack(batch_dvq2topic_ids_sen1).cuda().long()
        batch_dvq2topic_ids_sen2 = torch.stack(batch_dvq2topic_ids_sen2).cuda().long()
        return batch_bertWord2Token_position, batch_bert2dvq_position, batch_dvq2topic_ids_sen1, batch_dvq2topic_ids_sen2,batch_dvq_words_sen1,batch_dvq_words_sen2,batch_bert_words


def main():
    # Load data
    data_files = {}
    data_files["train"] = args.data_dir + "train.csv"
    data_files["validation"] =  args.data_dir + "validation.csv"
    data_files["test"] =  args.data_dir + "test.csv"
    raw_datasets = load_dataset("csv", data_files=data_files,column_names=["label",'sentence1','sentence2'])

    # Labels
    if args.task_name == "stsb":
        num_labels = 1
    else:
        label_list = raw_datasets["train"].unique("label")
        label_list.sort()  # Let's sort it for determinism
        num_labels =  len(label_list)

    if "roberta" not in args.model_name_or_path:
        config = BertConfig.from_pretrained(args.model_name_or_path)	
        tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path)
        model = BertForSequenceClassification.from_pretrained(args.model_name_or_path, config=config)	
    else:
        config = RobertaConfig.from_pretrained(args.model_name_or_path)	
        tokenizer = RobertaTokenizer.from_pretrained(args.model_name_or_path)
        model = RobertaForSequenceClassification.from_pretrained(args.model_name_or_path, config=config)	
    vae_task, vae_vocab, indexers, vae, vae_optimizer, vae_config = load_DVQ()



    if args.task_name != "stsb":
        label_to_id = {v: i for i, v in enumerate(label_list)}
        model.config.label2id = {l: i for i, l in enumerate(label_list)}
        model.config.id2label = {id: label for label, id in config.label2id.items()}
    else:
        label_to_id = None
    padding = "max_length" if args.pad_to_max_length else False   


    tDataset = topicDataset(raw_datasets["test"], args.task_name, args.data_dir)
    gsm, gsm_optimizer = load_topicModel(voc_size=tDataset.vocabsize)
    def preprocess_function(examples): 
        texts = (
            (examples["sentence1"], examples["sentence2"])
        )
        result = tokenizer(*texts, padding=padding, max_length=args.max_length, truncation=True)
        doc1 = examples["sentence1"]
        doc2 = examples["sentence2"]
        bow1 = tDataset.text2bow(doc1)
        bow2 = tDataset.text2bow(doc2)
        result["bow1"] = bow1
        result["bow2"] = bow2
        tokenIndex1 = sentence2tenorIndex(doc1, vae_vocab, indexers)
        tokenIndex2 = sentence2tenorIndex(doc2, vae_vocab, indexers)
        result["vae1"] = tokenIndex1
        result["vae2"] = tokenIndex2
        if "label" in examples:
            if label_to_id is not None:
                result["labels"] = label_to_id[examples["label"]] 
            else:
                result["labels"] = examples["label"]
        return result
    processed_datasets = raw_datasets.map(
        preprocess_function,
        batched=False,
        remove_columns=raw_datasets["test"].column_names, 
        desc="Running tokenizer on dataset",
    )
    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets["validation"]
    test_dataset = processed_datasets["test"]


    if args.pad_to_max_length:
        data_collator = default_data_collator
    else:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=(8 if accelerator.use_fp16 else None))
    train_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size)
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, shuffle=False, num_workers=4,batch_size=args.per_device_eval_batch_size)
    test_dataloader = DataLoader(test_dataset, collate_fn=data_collator, shuffle=False,num_workers=4, batch_size=args.per_device_eval_batch_size)


    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        }
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)


    model, optimizer, train_dataloader, eval_dataloader, test_dataloader, gsm, gsm_optimizer, vae, vae_optimizer = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, test_dataloader, gsm, gsm_optimizer, vae, vae_optimizer
    )


    num_update_steps_per_epoch = len(train_dataloader)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    if args.task_name =="stsb":
        metric = load_metric("glue", "stsb")
    else:
        metric = load_metric("glue", "mrpc") 


    # Train
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes
    print("***** Running training *****")
    print(f"  model = {args.model_name_or_path}")
    print(f"  Num examples = {len(train_dataset)}")
    print(f"  Num Epochs = {args.num_train_epochs}")
    print(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    print(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    print(f"  Total optimization steps = {args.max_train_steps}")
    print(f"  pretrain_vq  = {args.pretrain_vq}")
    print(f"  max_length  = {args.max_length}")
    print(f"  learning_rate  = {args.learning_rate}")
    print(f"  pretrain_vq_model  = {args.pretrain_vq_model}")
    print(f"  topic_num  = {args.topic_num}")
    print(f"  task  = {args.task_name}")
    # print(f"  seed  = {seed}")

    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    
    completed_steps = 0 
    word_topic = np.zeros((config.vocab_size,50), dtype=int)
    loss_theta_list = []
    start_time = time.time()
    def _one_step_topic_model(topic_batch_sen):
        p_x,mus,log_vars,theta,beta,topic_embedding = gsm(topic_batch_sen)
        logsoftmax = torch.log_softmax(p_x + 1e-10, dim=1) 
        rec_loss = -1.0 * torch.sum(topic_batch_sen*logsoftmax)            
        kl_div = -0.5 * torch.sum(1+log_vars-mus.pow(2)-log_vars.exp())
        loss_topic = rec_loss + kl_div
        return loss_topic, theta, beta, topic_embedding
    def _one_step_vq_model(vae_batch, topic_embedding, theta, beta, batch_dvq2topic_ids):
            batch_vae = {}
            batch_vae["input"] = vae_batch
            batch_vae["topic_embedding"] = topic_embedding
            batch_vae["theta"] = theta
            batch_vae["batch_dvq2topic_ids"] = batch_dvq2topic_ids
            batch_vae["beta"] = beta
            batch_size = batch_vae["input"].size(0) 
            output_dict = vae(batch_vae, args.pretrain_vq)
            loss_vq = output_dict["loss"]    
            z_q =  output_dict["z_q"]
            src_nopad_mask = output_dict["src_nopad_mask"]
            loss_vq = loss_vq.div(batch_size)
            nll = output_dict["loss_reconstruct"].item()
            ntoken =  output_dict["ntokens"] 
            return loss_vq, z_q, src_nopad_mask, nll, ntoken
    def _one_step_bert(batch, batch_bertWord2Token_position, batch_bert2dvq_position, z_q_sen1, z_q_sen2, batch_dvq_words_sen1, batch_dvq_words_sen2, batch_bert_words, src_nopad_mask_1, src_nopad_mask_2):
            batch["batch_bertWord2Token_position"] = batch_bertWord2Token_position
            batch["batch_bert2dvq_position"] = batch_bert2dvq_position
            batch["z_q_1"] = z_q_sen1
            batch["z_q_2"] = z_q_sen2
            batch["batch_dvq_words_sen1"] = batch_dvq_words_sen1
            batch["batch_dvq_words_sen2"] = batch_dvq_words_sen2
            batch["batch_bert_words"] = batch_bert_words
            batch["src_nopad_mask_1"] = src_nopad_mask_sen1
            batch["src_nopad_mask_2"] = src_nopad_mask_sen2
            batch.pop("vae1")
            batch.pop("vae2")
            batch.pop("bow2")
            batch.pop("bow1")
            return batch
    best_result = 0
    best_model = 0   
    for epoch in range(args.num_train_epochs):
        model.train()
        gsm.train()
        vae.train()
        for step, batch in enumerate(train_dataloader):
            batch_bertWord2Token_position, batch_bert2dvq_position , batch_dvq2topic_ids_sen1, batch_dvq2topic_ids_sen2, batch_dvq_words_sen1, batch_dvq_words_sen2, batch_bert_words= \
                get_token_word_position_map(epoch, batch["input_ids"], tokenizer, tDataset, batch["vae1"], batch["vae2"], vae_vocab)
            #topic_model    
            loss_topic_sen1, theta_sen1, beta_sen1, topic_embedding_sen1 = _one_step_topic_model(batch["bow1"])
            loss_topic_sen2, theta_sen2, beta_sen2, topic_embedding_sen2 = _one_step_topic_model(batch["bow2"])
            # vq model
            loss_vq_sen1, z_q_sen1, src_nopad_mask_sen1, _, _ = _one_step_vq_model(batch["vae1"], topic_embedding_sen1, theta_sen1, beta_sen1, batch_dvq2topic_ids_sen1)
            loss_vq_sen2, z_q_sen2, src_nopad_mask_sen2, _, _ = _one_step_vq_model(batch["vae2"], topic_embedding_sen2, theta_sen2, beta_sen2, batch_dvq2topic_ids_sen2)
            # pretrain VQ
            if args.pretrain_vq:
                accelerator.backward(loss_vq_sen1,retain_graph=True)
                accelerator.backward(loss_vq_sen2,retain_graph=True)
                if vae_config.pretrain.grad_norm: #grad_norm=5
                    clip_grad_norm_(vae.parameters(), vae_config.pretrain.grad_norm)
                vae_optimizer.step()
                vae_optimizer.zero_grad()
                #train topic
                accelerator.backward(loss_topic_sen1)
                accelerator.backward(loss_topic_sen2)
                gsm_optimizer.step()
                gsm_optimizer.zero_grad()  
            else:
                batch = _one_step_bert(batch, batch_bertWord2Token_position, batch_bert2dvq_position, z_q_sen1, z_q_sen2, batch_dvq_words_sen1, batch_dvq_words_sen2, batch_bert_words, src_nopad_mask_sen1, src_nopad_mask_sen2)
                outputs = model(**batch)
                loss = outputs.loss
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1            
                if completed_steps >= args.max_train_steps:
                    break

        ## eval&test
        for dataloader in [eval_dataloader, test_dataloader]:
            model.eval()
            vae.eval()
            gsm.eval()
            nll_discrete = 0.0
            ntoken_discrete = 0.0
            for step, batch in enumerate(dataloader):           
                batch_bertWord2Token_position, batch_bert2dvq_position , batch_dvq2topic_ids_sen1, batch_dvq2topic_ids_sen2,batch_dvq_words_sen1,batch_dvq_words_sen2,batch_bert_words= \
                    get_token_word_position_map(epoch, batch["input_ids"], tokenizer,tDataset, batch["vae1"], batch["vae2"], vae_vocab)
                #topic model
                #sen1
                loss_topic_sen1, theta_sen1, beta_sen1, topic_embedding_sen1 = _one_step_topic_model(batch["bow1"])
                loss_topic_sen2, theta_sen2, beta_sen2, topic_embedding_sen2 = _one_step_topic_model(batch["bow2"])
                # vq model
                loss_vq_sen1, z_q_sen1, src_nopad_mask_sen1, nll_sen1, ntoken_sen1 = _one_step_vq_model(batch["vae1"], topic_embedding_sen1, theta_sen1, beta_sen1, batch_dvq2topic_ids_sen1)
                loss_vq_sen2, z_q_sen2, src_nopad_mask_sen2, nll_sen2, ntoken_sen2 = _one_step_vq_model(batch["vae2"], topic_embedding_sen2, theta_sen2, beta_sen2, batch_dvq2topic_ids_sen2)
                nll_discrete += nll_sen1
                nll_discrete += nll_sen2
                ntoken_discrete += ntoken_sen1
                ntoken_discrete += ntoken_sen2

                #bert_task
                if not args.pretrain_vq:
                    batch = _one_step_bert(batch, batch_bertWord2Token_position, batch_bert2dvq_position, z_q_sen1, z_q_sen2, batch_dvq_words_sen1, batch_dvq_words_sen2, batch_bert_words, src_nopad_mask_sen1, src_nopad_mask_sen2)
                    outputs = model(**batch)
                    predictions = outputs.logits.argmax(dim=-1) if args.task_name!="stsb" else outputs.logits.squeeze()
                    metric.add_batch(
                        predictions=accelerator.gather(predictions),
                        references=accelerator.gather(batch["labels"]),
                    )  
            #bert
            if not args.pretrain_vq:
                eval_metric = metric.compute()
                if dataloader == test_dataloader:
                    print(f"test epoch {epoch}: {eval_metric}")
                else:
                    if float(eval_metric["f1"]) > best_result: 
                        print("++++++This is best result++++++") 
                        best_result = float(eval_metric["f1"])
                        best_model = model
                    print(f"eval epoch {epoch}: {eval_metric}")
                    


            # save_model
            if args.pretrain_vq:
                perplexity_discrete = math.exp(1.0*nll_discrete/ntoken_discrete)
                print("dvq ppx_document:", perplexity_discrete)
                checkpoint_state = {"model": vae.state_dict()}
                checkpoint_path = args.output_dir + "vae_ckpt/" + f"{args.task_name}/" + f"{args.task_name}_{args.topic_num}" + f"_epoch_{epoch}"
                torch.save(checkpoint_state, checkpoint_path)



    if args.output_dir is not None and args.pretrain_vq==0:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(best_model)
        unwrapped_model.save_pretrained(args.output_dir + f"mrpc_bert_{best_result}/", save_function=accelerator.save)
        if accelerator.is_main_process:
            tokenizer.save_pretrained(args.output_dir+ f"mrpc_bert_{best_result}/")

if __name__ == "__main__":
    main()
