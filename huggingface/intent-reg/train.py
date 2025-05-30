import os
import argparse
import json

import numpy as np

import torch
from torch.utils.data import DataLoader

import transformers
from transformers import BertTokenizer, BertConfig
from transformers import get_linear_schedule_with_warmup

from datasets import IntentSlotDataset
from models import JointBert
from tools import save_module

from detector import JointIntentSlotDetector
import warnings
warnings.filterwarnings("ignore")

def evaluate_model(args, model, tokenizer, test_data, intent_dict, slot_dict):

    # def calculate_slot_precision(p_slots, o_slots):
    #     results = []
    #     for slot_name, slot_value_list in p_slots.items():
    #         for slot_value in slot_value_list:
    #             results.append(float(slot_name in o_slots and slot_value in o_slots[slot_name]))
                
    #     if len(results) == 0:
    #         return 1.
    #     return np.mean(results)

    def calculate_slot_precision(predicted_slots, label_slots):
        if len(predicted_slots) == 0:
            return 0.0  # 没有预测出的槽位时，精确率为0
        
        correct_predictions = 0
        for slot in predicted_slots:
            if slot in label_slots:
                correct_predictions += 1
        
        precision = correct_predictions / len(predicted_slots)
        return precision

    def calculate_slot_recall(predicted_slots, label_slots):
        if len(label_slots) == 0:
            return 0.0  # 没有实际槽位时，召回率为0
        
        correct_predictions = 0
        for slot in label_slots:
            if slot in predicted_slots:
                correct_predictions += 1
        
        recall = correct_predictions / len(label_slots)
        return recall


    def filter_slots(text, slots):
        outputs = {}
        for slot_name, slot_value_list in slots.items():
            output_value_list = []
            for slot_value in slot_value_list:
                if slot_value in text:
                    output_value_list.append(slot_value)
            if len(output_value_list) > 0:
                outputs[slot_name] = output_value_list
        return outputs
                           
    detector = JointIntentSlotDetector(
        model=model,
        tokenizer=tokenizer,
        intent_dict=intent_dict,
        slot_dict=slot_dict
    )

    intent_acc_results = []
    slot_precision_results = []
    slot_recall_results = []
    for item in test_data:
        outputs = detector.detect(item['text'])
        label_slots = filter_slots(item['text'], item['slots'])
        
        intent_acc_results.append(float(outputs['intent'] == item['intent']))
        slot_precision_results.append(calculate_slot_precision(outputs['slots'], label_slots))
        slot_recall_results.append(calculate_slot_recall(outputs['slots'], label_slots))
        # print(slot_precision_results)
        # print(slot_recall_results)
    return np.mean(intent_acc_results), np.mean(slot_precision_results), np.mean(slot_recall_results)
                                      
def train(args):
    #-----------set cuda environment-------------
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices
    device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"

    #-----------load tokenizer-----------
    tokenizer = BertTokenizer.from_pretrained(args.tokenizer_path)
    save_module(tokenizer, args.save_dir, module_name="tokenizer", additional_name="")
    
    #-----------load data-----------------
    dataset = IntentSlotDataset.load_from_path(
        data_path=args.train_data_path,
        intent_label_path=args.intent_label_path,
        slot_label_path=args.slot_label_path,
        tokenizer=tokenizer
    )

    with open(args.test_data_path, 'r', encoding="utf-8") as f:
        test_data = json.load(f)

    #-----------load model-----------
    model = JointBert.from_pretrained(
        args.model_path,
        slot_label_num = dataset.slot_label_num,
        intent_label_num = dataset.intent_label_num
    )
    #print(model)
    model = model.to(device).train()
    save_module(model, args.save_dir, module_name='model', additional_name="epoch0")
     
    dataloader = DataLoader(
        dataset,
        shuffle=True,
        batch_size=args.batch_size,
        collate_fn=dataset.batch_collate_fn)

    #-----------calculate training steps-----------
    if args.max_training_steps > 0:
        total_steps = args.max_training_steps
    else:
        total_steps = len(dataset) * args.train_epochs // args.gradient_accumulation_steps // args.batch_size
        

    print('calculated total optimizer update steps : {}'.format(total_steps))

    #-----------prepare optimizer and schedule------------
    parameter_names_no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [
            para for para_name, para in model.named_parameters()
            if not any(nd_name in para_name for nd_name in parameter_names_no_decay)
            ],
         'weight_decay': args.weight_decay},
        {'params': [
            para for para_name, para in model.named_parameters()
            if any(nd_name in para_name for nd_name in parameter_names_no_decay)
            ],
         'weight_decay': 0.0}
    ]
    optimizer = transformers.AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=total_steps)

    #-----------training-------------
    update_steps = 0
    total_loss = 0.
    
    for epoch in range(args.train_epochs):
        step = 0
        for batch in dataloader:
            step += 1
            input_ids, intent_labels, slot_labels = batch
        
            outputs = model(
                input_ids=torch.tensor(input_ids).long().to(device),
                intent_labels=torch.tensor(intent_labels).long().to(device),
                slot_labels=torch.tensor(slot_labels).long().to(device)
            )

            loss = outputs['loss']
            total_loss += loss.item()
            
            if args.gradient_accumulation_steps > 1:
                loss = loss/args.gradient_accumulation_steps

            loss.backward()

            if step % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()
                model.zero_grad()
                update_steps += 1

                if args.logging_steps > 0 and update_steps % args.logging_steps == 0:
                    print("total step {} epoch {} : loss {}".format(update_steps, epoch, total_loss / args.logging_steps))
                    total_loss = 0.

                # 以steps保存
                # if args.saving_steps > 0 and update_steps % args.saving_steps == 0:
                #     save_module(model, args.save_dir, module_name='model', additional_name="model_step{}".format(update_steps))
                    

        intent_acc, slot_prec, slot_recall = evaluate_model(
            args=args,
            model=model,
            tokenizer=tokenizer,
            test_data=test_data,
            intent_dict=dataset.intent_label_dict,
            slot_dict=dataset.slot_label_dict
        )

        print('\033[32m intent accuracy: {}; slot precision: {}; slot recall: {} \033[0m'.format(intent_acc, slot_prec, slot_recall))

        if args.saving_epochs > 0 and (epoch+1) % args.saving_epochs == 0:
            save_module(model, args.save_dir, module_name='model', additional_name="model_epoch{}".format(epoch))

        if update_steps > total_steps:
            break

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # environment parameters
    parser.add_argument("--cuda_devices", type=str, default='0', help='set cuda device numbers')
    parser.add_argument("--no_cuda", action='store_true', default=False, help='whether use cuda device for training')
    
    # model parameters
    parser.add_argument("--tokenizer_path", type=str, default='bert-base-chinese',  help="pretrained tokenizer loading path")
    parser.add_argument("--model_path", type=str, default='bert-base-chinese',  help="pretrained model loading path")

    # data parameters
    parser.add_argument("--train_data_path", type=str, default='path/to/data.json',  help="training data path")
    parser.add_argument("--test_data_path", type=str, default='path/to/data.json',  help="testing data path")
    parser.add_argument("--slot_label_path", type=str, default='data/slot_labels.txt',  help="slot label path")
    parser.add_argument("--intent_label_path", type=str, default='data/intent_labels.txt',  help="intent label path")

    # training parameters
    parser.add_argument("--save_dir", type=str, default='path/to/save/model',  help="directory to save the model")
    parser.add_argument("--max_training_steps", type=int, default=0, help = 'max training step for optimizer, if larger than 0')
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="number of updates steps to accumulate before performing a backward() pass.")
    parser.add_argument("--saving_steps", type=int, default=300, help="parameter update step number to save model")
    parser.add_argument("--logging_steps", type=int, default=10, help="parameter update step number to print logging info.")
    parser.add_argument("--eval_steps", type=int, default=10, help="parameter update step number to print logging info.")
    parser.add_argument("--saving_epochs", type=int, default=1, help="parameter update epoch number to save model")
    
    parser.add_argument("--batch_size", type=int, default=128, help = 'training data batch size')
    parser.add_argument("--train_epochs", type=int, default=10, help = 'training epoch number')

    parser.add_argument("--learning_rate", type=float, default=5e-5, help = 'learning rate')
    parser.add_argument("--adam_epsilon", type=float, default=1e-8, help="epsilon for Adam optimizer")
    parser.add_argument("--warmup_steps", type=int, default=0, help="warmup step number")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="weight decay rate")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="maximum norm for gradients")
        
    args = parser.parse_args()

    train(args)





                                      
