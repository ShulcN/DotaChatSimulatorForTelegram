import numpy as np
import torch
from transformers import BertTokenizerFast, BertForQuestionAnswering
from transformers import DistilBertTokenizerFast, DistilBertForQuestionAnswering
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
from SQuAD_Dataset import SQuAD_Dataset
from bert_dataset import CustomDataset
from tqdm import tqdm

class BertQA:

    def __init__(self, model, tokenizer_path='DeepPavlov/rubert-base-cased-sentence', n_classes=2, epochs=3,
                 model_save_path='bert_3.pt'):
        #self.model = BertForQuestionAnswering.from_pretrained(model_path)
        self.model = model
        #self.tokenizer = BertTokenizerFast.from_pretrained("blanchefort/rubert-base-cased-sentiment")
        self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
        #self.tokenizer =  DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
        #self.tokenizer = DistilBertTokenizerFast.from_pretrained(tokenizer_path)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.epochs = epochs

    def add_end_idx(self, answers, contexts):
        for answer, context in zip(answers, contexts):
            gold_text = answer['text']
            start_idx = answer['answer_start']
            end_idx = start_idx + len(gold_text)
            # sometimes squad answers are off by a character or two so we fix this
            if context[start_idx:end_idx] == gold_text:
                answer['answer_end'] = end_idx
            elif context[start_idx - 1:end_idx - 1] == gold_text:
                answer['answer_start'] = start_idx - 1
                answer['answer_end'] = end_idx - 1  # When the gold label is off by one character
            elif context[start_idx - 2:end_idx - 2] == gold_text:
                answer['answer_start'] = start_idx - 2
                answer['answer_end'] = end_idx - 2  # When the gold label is off by two characters

    def add_token_positions(self, encodings, answers):
        start_positions = []
        end_positions = []
        for i in range(len(answers)):
            start_positions.append(encodings.char_to_token(i, answers[i]["answer_start"]))
            end_positions.append(encodings.char_to_token(i, answers[i]["answer_end"] - 1))
            # check if page is truncated
            if start_positions[-1] is None:
                start_positions[-1] = self.tokenizer.model_max_length
            if end_positions[-1] is None:
                end_positions[-1] = self.tokenizer.model_max_length
        encodings.update({"start_positions": start_positions,
                          "end_positions": end_positions})


    def preparation(self, train_answers, train_contexts, train_questions):
        self.add_end_idx(train_answers, train_contexts)
        #self.add_end_idx(valid_answers, valid_contexts)

        train_encodings = self.tokenizer(train_contexts, train_questions, truncation=True, padding=True)
        #valid_encodings = self.tokenizer(valid_contexts, valid_questions, truncation=True, padding=True)

        self.add_token_positions(train_encodings, train_answers)
        #self.add_token_positions(valid_encodings, valid_answers)
        # create datasets
        self.train_set = SQuAD_Dataset(train_encodings)
        #self.valid_set = SQuAD_Dataset(valid_encodings)

        self.train_loader = DataLoader(self.train_set, batch_size=15, shuffle=True)
        self.train_loader = DataLoader(self.train_set, batch_size=15, shuffle=True)
        #self.valid_loader = DataLoader(self.valid_set, batch_size=2)
        #self.train_loader = DataLoader(self.train_set)
        #self.valid_loader = DataLoader(self.valid_set)


    def train(self):
        self.model.to(self.device)
        self.model.train()
        optim = AdamW(self.model.parameters(), lr=5e-5)
        for epoch in range(self.epochs):
            loop = tqdm(self.train_loader, leave=True)
            for batch in loop:
                optim.zero_grad()
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                start_positions = batch['start_positions'].to(self.device)
                end_positions = batch['end_positions'].to(self.device)
                outputs = self.model(input_ids, attention_mask=attention_mask, start_positions=start_positions,
                                end_positions=end_positions)
                loss = outputs[0]
                loss.backward()
                optim.step()

                loop.set_description(f'Epoch {epoch + 1}')
                loop.set_postfix(loss=loss.item())
        self.model = self.model.eval()

    # def train(self):
    #     N_EPOCHS = self.epochs
    #     self.model.to(self.device)
    #     self.model.train()
    #     optim = AdamW(self.model.parameters(), lr=5e-5)
    #     for epoch in range(N_EPOCHS):
    #         for batch in self.train_loader:
    #             optim.zero_grad()
    #             input_ids = batch["input_ids"].to(self.device)
    #             attention_mask = batch["attention_mask"].to(self.device)
    #             start_positions = batch["start_positions"].to(self.device)
    #             end_positions = batch["end_positions"].to(self.device)
    #             outputs = self.model(input_ids,
    #                             attention_mask=attention_mask,
    #                             start_positions=start_positions,
    #                             end_positions=end_positions)
    #             loss = outputs[0]
    #             loss.backward()
    #
    #             optim.step()
    #         print('epoch '+str(epoch))
    #     self.model = self.model.eval()

    def predict(self, question, text):
        inputs = self.tokenizer.encode_plus(question, text, return_tensors='pt').to(self.device)
        outputs = self.model(**inputs)

        answer_start = torch.argmax(outputs[0])
        answer_end = torch.argmax(outputs[1]) + 1

        answer = self.tokenizer.convert_tokens_to_string(
            self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0][answer_start:answer_end]))

        return answer
