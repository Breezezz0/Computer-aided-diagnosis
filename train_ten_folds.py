import io
import os
import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, accuracy_score
from transformers import (set_seed,
                          TrainingArguments,
                          Trainer,
                          GPT2Config,
                          GPT2Model,
                          GPT2Tokenizer,
                          AdamW, 
                          get_linear_schedule_with_warmup,
                          GPT2ForSequenceClassification)
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from Preprocessing_text_data import preprocess_data_and_get_index, patient_inf_v2
import random

#做dataset
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token
# labels = {'460': 0, '461.9': 1, '465.9': 2,'466' : 3}
labels = {'460': 0,  '465.9': 1}
fig_dir = "/home/u109001022/auto_diag/result_0327_tenfolds/"

class Dataset(torch.utils.data.Dataset):
    def __init__(self, dict):
        self.keys = dict.keys()
        self.texts = []
        self.labels = []
    # Since the labels are defined by folders with data we loop 
    # through each label.
        for keys in self.keys :
            self.texts.append(dict[keys]['徵侯'])
            # for label in ['460', '461.9', '465.9', '466'] :
            for label in ['460', '465.9'] :
                for diag in dict[keys]['診斷碼'] :
                    if str(diag) == str(label) :
                        self.labels.append(label)
                        break
                if str(diag) == str(label) :
                    break
        # Number of exmaples.
        self.n_examples = len(self.labels)
        self.labels = [labels[label] for label in self.labels]
        self.texts = [tokenizer(text,
                                padding='max_length',
                                max_length=256,
                                truncation=True,
                                return_tensors="pt") for text in self.texts]
        
    def classes(self):
        return self.labels
    
    def __len__(self):
        return len(self.labels)
    
    def get_batch_labels(self, idx):
        # Get a batch of labels
        return np.array(self.labels[idx])
    
    def get_batch_texts(self, idx):
        # Get a batch of inputs
        return self.texts[idx]
    
    def __getitem__(self, idx):
        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)
        return batch_texts, batch_y

#多疾病預測的資料集
class Dataset_V2(torch.utils.data.Dataset):
    def __init__(self, dict):
        self.keys = dict.keys()
        self.texts = []
        self.labels = []
    # Since the labels are defined by dict, we loop 
    # through each label.
        for keys in self.keys :
            diag_code = []
            self.texts.append(dict[keys]['徵侯'])
            for label in ['460', '461.9', '465.9', '466.0'] :
            #for label in ['460', '465.9'] :
                for diag in dict[keys]['診斷碼'] :
                    if str(diag) == str(label) :
                        diag_code.append(1)
                        break
                if str(diag) != str(label) :
                    diag_code.append(0)
            self.labels.append(diag_code)
        # Number of exmaples.
        self.n_examples = len(self.labels)
        # self.labels = [labels[label] for label in self.labels]
        self.texts = [tokenizer(text,
                                padding='max_length',
                                max_length=256,
                                truncation=True,
                                return_tensors="pt") for text in self.texts]
        
    def classes(self):
        return self.labels
    
    def __len__(self):
        return len(self.labels)
    
    def get_batch_labels(self, idx):
        # Get a batch of labels
        return np.array(self.labels[idx])
    
    def get_batch_texts(self, idx):
        # Get a batch of inputs
        return self.texts[idx]
    
    def __getitem__(self, idx):
        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)
        return batch_texts, batch_y

#model
class SimpleGPT2SequenceClassifier(nn.Module):
    def __init__(self, hidden_size: int, num_classes:int ,max_seq_len:int, gpt_model_name:str):
        super(SimpleGPT2SequenceClassifier,self).__init__()
        self.gpt2model = GPT2Model.from_pretrained(gpt_model_name)
        self.fc1 = nn.Linear(hidden_size*max_seq_len, num_classes)

        
    def forward(self, input_id, mask):
        """
        Args:
                input_id: encoded inputs ids of sent.
        """
        gpt_out, _ = self.gpt2model(input_ids=input_id, attention_mask=mask, return_dict=False)
        batch_size = gpt_out.shape[0]
        linear_output = self.fc1(gpt_out.view(batch_size,-1))
        return linear_output
    
#model for 多疾病分類(二元)
class SimpleGPT2SequenceClassifier_V2(nn.Module):
    def __init__(self, hidden_size: int, num_classes:int ,max_seq_len:int, gpt_model_name:str):
        super(SimpleGPT2SequenceClassifier_V2,self).__init__()
        self.gpt2model = GPT2Model.from_pretrained(gpt_model_name)
        self.fc1 = nn.Linear(hidden_size*max_seq_len, num_classes)
        self.fc2 = nn.Linear(hidden_size*max_seq_len, num_classes)
        self.fc3 = nn.Linear(hidden_size*max_seq_len, num_classes)
        self.fc4 = nn.Linear(hidden_size*max_seq_len, num_classes)

        
    def forward(self, input_id, mask):
        """
        Args:
                input_id: encoded inputs ids of sent.
        """
        gpt_out, _ = self.gpt2model(input_ids=input_id, attention_mask=mask, return_dict=False)
        batch_size = gpt_out.shape[0]
        linear_output1 = self.fc1(gpt_out.view(batch_size,-1))
        linear_output2 = self.fc2(gpt_out.view(batch_size,-1))
        linear_output3 = self.fc3(gpt_out.view(batch_size,-1))
        linear_output4 = self.fc4(gpt_out.view(batch_size,-1))
        return linear_output1 , linear_output2, linear_output3, linear_output4

def train(model, train_data, val_data, learning_rate, epochs):
    train, val = Dataset(train_data), Dataset(val_data)
    
    train_dataloader = torch.utils.data.DataLoader(train, batch_size=4, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val, batch_size=4)
    
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)
    
    if use_cuda:
        model = model.cuda()
        criterion = criterion.cuda()

    for epoch_num in range(epochs):
        total_acc_train = 0
        total_loss_train = 0
        
        for train_input, train_label in tqdm(train_dataloader):
            #train_label = train_label.to(device).to(torch.int64)
            train_label = train_label.type(torch.LongTensor).to(device)

            mask = train_input['attention_mask'].to(device)
            input_id = train_input["input_ids"].squeeze(1).to(device)
            
            model.zero_grad()

            output = model(input_id, mask)
            
            batch_loss = criterion(output, train_label)
            total_loss_train += batch_loss.item()
            
            acc = (output.argmax(dim=1)==train_label).sum().item()
            total_acc_train += acc

            batch_loss.backward()
            optimizer.step()
            
        total_acc_val = 0
        total_loss_val = 0
        
        with torch.no_grad():
            
            for val_input, val_label in val_dataloader:
                val_label = val_label.type(torch.LongTensor).to(device)
                mask = val_input['attention_mask'].to(device)
                input_id = val_input['input_ids'].squeeze(1).to(device)
                
                output = model(input_id, mask)
                
                batch_loss = criterion(output, val_label)
                total_loss_val += batch_loss.item()
                
                acc = (output.argmax(dim=1)==val_label).sum().item()
                total_acc_val += acc
                
            print(
            f"Epochs: {epoch_num + 1} | Train Loss: {total_loss_train/len(train_data): .3f} \
            | Train Accuracy: {total_acc_train / len(train_data): .3f} \
            | Val Loss: {total_loss_val / len(val_data): .3f} \
            | Val Accuracy: {total_acc_val / len(val_data): .3f}")




#個別訓練疾病種類
def train_V2(model, train_data, val_data, learning_rate, epochs):
    train, val = Dataset_V2(train_data), Dataset_V2(val_data)
    
    train_dataloader = torch.utils.data.DataLoader(train, batch_size=4, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val, batch_size=4)
    
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)
    history_train_acc = []
    history_valid_acc = []
    history_train_loss = []
    history_validation_loss = []
    if use_cuda:
        model = model.cuda()
        criterion = criterion.cuda()

    for epoch_num in range(epochs):
        total_acc_train_1 = 0
        total_acc_train_2 = 0
        total_acc_train_3 = 0
        total_acc_train_4 = 0
        total_loss_train = 0
        
        for train_input, train_label in tqdm(train_dataloader):
            #train_label = train_label.to(device).to(torch.int64)
            train_label = train_label.type(torch.LongTensor).to(device)

            mask = train_input['attention_mask'].to(device)
            input_id = train_input["input_ids"].squeeze(1).to(device)
            
            model.zero_grad()

            output1, output2, output3, output4  = model(input_id, mask)
            
            batch_loss1 = criterion(output1, train_label[:,0])
            batch_loss2 = criterion(output2, train_label[:,1])
            batch_loss3 = criterion(output3, train_label[:,2])
            batch_loss4 = criterion(output4, train_label[:,3])
            total_loss = (batch_loss1 + batch_loss2 + batch_loss3 + batch_loss4)/4
            total_loss_train += total_loss.item()
            
            acc1 = (output1.argmax(dim=1)==train_label[:,0]).sum().item()
            acc2 = (output2.argmax(dim=1)==train_label[:,1]).sum().item()
            acc3 = (output3.argmax(dim=1)==train_label[:,2]).sum().item()
            acc4 = (output4.argmax(dim=1)==train_label[:,3]).sum().item()
            total_acc_train_1 += acc1
            total_acc_train_2 += acc2
            total_acc_train_3 += acc3
            total_acc_train_4 += acc4

            total_loss.backward()
            optimizer.step()
        total_train_acc = [total_acc_train_1, total_acc_train_2, total_acc_train_3, total_acc_train_4]
        history_train_acc.append(total_train_acc)

        total_acc_val_1 = 0
        total_acc_val_2 = 0
        total_acc_val_3 = 0
        total_acc_val_4 = 0
        total_loss_val = 0
        
        with torch.no_grad():
            
            for val_input, val_label in val_dataloader:
                val_label = val_label.type(torch.LongTensor).to(device)
                mask = val_input['attention_mask'].to(device)
                input_id = val_input['input_ids'].squeeze(1).to(device)
                
                output1, output2, output3, output4  = model(input_id, mask)
            
                batch_loss1 = criterion(output1, val_label[:,0])
                batch_loss2 = criterion(output2, val_label[:,1])
                batch_loss3 = criterion(output3, val_label[:,2])
                batch_loss4 = criterion(output4, val_label[:,3])
                total_loss = (batch_loss1 + batch_loss2 + batch_loss3 + batch_loss4)/4
                total_loss_val += total_loss.item()
                
                acc1 = (output1.argmax(dim=1)==val_label[:,0]).sum().item()
                acc2 = (output2.argmax(dim=1)==val_label[:,1]).sum().item()
                acc3 = (output3.argmax(dim=1)==val_label[:,2]).sum().item()
                acc4 = (output4.argmax(dim=1)==val_label[:,3]).sum().item()
                total_acc_val_1 += acc1
                total_acc_val_2 += acc2
                total_acc_val_3 += acc3
                total_acc_val_4 += acc4
            total_valid_acc = [total_acc_val_1, total_acc_val_2, total_acc_val_3, total_acc_val_4]
            history_valid_acc.append(total_valid_acc)
                
                
            print(
            f"Epochs: {epoch_num + 1} | Train Loss: {total_loss_train/len(train_data): .3f} \
            | Train Accuracy1: {total_acc_train_1 / len(train_data): .3f} \
            | Train Accuracy2: {total_acc_train_2 / len(train_data): .3f} \
            | Train Accuracy3: {total_acc_train_3 / len(train_data): .3f} \
            | Train Accuracy4: {total_acc_train_4 / len(train_data): .3f} \
            | Val Loss: {total_loss_val / len(val_data): .3f} \
            | Val Accuracy1: {total_acc_val_1 / len(val_data): .3f} \
            | Val Accuracy2: {total_acc_val_2 / len(val_data): .3f} \
            | Val Accuracy3: {total_acc_val_3 / len(val_data): .3f} \
            | Val Accuracy4: {total_acc_val_4 / len(val_data): .3f} ")
            history_train_loss.append(total_loss_train/len(train_data))
            history_validation_loss.append(total_loss_val / len(val_data))
    return history_train_acc, history_valid_acc, history_train_loss, history_validation_loss

def evaluate(model, test_data):

        test = Dataset(test_data)

        test_dataloader = torch.utils.data.DataLoader(test, batch_size=4)

        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")

        if use_cuda:

            model = model.cuda()

            
        # Tracking variables
        predictions_labels = []
        true_labels = []
        
        total_acc_test = 0
        with torch.no_grad():

            for test_input, test_label in test_dataloader:

                test_label = test_label.to(device)
                mask = test_input['attention_mask'].to(device)
                input_id = test_input['input_ids'].squeeze(1).to(device)

                output = model(input_id, mask)

                acc = (output.argmax(dim=1) == test_label).sum().item()
                total_acc_test += acc
                
                # add original labels
                true_labels += test_label.cpu().numpy().flatten().tolist()
                # get predicitons to list
                predictions_labels += output.argmax(dim=1).cpu().numpy().flatten().tolist()
        
        print(f'Test Accuracy: {total_acc_test / len(test_data): .3f}')
        return true_labels, predictions_labels


def evaluate_V2(model, test_data):

    test = Dataset_V2(test_data)

    test_dataloader = torch.utils.data.DataLoader(test, batch_size=16)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:

        model = model.cuda()

        
    # Tracking variables
    predictions_labels_1 = []
    predictions_labels_2 = []
    predictions_labels_3 = []
    predictions_labels_4 = []
    true_labels_1 = []
    true_labels_2 = []
    true_labels_3 = []
    true_labels_4 = []
    
    total_acc_test_1 = 0
    total_acc_test_2 = 0
    total_acc_test_3 = 0
    total_acc_test_4 = 0
    with torch.no_grad():

        for test_input, test_label in test_dataloader:

            test_label = test_label.to(device)
            mask = test_input['attention_mask'].to(device)
            input_id = test_input['input_ids'].squeeze(1).to(device)

            output1, output2, output3, output4  = model(input_id, mask)
            acc1 = (output1.argmax(dim=1)==test_label[:,0]).sum().item()
            acc2 = (output2.argmax(dim=1)==test_label[:,1]).sum().item()
            acc3 = (output3.argmax(dim=1)==test_label[:,2]).sum().item()
            acc4 = (output4.argmax(dim=1)==test_label[:,3]).sum().item()
            total_acc_test_1 += acc1
            total_acc_test_2 += acc2
            total_acc_test_3 += acc3
            total_acc_test_4 += acc4
            
            # add original labels
            true_labels_1 += test_label[:,0].cpu().numpy().flatten().tolist()
            true_labels_2 += test_label[:,1].cpu().numpy().flatten().tolist()
            true_labels_3 += test_label[:,2].cpu().numpy().flatten().tolist()
            true_labels_4 += test_label[:,3].cpu().numpy().flatten().tolist()
            # get predicitons to list
            predictions_labels_1 += output1.argmax(dim=1).cpu().numpy().flatten().tolist()
            predictions_labels_2 += output2.argmax(dim=1).cpu().numpy().flatten().tolist()
            predictions_labels_3 += output3.argmax(dim=1).cpu().numpy().flatten().tolist()
            predictions_labels_4 += output4.argmax(dim=1).cpu().numpy().flatten().tolist()
    total_acc = ((total_acc_test_1+total_acc_test_2+total_acc_test_3+total_acc_test_4) /4)/len(test_data)
    print(f'Test Accuracy: {((total_acc_test_1+total_acc_test_2+total_acc_test_3+total_acc_test_4) /4)/len(test_data): .3f}')
    return [true_labels_1,true_labels_2,true_labels_3,true_labels_4], [predictions_labels_1,predictions_labels_2,predictions_labels_3,predictions_labels_4], total_acc   



data_path = [r"/home/u109001022/auto_diag/111data.csv", r"/home/u109001022/auto_diag/110data.csv", r"/home/u109001022/auto_diag/109data.csv", r"/home/u109001022/auto_diag/108data.csv", r"/home/u109001022/auto_diag/107data.csv", r"/home/u109001022/auto_diag/106data.csv"]
dataset_inf = {}
for data in data_path :
    df_path = data
    df_abbr_path = 'abbreviation.xlsx'
    df_preprocessed , index_eng = preprocess_data_and_get_index(df_path, df_abbr_path)
    for date, patient in index_eng :
        key = str(date)+str(patient)
        dataset_inf[key] = patient_inf_v2(df_preprocessed, patient, date).get_inf()

#remove useless data
dataset_emr = {}
for key in dataset_inf.keys() :
    for diagnosis in [460, 461.9, 465.9, 466.0] :
    # for diagnosis in [460, 465.9] :
        for code in dataset_inf[key]['診斷碼'] :
            if code == str(diagnosis) :
                dataset_emr[key] = dataset_inf[key]
                break
        if code == str(diagnosis) :
            break

all_train_acc = []
all_valid_acc = []
all_test_acc = []


for fold in range(10) :
        
    #做資料分割
    print(len(dataset_emr.keys()))
    dataset_train = {}
    dataset_valid = {}
    dataset_test = {}
    num_train = 14148
    num_valid = 4042
    num_test = 2020
    index_for_train_and_valid = []
    for num in range(len(dataset_emr.keys())) :
        if  (num < num_test*fold) or (num >= num_test*(fold+1)) :
            index_for_train_and_valid.append(num)
    index_for_train = random.sample(index_for_train_and_valid,num_train)
    for num,keys in enumerate(dataset_emr.keys()) :
        if (num >= num_test*fold) and (num < num_test*(fold+1)):
            dataset_test[keys] = dataset_emr[keys]
        elif num in index_for_train :
            dataset_train[keys] = dataset_emr[keys]
        else :
            dataset_valid[keys] = dataset_emr[keys]
    print(len(dataset_train.keys()))
    print(len(dataset_valid.keys()))
    print(len(dataset_test.keys()))

    EPOCHS = 10
    model = SimpleGPT2SequenceClassifier_V2(hidden_size=768, num_classes=2, max_seq_len=256, gpt_model_name="gpt2")
    LR = 1e-5

    train_acc, valid_acc, train_loss, valid_loss = train_V2(model, dataset_train, dataset_valid, LR, EPOCHS)
    torch.save(model.state_dict(), f"{fig_dir}GPT2-model_0327_{fold}.pth")
    total_train_acc = 0 
    for i in train_acc :
        for j in i :
            total_train_acc += j
    total_train_acc = total_train_acc/40
    total_valid_acc = 0 
    for i in valid_acc :
        for j in i :
            total_valid_acc += j
    total_valid_acc = total_valid_acc/40
    all_train_acc.append(total_train_acc)
    all_valid_acc.append(total_valid_acc)
    #繪製訓練過程
    labels = ['nasopharyngitis', 'sinusitis', 'respiratory infection', 'bronchitis&Bronchiolitis']
    colors = ['red','blue', 'purple', 'green']
    x = [i for i in range(1,EPOCHS+1)]
    for i in range(4) :
        y = []
        for j in range(len(train_acc)) :
            y.append(train_acc[j][i]/len(dataset_train))
        plt.plot(x, y, color=colors[i], linestyle="-", linewidth="2", markersize="16", marker=".", label=labels[i])
    plt.legend()
    plt.title('train_acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.yticks(np.arange(0.5, 1, step=0.05))
    plt.savefig(f'train_acc_{fold}.png')
    plt.close()

    #繪製validation過程
    labels = ['nasopharyngitis', 'sinusitis', 'respiratory infection', 'bronchitis&Bronchiolitis']
    colors = ['red','blue', 'purple', 'green']
    x = [i for i in range(1,EPOCHS+1)]
    for i in range(4) :
        y = []
        for j in range(len(valid_acc)) :
            y.append(valid_acc[j][i]/len(dataset_valid))
        plt.plot(x, y, color=colors[i], linestyle="-", linewidth="2", markersize="16", marker=".", label=labels[i])
    plt.legend()
    plt.title('validation_acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.yticks(np.arange(0.5, 1, step=0.05))
    plt.savefig(f'valid_acc_{fold}.png')
    plt.close()

    #繪製loss過程
    x = [i for i in range(1,EPOCHS+1)]
    plt.plot(x, train_loss, color='red', linestyle="-", linewidth="2", markersize="16", marker=".", label='Train_loss')
    plt.plot(x, valid_loss, color='blue', linestyle="-", linewidth="2", markersize="16", marker=".", label='Validation_loss')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(f'{fig_dir}loss_{fold}.png')
    plt.close()

    
    #true_labels, pred_labels = evaluate_V2(model, dataset_test)
    true_labels, pred_labels, test_acc = evaluate_V2(model, dataset_test)
    all_test_acc.append(test_acc)

    # Plot confusion matrix.
    labels = {'True':1 ,'False':0}
    fig, ax = plt.subplots(2,2,figsize=(10, 10))
    for i in range(4) :
        cm = confusion_matrix(y_true=true_labels[i], y_pred=pred_labels[i], labels=range(len(labels)))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(labels.keys()))
        disp.plot(ax=ax[i//2][i%2])
    plt.savefig(f'{fig_dir}confusion_matrix_{fold}.png')
    plt.close()

    acc_count = [] 
    for i in range(len(true_labels[0])) :
        count_class= 0 #得幾種病
        count_acc= 0  #跟預測比
        for j in range(4) :
            count_class += true_labels[j][i]
            if true_labels[j][i] == pred_labels[j][i] :
                count_acc +=1
        acc_count.append([str(count_class),str(count_acc)])

    #計算個別的正確率
    acc_dict = {'1':{}, '2':{}, '3':{}, '4':{}}
    for i in acc_count :
        acc_dict[i[0]][i[1]] = acc_dict[i[0]].get(i[1],0) +1
    print('acc_dict= ', acc_dict)

    #統計dataset
    dataset_count = Dataset_V2(dataset_train)
    diag_set_count = {}
    for i in range(len(dataset_count)) :
        _, target = dataset_count[i] 
        target = np.array2string(target)
        diag_set_count[target] = diag_set_count.get(target, 0) + 1
    print(diag_set_count)


    key_dict = {'[1 0 0 0]' : 'nasopharyngitis','[0 1 0 0]' : 'sinusitis', '[0 0 1 0]' : 'respiratory infection', 
                '[0 0 0 1]' : 'bronchitis&Bronchiolitis', '[1 0 1 0]' : 'nasopharyngitis,respiratory infection','[1 1 0 0]' : 'sinusitis,nasopharyngitis',
                '[1 0 0 1]' : 'nasopharyngitis,bronchitis&Bronchiolitis', '[0 1 1 0]' : 'sinusitis,respiratory infection',
                '[0 1 0 1]' : 'sinusitis,bronchitis&Bronchiolitis',
                '[0 0 1 1]' : 'respiratory infection,bronchitis&Bronchiolitis', '[0 1 1 1]' : 'sinusitis,respiratory infection,bronchitis&Bronchiolitis', 
                '[1 0 1 1]' : 'nasopharyngitis,respiratory infection,bronchitis&Bronchiolitis',
                '[1 1 0 1]' : 'sinusitis,nasopharyngitis,bronchitis&Bronchiolitis', '[1 1 1 0]' : 'sinusitis,nasopharyngitis,respiratory infection' }
    x = [i for i in range(1,len(diag_set_count.keys())+1)]
    x_labels = [x[0] for x in diag_set_count.items()]
    y = [x[1] for x in diag_set_count.items()]
    plt.bar(x, y, color='blue', linewidth=2)
    plt.xlabel('Disease')
    plt.ylabel('Number')
    plt.xticks(x,x_labels,size='small')
    for a,b in zip(x,y) :
        plt.text(a, b+0.1, '%.0f' % b, ha='center', va= 'bottom',fontsize=12)
    plt.savefig(f'{fig_dir}dataset_train_{fold}.png')
    plt.close()

    dataset_count = Dataset_V2(dataset_valid)
    diag_set_count = {}
    for i in range(len(dataset_count)) :
        _, target = dataset_count[i] 
        target = np.array2string(target)
        diag_set_count[target] = diag_set_count.get(target, 0) + 1
    print(diag_set_count)


    key_dict = {'[1 0 0 0]' : 'nasopharyngitis','[0 1 0 0]' : 'sinusitis', '[0 0 1 0]' : 'respiratory infection', 
                '[0 0 0 1]' : 'bronchitis&Bronchiolitis', '[1 0 1 0]' : 'nasopharyngitis,respiratory infection','[1 1 0 0]' : 'sinusitis,nasopharyngitis',
                '[1 0 0 1]' : 'nasopharyngitis,bronchitis&Bronchiolitis', '[0 1 1 0]' : 'sinusitis,respiratory infection',
                '[0 1 0 1]' : 'sinusitis,bronchitis&Bronchiolitis',
                '[0 0 1 1]' : 'respiratory infection,bronchitis&Bronchiolitis', '[0 1 1 1]' : 'sinusitis,respiratory infection,bronchitis&Bronchiolitis', 
                '[1 0 1 1]' : 'nasopharyngitis,respiratory infection,bronchitis&Bronchiolitis',
                '[1 1 0 1]' : 'sinusitis,nasopharyngitis,bronchitis&Bronchiolitis', '[1 1 1 0]' : 'sinusitis,nasopharyngitis,respiratory infection' }
    x = [i for i in range(1,len(diag_set_count.keys())+1)]
    x_labels = [x[0] for x in diag_set_count.items()]
    y = [x[1] for x in diag_set_count.items()]
    plt.bar(x, y, color='blue', linewidth=2)
    plt.xlabel('Disease')
    plt.ylabel('Number')
    plt.xticks(x,x_labels,size='small')
    for a,b in zip(x,y) :
        plt.text(a, b+0.1, '%.0f' % b, ha='center', va= 'bottom',fontsize=12)
    plt.savefig(f'{fig_dir}dataset_valid_{fold}.png')
    plt.close()

    dataset_count = Dataset_V2(dataset_test)
    diag_set_count = {}
    for i in range(len(dataset_count)) :
        _, target = dataset_count[i] 
        target = np.array2string(target)
        diag_set_count[target] = diag_set_count.get(target, 0) + 1
    print(diag_set_count)


    key_dict = {'[1 0 0 0]' : 'nasopharyngitis','[0 1 0 0]' : 'sinusitis', '[0 0 1 0]' : 'respiratory infection', 
                '[0 0 0 1]' : 'bronchitis&Bronchiolitis', '[1 0 1 0]' : 'nasopharyngitis,respiratory infection','[1 1 0 0]' : 'sinusitis,nasopharyngitis',
                '[1 0 0 1]' : 'nasopharyngitis,bronchitis&Bronchiolitis', '[0 1 1 0]' : 'sinusitis,respiratory infection',
                '[0 1 0 1]' : 'sinusitis,bronchitis&Bronchiolitis',
                '[0 0 1 1]' : 'respiratory infection,bronchitis&Bronchiolitis', '[0 1 1 1]' : 'sinusitis,respiratory infection,bronchitis&Bronchiolitis', 
                '[1 0 1 1]' : 'nasopharyngitis,respiratory infection,bronchitis&Bronchiolitis',
                '[1 1 0 1]' : 'sinusitis,nasopharyngitis,bronchitis&Bronchiolitis', '[1 1 1 0]' : 'sinusitis,nasopharyngitis,respiratory infection' }
    x = [i for i in range(1,len(diag_set_count.keys())+1)]
    x_labels = [x[0] for x in diag_set_count.items()]
    y = [x[1] for x in diag_set_count.items()]
    plt.bar(x, y, color='blue', linewidth=2)
    plt.xlabel('Disease')
    plt.ylabel('Number')
    plt.xticks(x,x_labels,size='small')
    for a,b in zip(x,y) :
        plt.text(a, b+0.1, '%.0f' % b, ha='center', va= 'bottom',fontsize=12)
    plt.savefig(f'{fig_dir}dataset_test_{fold}.png')
    plt.close()
print('train_acc = ', ' '.join(all_train_acc))
print('valid_acc = ', ' '.join(all_valid_acc))
print('test_acc = ', ' '.join(all_test_acc))
df_acc = pd.DataFrame(list(zip(all_train_acc,all_valid_acc,all_test_acc)), columns =['Train', 'Valid', 'Test'])
df_acc.to_excel(f'{fig_dir}accuracy.xlsx')