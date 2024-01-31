import argparse
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertModel
from torchvision import transforms
from torchvision.models import resnet50
import torch.nn.functional as F


#数据处理
tokenizer = BertTokenizer.from_pretrained('./bert-base-uncased')
transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

train_data = pd.read_csv('train.txt')
label = {"positive": 0, "negative": 1,"neutral":2}
train_dataset = []
for guid,tag in train_data.values:
    image_path = './data/' + str(guid) + '.jpg'
    text_path = './data/' + str(guid) + '.txt'

    image = Image.open(image_path)
    image = transform(image)

    with open(text_path, 'r', encoding='gb18030', errors='replace') as f:
        text = f.readline().strip()
        tokenized_text = tokenizer(text,padding='max_length',max_length=100,truncation=True,return_tensors="pt")
        input_ids = tokenized_text['input_ids']
        attention_mask = tokenized_text['attention_mask']

    data = [torch.tensor(int(guid), dtype=torch.long), image, input_ids, attention_mask, torch.tensor(label[tag], dtype=torch.long)]
    train_dataset.append(data)
    
train_dataset, val_dataset = train_test_split(train_dataset, test_size=0.2, random_state=47)
loader_train = DataLoader(train_dataset, batch_size=16)
loader_val = DataLoader(val_dataset, batch_size=16)


#test_without_label
test_data = pd.read_csv('test_without_label.txt')
test_dataset = []
for guid, _ in test_data.values:
    image_path = './data/' + str(int(guid)) + '.jpg'
    text_path = './data/' + str(int(guid)) + '.txt'

    image = Image.open(image_path)
    image = transform(image)

    with open(text_path, 'r', encoding='gb18030', errors='replace') as f:
        text = f.readline().strip()
        tokenized_text = tokenizer(text, padding='max_length', max_length=100, truncation=True, return_tensors="pt")
        input_ids = tokenized_text['input_ids']
        attention_mask = tokenized_text['attention_mask']

    data = [torch.tensor(int(guid), dtype=torch.long), image, input_ids, attention_mask]
    test_dataset.append(data)
loader_test = DataLoader(test_dataset, batch_size=16)



USE_GPU = True

dtype = torch.float32

if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print('using device:', device)


#模型
class img_feature(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = resnet50(pretrained=True)
    
    def forward(self, image):
        output = self.resnet(image)
        return output
    
class pre_img_feature(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = resnet50(pretrained=True)
        m = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*m)
        self.pool = nn.AdaptiveAvgPool2d((3,1))
        self.linear = nn.Linear(2048, 768)
    
    def forward(self, image):
        output = self.resnet(image)
        output = self.pool(output)
        output = torch.flatten(output, start_dim=2)
        output = output.transpose(1, 2).contiguous()  #batchsize*3*2048
        output = self.linear(output)
        return output

class text_feature(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained('./bert-base-uncased')

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        output = outputs[1]
        return output
    
class pre_text_feature(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained('./bert-base-uncased')

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        token_output = outputs.last_hidden_state
        return token_output
    
class Model1(nn.Module):
    def __init__(self, num = 3):
        super().__init__()
        self.image_feature = img_feature()
        self.text_feature = text_feature()
        self.classifier = nn.Sequential(
            nn.Dropout(p = 0.4),
            nn.Linear(1768, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p = 0.4),
            nn.Linear(512, num),
        )

    def forward(self, image, input_ids, attention_mask):
        image_features = self.image_feature(image)
        text_features = self.text_feature(input_ids,attention_mask)
        all_features = torch.cat((text_features,image_features), dim=-1)
        output = self.classifier(all_features)
        return output
    
class Model2(nn.Module):
    def __init__(self, num = 3):
        super().__init__()
        self.image_feature = img_feature()
        self.text_feature = text_feature()
        self.encoder = nn.TransformerEncoderLayer(d_model=1768, nhead=8, dropout= 0.4)
        self.classifier = nn.Sequential(
            nn.Dropout(p = 0.4),
            nn.Linear(1768, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p = 0.4),
            nn.Linear(512, num),
        )

    def forward(self, image, input_ids, attention_mask):
        image_features = self.image_feature(image)
        text_features = self.text_feature(input_ids,attention_mask)
        all_features = torch.cat((text_features,image_features), dim=-1)
        encoded_layers = self.encoder(all_features)
        output = self.classifier(encoded_layers)
        return output
    
class Model3(nn.Module):
    def __init__(self, num = 3):
        super().__init__()
        self.image_feature = pre_img_feature()
        self.text_feature = pre_text_feature()
        self.encoder = nn.TransformerEncoderLayer(d_model=768, nhead=8, dropout= 0.4)
        self.classifier = nn.Sequential(
            nn.Dropout(p = 0.4),
            nn.Linear(768, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p = 0.4),
            nn.Linear(128, num),
        )

    def forward(self, image, input_ids, attention_mask):
        image_features = self.image_feature(image)
        text_features = self.text_feature(input_ids,attention_mask)
        all_features = torch.cat((text_features,image_features), dim=1) #batchsize*100*768
        encoded_layers = self.encoder(all_features)
        result = torch.mean(encoded_layers, dim=1).squeeze(0)
        output = self.classifier(result)
        return output

class Model_only_img(nn.Module):
    def __init__(self, num = 3):
        super().__init__()
        self.image_feature = img_feature()
        self.classifier = nn.Sequential(
            nn.Dropout(p = 0.4),
            nn.Linear(1000, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p = 0.4),
            nn.Linear(128, num),
        )

    def forward(self, image, input_ids, attention_mask):
        image_features = self.image_feature(image)
        output = self.classifier(image_features)
        return output

class Model_only_text(nn.Module):
    def __init__(self, num = 3):
        super().__init__()
        self.text_feature = text_feature()
        self.classifier = nn.Sequential(
            nn.Dropout(p = 0.4),
            nn.Linear(768, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p = 0.4),
            nn.Linear(128, num),
        )

    def forward(self, image, input_ids, attention_mask):
        text_features = self.text_feature(input_ids,attention_mask)
        output = self.classifier(text_features)
        return output  

    
#训练和验证
def check_accuracy_part(loader, model):
    print('Checking accuracy on validation set')
    num_correct = 0
    num_samples = 0
    model.eval()
    with torch.no_grad():
        for _, images, input_ids, attention_mask, labels in loader:
            images = images.to(device, dtype = torch.float32)
            input_ids = input_ids.squeeze(1).to(device)
            attention_mask = attention_mask.to(device)     
            labels = labels.to(device, torch.long)

            outputs = model(images, input_ids, attention_mask)
            _, preds = torch.max(outputs, 1)
            num_correct += torch.sum(preds == labels)
            num_samples += preds.size(0)

        acc = float(num_correct) / num_samples
        print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))

def train_part(model, optimizer, epochs):
    model = model.to(device=device)
    for e in range(epochs):
        model.train()
        t = 0
        for _, images, input_ids, attention_mask, labels in loader_train:
            images = images.to(device, dtype = torch.float32)
            input_ids = input_ids.squeeze(1).to(device)
            attention_mask = attention_mask.to(device)     
            labels = labels.to(device, dtype = torch.long)  

            outputs = model(images, input_ids, attention_mask)
            loss = F.cross_entropy(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            t += 1
            if t % 5 == 0:
                print('Iteration %d, loss = %.4f' % (t, loss.item()))
                print()
            if t % 100 == 0:    
                check_accuracy_part(loader_val, model)

def predict_tags(loader, model):
    model.eval()
    predictions = []
    with torch.no_grad():
        for _, images, input_ids, attention_mask in loader:
            images = images.to(device, dtype=torch.float32)
            input_ids = input_ids.squeeze(1).to(device)
            attention_mask = attention_mask.to(device)

            outputs = model(images, input_ids, attention_mask)
            _, preds = torch.max(outputs, 1)
            predictions.extend(preds.cpu().numpy())
    return predictions

def save_predictions_to_file(guids, predictions, filename='result1.txt'):
    with open(filename, 'w') as f:
        f.write('guid,tag\n')
        for guid, pred in zip(guids, predictions):
            tag = {0: 'positive', 1: 'negative', 2: 'neutral'}[pred]
            f.write(f'{guid},{tag}\n')

#运行
def parse_args():
    parser = argparse.ArgumentParser(description="Choose model and hyperparameters")
    parser.add_argument("--model", type=str, default="3", choices=["concat", "encode_concat", "pre_concat", "only_img", "only_text"],
                        help="Choose the model architecture (concat, encode_concat, pre_concat, only_img, only_text)")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=2, help="Number of training epochs")
    return parser.parse_args()


args = parse_args()

if args.model == "concat":
    model = Model1()
elif args.model == "encode_concat":
    model = Model2()
elif args.model == "pre_concat":
    model = Model3()
elif args.model == "only_img":
    model = Model_only_img()
elif args.model == "only_text":
    model = Model_only_text()
else:
    raise ValueError("Invalid model choice")

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
train_part(model, optimizer, epochs=args.epochs)

test_guids = test_data['guid'].tolist()
test_predictions = predict_tags(loader_test, model)
save_predictions_to_file(test_guids, test_predictions)