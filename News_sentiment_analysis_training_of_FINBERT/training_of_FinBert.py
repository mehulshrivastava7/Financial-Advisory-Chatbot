# %%
import pandas as pd 
import math 
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
import requests
from langchain import OpenAI, LLMChain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os
import openai
from tqdm import tqdm


# %%
df = pd.read_csv('News_sentiment_Jan2017_to_Apr2021.csv/News_sentiment_Jan2017_to_Apr2021.csv')
print(len(df))
df.head()

# %%
# drop the date, url and unamed columns
df = df.drop(['Date', 'URL', 'Unnamed: 5'], axis=1)
df['sentiment'] = df['sentiment'].str.lower()
df.head()

# %%
# changing the confidence from -1 to 1 to 0 to 1 by abs
df['confidence'] = df['confidence'].apply(lambda x: abs(x))

# Update sentiments to 'Neutral' where confidence is less than 0.9
# df.loc[df['confidence'] < 0.95, 'sentiment'] = 'Neutral'

# Assuming your original dataframe is named df
# Filter top 1000 positive and 1000 negative sentiments based on confidence
df1 = df[df['sentiment'] == 'positive'].sort_values(by='confidence', ascending=False).head(800).reset_index(drop=True)
df2 = df[df['sentiment'] == 'negative'].sort_values(by='confidence', ascending=False).head(1000).reset_index(drop=True)

# Filter df2 with next 1000-2000 positive sentiments based on confidence
df3 = df[df['sentiment'] == 'positive'].sort_values(by='confidence', ascending=False).iloc[800:1600].reset_index(drop=True)
df4 = df[df['sentiment'] == 'negative'].sort_values(by='confidence', ascending=False).iloc[1000:2000].reset_index(drop=True)

# Combine the dataframes df1 and df2
df1 = pd.concat([df1, df2], axis=0)

# Combine the dataframes df3 and df4
df2 = pd.concat([df3, df4], axis=0)

# %%
print(df1.shape)    
print(df2.shape)
print(df1["sentiment"].value_counts())
print(df2["sentiment"].value_counts())

# %%
# changing the confidence to 1 in df1 and 0.8 in df2
df1['confidence'] = 1
df2['confidence'] = 0.9

# %%
# Read the text files and create DataFrames
fp_50 = 'FinancialPhraseBank-v1.0/FinancialPhraseBank-v1.0/Sentences_50Agree.txt'
fp_66 = 'FinancialPhraseBank-v1.0/FinancialPhraseBank-v1.0/Sentences_66Agree.txt'
fp_75 = 'FinancialPhraseBank-v1.0/FinancialPhraseBank-v1.0/Sentences_75Agree.txt'
fp_100 = 'FinancialPhraseBank-v1.0/FinancialPhraseBank-v1.0/Sentences_AllAgree.txt'
files = [fp_50, fp_66, fp_75, fp_100]

dataframes = {}

for file_path in files:
    data = []
    with open(file_path, 'r', encoding='latin-1') as file:
        for line in file:
            title, sentiment = line.rsplit('@', 1)
            data.append({'Title': title.strip(), 'sentiment': sentiment.strip()})
    if file_path == fp_50:
        df = pd.DataFrame(data)
        df['confidence'] = 0.85
        dataframes['df3'] = df
    elif file_path == fp_66:
        df = pd.DataFrame(data)
        df['confidence'] = 0.9
        dataframes['df4'] = df
    elif file_path == fp_75:
        df = pd.DataFrame(data)
        df['confidence'] = 0.95
        dataframes['df5'] = df
    else:
        df = pd.DataFrame(data)
        df['confidence'] = 1
        dataframes['df6'] = df

# Step 1: Remove duplicates across the DataFrames to retain only unique sentences in each
df6_unique = dataframes['df6']  # Highest confidence, keep as is

# Remove titles in df6 from df5
df5_unique = dataframes['df5'][~dataframes['df5']['Title'].isin(df6_unique['Title'])]

# Remove titles in df6 and df5 from df4
df4_unique = dataframes['df4'][~dataframes['df4']['Title'].isin(df6_unique['Title'])]
df4_unique = df4_unique[~df4_unique['Title'].isin(df5_unique['Title'])]

# Remove titles in df6, df5, and df4 from df3
df3_unique = dataframes['df3'][~dataframes['df3']['Title'].isin(df6_unique['Title'])]
df3_unique = df3_unique[~df3_unique['Title'].isin(df5_unique['Title'])]
df3_unique = df3_unique[~df3_unique['Title'].isin(df4_unique['Title'])]

# Display the number of unique rows in each DataFrame
print(f"Unique rows in df3 (50% confidence): {len(df3_unique)}")
print(f"Unique rows in df4 (66% confidence): {len(df4_unique)}")
print(f"Unique rows in df5 (75% confidence): {len(df5_unique)}")
print(f"Rows in df6 (100% confidence): {len(df6_unique)}")

print(len(df1))
print(len(df2))
print(len(df3_unique))
print(len(df4_unique))
print(len(df5_unique))
print(len(df6_unique))

# Now df3_unique, df4_unique, df5_unique, and df6_unique contain non-overlapping sentences.


# %%
# appending all the dataframes
df = pd.concat([df1, df2, df3_unique, df4_unique, df5_unique, df6_unique], ignore_index=True)
# Shuffle the DataFrame
df = df.sample(frac=1).reset_index(drop=True)
print(df.head())

# %%
# finding total positive and negative and neutral sentiments
print(df['sentiment'].value_counts())


# %%
# Define label mapping
label_mapping = {'negative': 0, 'neutral': 1, 'positive': 2}

# Apply mapping
df['label'] = df['sentiment'].map(label_mapping)

# Verify the mapping
print(df[['sentiment', 'label']].head())

# %%
# First split into training and temp (validation + test)
train_df, temp_df = train_test_split(df, test_size=0.3, random_state=4, stratify=df['label'])

# Then split temp into validation and test
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=4, stratify=temp_df['label'])

print(f"Training set size: {train_df.shape}")
print(f"Validation set size: {val_df.shape}")
print(f"Test set size: {test_df.shape}")



# %%
class FinancialDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


# %%
# Define the model name (use the specific FinBERT model you have)
model = 'ProsusAI/finbert'
tokenizer = BertTokenizer.from_pretrained(model)


# Create dataset instances
train_dataset = FinancialDataset(
    texts=train_df['Title'].to_numpy(),
    labels=train_df['label'].to_numpy(),
    tokenizer=tokenizer
)

val_dataset = FinancialDataset(
    texts=val_df['Title'].to_numpy(),
    labels=val_df['label'].to_numpy(),
    tokenizer=tokenizer
)

test_dataset = FinancialDataset(
    texts=test_df['Title'].to_numpy(),
    labels=test_df['label'].to_numpy(),
    tokenizer=tokenizer
)

# Define batch size
batch_size = 16

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

print("\nDataLoaders created successfully.")


# %%
# Load the pre-trained FinBERT model
model = BertForSequenceClassification.from_pretrained(model, num_labels=3)

# Move model to GPU if available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

print(f"Model loaded and moved to {device}.")


# %%
# Define optimizer
optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)

# Total number of training steps
epochs = 4
total_steps = len(train_loader) * epochs

# Define scheduler
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)

print("\nOptimizer and scheduler initialized.")


# %%
def train_epoch(model, data_loader, optimizer, scheduler, device):
    model.train()
    losses = []
    
    for batch in tqdm(data_loader, desc="Training"):
        optimizer.zero_grad()
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss
        losses.append(loss.item())
        
        loss.backward()
        optimizer.step()
        scheduler.step()
    
    return np.mean(losses)

def eval_model(model, data_loader, device):
    model.eval()
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1).flatten()
            
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    
    return predictions, true_labels


# %%
for epoch in range(epochs):
    print(f"\nEpoch {epoch + 1}/{epochs}")
    train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
    print(f"Training Loss: {train_loss:.4f}")
    
    val_preds, val_labels = eval_model(model, val_loader, device)
    val_accuracy = accuracy_score(val_labels, val_preds)
    print(f"Validation Accuracy: {val_accuracy:.4f}")
    
    print("Validation Classification Report:")
    print(classification_report(val_labels, val_preds, target_names=['Negative', 'Neutral', 'Positive']))


# %%
# Evaluate on test set
test_preds, test_labels = eval_model(model, test_loader, device)
test_accuracy = accuracy_score(test_labels, test_preds)
print(f"Test Accuracy: {test_accuracy}")

print(classification_report(test_labels, test_preds, target_names=['Negative', 'Neutral', 'Positive']))


# %%
# After training or loading the model
model.save_pretrained('./temp')  # Saves the model weights to a local directory
tokenizer.save_pretrained('./temp')  # Save the tokenizer as well




