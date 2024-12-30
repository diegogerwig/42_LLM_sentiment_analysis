import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    logging
)
from datasets import load_dataset
import warnings
from tqdm import tqdm
import time

# Suppress warnings
logging.set_verbosity_error()
warnings.filterwarnings('ignore')

class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.encodings = tokenizer(texts, truncation=True, padding=True,
                                 max_length=max_length, return_tensors='pt')
        self.labels = torch.tensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

class SentimentAnalyzer:
    def __init__(self, model_name='distilbert-base-uncased'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        self.model = DistilBertForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=2
        )
        self.model.to(self.device)
        
        # Training parameters
        self.batch_size = 8
        self.learning_rate = 0.0005
        self.num_epochs = 3
        
        print(f"Using device: {self.device}")

    def prepare_data(self, sample_size=1000):
        """Load and prepare the datasets"""
        # Load IMDB dataset
        imdb = load_dataset("stanfordnlp/imdb")
        df = pd.DataFrame(imdb['train'])
        
        # Reduce dataset size if needed
        if sample_size:
            df = df.sample(n=min(sample_size, len(df)), random_state=42)
        
        # Split data
        train_texts, temp_texts, train_labels, temp_labels = train_test_split(
            df['text'].values,
            df['label'].values,
            test_size=0.3,
            random_state=42
        )
        
        val_texts, test_texts, val_labels, test_labels = train_test_split(
            temp_texts,
            temp_labels,
            test_size=0.5,
            random_state=42
        )

        # Create datasets
        train_dataset = SentimentDataset(train_texts.tolist(), train_labels, self.tokenizer)
        val_dataset = SentimentDataset(val_texts.tolist(), val_labels, self.tokenizer)
        test_dataset = SentimentDataset(test_texts.tolist(), test_labels, self.tokenizer)

        # Create dataloaders
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=self.batch_size)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size)

        print("\nData split information:")
        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
        print(f"Test samples: {len(test_dataset)}")

    def train(self):
        """Train the model"""
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        
        best_val_accuracy = 0
        
        for epoch in range(self.num_epochs):
            # Training
            self.model.train()
            train_loss = 0
            train_steps = 0
            
            progress_bar = tqdm(self.train_loader, desc=f'Epoch {epoch + 1}/{self.num_epochs}')
            for batch in progress_bar:
                optimizer.zero_grad()
                
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                train_steps += 1
                progress_bar.set_postfix({'training_loss': train_loss/train_steps})

            # Validation
            val_accuracy = self.evaluate(self.val_loader)
            print(f"Epoch {epoch + 1}/{self.num_epochs}")
            print(f"Training Loss: {train_loss/train_steps:.4f}")
            print(f"Validation Accuracy: {val_accuracy:.4f}\n")
            
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                # You could save the model here if needed
                # torch.save(self.model.state_dict(), 'best_model.pt')

    def evaluate(self, dataloader):
        """Evaluate the model on the given dataloader"""
        self.model.eval()
        total_preds = []
        total_labels = []
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                preds = torch.argmax(outputs.logits, dim=1)
                total_preds.extend(preds.cpu().numpy())
                total_labels.extend(labels.cpu().numpy())
        
        accuracy = accuracy_score(total_labels, total_preds)
        return accuracy

    def test(self):
        """Test the model and print metrics"""
        test_accuracy = self.evaluate(self.test_loader)
        print(f"\nTest Accuracy: {test_accuracy:.4f}")
        return test_accuracy

def main():
    # Initialize analyzer
    analyzer = SentimentAnalyzer()
    
    # Prepare data (using a smaller dataset for demonstration)
    analyzer.prepare_data(sample_size=1000)
    
    # Train model
    start_time = time.time()
    analyzer.train()
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.2f} seconds")
    
    # Test model
    test_accuracy = analyzer.test()
    
    return test_accuracy

if __name__ == "__main__":
    main()