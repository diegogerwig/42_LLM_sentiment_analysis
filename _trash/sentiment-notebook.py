# 1. Setup and Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

# 2. System Information
print("=== System Info ===")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print("\n")

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 3. Load and Analyze Dataset
df_sample = pd.read_parquet('../data/sample_reviews.parquet')
print("\n=== Dataset Overview ===")
print(f"Total samples: {len(df_sample)}")
print("\nLabel distribution:")
print(df_sample['label'].value_counts(normalize=True))

# Calculate text statistics
text_lengths = df_sample['sentence'].str.len()
word_lengths = df_sample['sentence'].str.split().str.len()

print("\n=== Text Statistics ===")
print("Character lengths:")
print(f"Mean: {text_lengths.mean():.1f}")
print(f"Median: {text_lengths.median():.1f}")
print(f"Max: {text_lengths.max()}")
print("\nWord lengths:")
print(f"Mean: {word_lengths.mean():.1f}")
print(f"Median: {word_lengths.median():.1f}")
print(f"Max: {word_lengths.max()}")

# Visualize length distributions
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.hist(text_lengths, bins=50)
plt.title('Character Length Distribution')
plt.xlabel('Number of Characters')
plt.ylabel('Count')

plt.subplot(1, 2, 2)
plt.hist(word_lengths, bins=50)
plt.title('Word Length Distribution')
plt.xlabel('Number of Words')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# 4. Data Preparation
# Set parameters
MAX_SAMPLES = 1000  # Reduce this if you have memory constraints
TEST_SIZE = 0.2
VAL_SIZE = 0.1
BATCH_SIZE = 16
NUM_EPOCHS = 3
LEARNING_RATE = 2e-5

# Sample data if needed
if MAX_SAMPLES:
    data = df_sample.sample(n=MAX_SAMPLES, random_state=42)
else:
    data = df_sample

texts = data['sentence'].values
labels = data['label'].values

# Split data
train_texts, test_texts, train_labels, test_labels = train_test_split(
    texts, labels, test_size=TEST_SIZE, random_state=42, stratify=labels
)

train_texts, val_texts, train_labels, val_labels = train_test_split(
    train_texts, train_labels, 
    test_size=VAL_SIZE/(1-TEST_SIZE), 
    random_state=42,
    stratify=train_labels
)

print("\n=== Data Split Information ===")
print(f"Training samples: {len(train_texts)}")
print(f"Validation samples: {len(val_texts)}")
print(f"Test samples: {len(test_texts)}")

# 5. Model and Tokenizer Initialization
print("\n=== Model Initialization ===")
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification.from_pretrained(
    'distilbert-base-uncased', 
    num_labels=2
).to(device)

# Show tokenization example
print("\n=== Tokenization Example ===")
sample_text = train_texts[0]
print(f"Original text: {sample_text}")
tokenized = tokenizer(sample_text, truncation=True, padding=True, return_tensors='pt')
print(f"Tokenized length: {len(tokenized['input_ids'][0])}")
print(f"First 10 tokens: {tokenizer.convert_ids_to_tokens(tokenized['input_ids'][0])[:10]}")





# 6. Dataset Creation
print("\n=== Creating Datasets ===")
# Tokenize all texts
train_texts_encoded = tokenizer(train_texts.tolist(), truncation=True, padding=True)
val_texts_encoded = tokenizer(val_texts.tolist(), truncation=True, padding=True)
test_texts_encoded = tokenizer(test_texts.tolist(), truncation=True, padding=True)

# Create TensorDatasets
train_dataset = TensorDataset(
    torch.tensor(train_texts_encoded['input_ids']),
    torch.tensor(train_texts_encoded['attention_mask']),
    torch.tensor(train_labels)
)

val_dataset = TensorDataset(
    torch.tensor(val_texts_encoded['input_ids']),
    torch.tensor(val_texts_encoded['attention_mask']),
    torch.tensor(val_labels)
)

test_dataset = TensorDataset(
    torch.tensor(test_texts_encoded['input_ids']),
    torch.tensor(test_texts_encoded['attention_mask']),
    torch.tensor(test_labels)
)

print("Datasets created successfully!")
print(f"Train dataset size: {len(train_dataset)}")
print(f"Validation dataset size: {len(val_dataset)}")
print(f"Test dataset size: {len(test_dataset)}")

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# Show batch structure
sample_batch = next(iter(train_loader))
print("\n=== Batch Structure ===")
print(f"Input IDs shape: {sample_batch[0].shape}")
print(f"Attention mask shape: {sample_batch[1].shape}")
print(f"Labels shape: {sample_batch[2].shape}")






# 7. Training Functions
def evaluate_model(model, dataloader, device):
    """Evaluate the model and return loss and accuracy."""
    model.eval()
    total_loss = 0
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids, attention_mask, labels = [b.to(device) for b in batch]
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            
            total_loss += outputs.loss.item()
            preds = torch.argmax(outputs.logits, dim=-1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(true_labels, predictions) * 100
    avg_loss = total_loss / len(dataloader)
    
    return avg_loss, accuracy, predictions, true_labels

# 8. Training Loop
print("\n=== Starting Training ===")
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
best_val_acc = 0

for epoch in range(NUM_EPOCHS):
    # Training
    model.train()
    total_loss = 0
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{NUM_EPOCHS}')
    
    for batch in progress_bar:
        input_ids, attention_mask, labels = [b.to(device) for b in batch]
        
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    # Validation
    val_loss, val_acc, val_preds, val_true = evaluate_model(model, val_loader, device)
    
    # Save metrics
    avg_train_loss = total_loss / len(train_loader)
    history['train_loss'].append(avg_train_loss)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
    
    print(f"\nEpoch {epoch+1} Summary:")
    print(f"Average training loss: {avg_train_loss:.4f}")
    print(f"Validation loss: {val_loss:.4f}")
    print(f"Validation accuracy: {val_acc:.2f}%")
    print("\nValidation Classification Report:")
    print(classification_report(val_true, val_preds))
    
    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), 'best_model.pt')
        print(f"New best model saved with accuracy: {val_acc:.2f}%")

# 9. Final Evaluation
print("\n=== Final Evaluation ===")
test_loss, test_acc, test_preds, test_true = evaluate_model(model, test_loader, device)
print("\nTest Results:")
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.2f}%")
print("\nTest Classification Report:")
print(classification_report(test_true, test_preds))

# 10. Plot Training History
plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
plt.plot(history['train_loss'], label='Training Loss', marker='o')
plt.plot(history['val_loss'], label='Validation Loss', marker='o')
plt.title('Loss vs. Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history['val_acc'], label='Validation Accuracy', marker='o')
plt.title('Validation Accuracy vs. Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 11. Save Model
model_save_path = './models/sentiment_model'
model.save_pretrained(model_save_path)
tokenizer.save_pretrained(model_save_path)
print(f"\nModel and tokenizer saved in {model_save_path}")

# Print final summary
print("\n=== Training Complete ===")
print(f"Best validation accuracy: {best_val_acc:.2f}%")
print(f"Final test accuracy: {test_acc:.2f}%")