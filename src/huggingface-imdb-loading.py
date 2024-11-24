from datasets import load_dataset
import pandas as pd


def load_imdb_dataset(max_samples=10000):
    # Load the IMDB dataset
    dataset = load_dataset("imdb")

    # Print the dataset statistics
    print(dataset)

    # Save the dataset to a CSV file
    for split, dataset in dataset.items():
        df = pd.DataFrame(dataset)
        df.to_csv(f'{split}.csv', index=False)
    
    # Combine train and test sets
    all_data = dataset['train'].shuffle(seed=42)
    if max_samples:
        all_data = all_data.select(range(min(max_samples, len(all_data))))
    
    # Split into train and validation sets
    train_val_split = all_data.train_test_split(test_size=0.2, seed=42)
    
    train_texts = train_val_split['train']['text']
    train_labels = train_val_split['train']['label']
    val_texts = train_val_split['test']['text']
    val_labels = train_val_split['test']['label']
    
    return train_texts, val_texts, train_labels, val_labels





# Usage
train_texts, val_texts, train_labels, val_labels = load_imdb_dataset()

print(f"Training set size: {len(train_texts)}")
print(f"Validation set size: {len(val_texts)}")
