#!/usr/bin/env python3
"""
FactGuard AI Model Training Script

This script provides functionality to fine-tune AI models for:
1. Claim extraction from text
2. Stance detection between claims and evidence
3. Misinformation classification

Usage:
    python train.py --task stance --data_path ./data/stance_data.json
    python train.py --task claim_extraction --data_path ./data/claims_data.json
    python train.py --task misinformation --data_path ./data/misinfo_data.json
"""

import argparse
import json
import os
import logging
from datetime import datetime
from typing import List, Dict, Any, Tuple
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, EarlyStoppingCallback
)
from datasets import Dataset as HFDataset
import wandb

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FactGuardDataset(Dataset):
    """Custom dataset class for FactGuard training data"""
    
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class FactGuardTrainer:
    """Main trainer class for FactGuard models"""
    
    def __init__(self, task: str, model_name: str = "microsoft/DialoGPT-medium"):
        self.task = task
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.label_map = {}
        self.reverse_label_map = {}
        
        # Task-specific configurations
        self.task_configs = {
            'stance': {
                'labels': ['supports', 'contradicts', 'neutral'],
                'model_name': 'facebook/bart-large-mnli'
            },
            'claim_extraction': {
                'labels': ['claim', 'not_claim'],
                'model_name': 'bert-base-uncased'
            },
            'misinformation': {
                'labels': ['Real', 'Fake', 'NaN' ],
                'model_name': 'roberta-base'
            }
        }
        
        if task not in self.task_configs:
            raise ValueError(f"Unsupported task: {task}")
        
        self.config = self.task_configs[task]
        self.setup_label_mapping()
    
    def setup_label_mapping(self):
        """Setup label mapping for the task"""
        labels = self.config['labels']
        self.label_map = {label: idx for idx, label in enumerate(labels)}
        self.reverse_label_map = {idx: label for label, idx in self.label_map.items()}
        logger.info(f"Label mapping for {self.task}: {self.label_map}")
    
    def load_data(self, data_path: str) -> Tuple[List[str], List[int]]:
        """Load and preprocess training data"""
        logger.info(f"Loading data from {data_path}")
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        texts = []
        labels = []
        
        if self.task == 'stance':
            # Expected format: [{"claim": "...", "evidence": "...", "label": "supports"}]
            for item in data:
                text = f"Claim: {item['claim']} Evidence: {item['evidence']}"
                texts.append(text)
                labels.append(self.label_map[item['label']])
        
        elif self.task == 'claim_extraction':
            # Expected format: [{"text": "...", "label": "claim"}]
            for item in data:
                texts.append(item['text'])
                labels.append(self.label_map[item['label']])
        
        elif self.task == 'misinformation':
            # Expected format: [{"text": "...", "label": "reliable"}]
            for item in data:
                texts.append(item['text'])
                labels.append(self.label_map[item['label']])
        
        logger.info(f"Loaded {len(texts)} samples")
        return texts, labels
    
    def prepare_model(self):
        """Initialize tokenizer and model"""
        model_name = self.config['model_name']
        logger.info(f"Loading model: {model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=len(self.config['labels'])
        )
        
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def create_datasets(self, texts: List[str], labels: List[int], test_size: float = 0.2):
        """Create train and validation datasets"""
        stratify = labels if len(set(labels)) > 1 and min([labels.count(l) for l in set(labels)]) > 1 else None

        train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=test_size, random_state=42, stratify=stratify)

        
        train_dataset = FactGuardDataset(train_texts, train_labels, self.tokenizer)
        val_dataset = FactGuardDataset(val_texts, val_labels, self.tokenizer)
        
        logger.info(f"Train samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
        return train_dataset, val_dataset
    
    def train(self, data_path: str, output_dir: str, **kwargs):
        """Train the model"""
        # Load data
        texts, labels = self.load_data(data_path)
        
        # Prepare model
        self.prepare_model()
        
        # Create datasets
        train_dataset, val_dataset = self.create_datasets(texts, labels)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=kwargs.get('epochs', 3),
            per_device_train_batch_size=kwargs.get('batch_size', 8),
            per_device_eval_batch_size=kwargs.get('batch_size', 8),
            warmup_steps=kwargs.get('warmup_steps', 500),
            weight_decay=kwargs.get('weight_decay', 0.01),
            logging_dir=f'{output_dir}/logs',
            logging_steps=100,
            #evaluation_strategy="steps",
            #eval_steps=500,
            #save_strategy="steps",
            save_steps=500,
            #load_best_model_at_end=True,
            #metric_for_best_model="eval_loss",
            #greater_is_better=False,
            do_eval=True,
            report_to="wandb" if kwargs.get('use_wandb', False) else None,
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            #callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
        
        # Train model
        logger.info("Starting training...")
        trainer.train()
        
        # Save model
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
        
        # Save label mapping
        with open(f"{output_dir}/label_map.json", 'w') as f:
            json.dump(self.label_map, f)
        
        logger.info(f"Model saved to {output_dir}")
        
        # Evaluate model
        eval_results = trainer.evaluate()
        logger.info(f"Evaluation results: {eval_results}")
        
        return trainer

def create_sample_data():
    """Create sample training data for demonstration"""
    
    # Sample stance detection data
    stance_data = [
        {
            "claim": "Vaccines are safe and effective",
            "evidence": "Multiple clinical trials show vaccines reduce disease risk by 95%",
            "label": "supports"
        },
        {
            "claim": "Climate change is caused by human activities",
            "evidence": "Solar activity has remained constant while temperatures rise",
            "label": "supports"
        },
        {
            "claim": "The Earth is flat",
            "evidence": "Satellite images show Earth's curvature from space",
            "label": "contradicts"
        },
        {
            "claim": "Coffee prevents cancer",
            "evidence": "Some studies show correlation, others show no effect",
            "label": "neutral"
        }
    ]
    
    # Sample claim extraction data
    claim_data = [
        {
            "text": "Scientists have proven that drinking water is healthy",
            "label": "claim"
        },
        {
            "text": "I think the weather is nice today",
            "label": "not_claim"
        },
        {
            "text": "The study shows a 50% reduction in symptoms",
            "label": "claim"
        },
        {
            "text": "Maybe we should consider other options",
            "label": "not_claim"
        }
    ]
    
    # Sample misinformation data
    misinfo_data = [
        {
            "text": "According to peer-reviewed research published in Nature, the treatment shows promising results",
            "label": "reliable"
        },
        {
            "text": "SHOCKING: Doctors don't want you to know this one weird trick!",
            "label": "unreliable"
        },
        {
            "text": "The preliminary study suggests potential benefits, but more research is needed",
            "label": "mixed"
        }
    ]
    
    # Create data directory
    os.makedirs("data", exist_ok=True)
    
    # Save sample data
    with open("data/stance_data.json", 'w') as f:
        json.dump(stance_data * 25, f, indent=2)  # Multiply for more samples
    
    with open("data/claims_data.json", 'w') as f:
        json.dump(claim_data * 25, f, indent=2)
    
    with open("data/misinfo_data.json", 'w') as f:
        json.dump(misinfo_data * 25, f, indent=2)
    
    logger.info("Sample data created in ./data/ directory")

def main():
    parser = argparse.ArgumentParser(description="Train FactGuard AI models")
    parser.add_argument("--task", required=True, choices=['stance', 'claim_extraction', 'misinformation'],
                       help="Training task")
    parser.add_argument("--data_path", required=True, help="Path to training data JSON file")
    parser.add_argument("--output_dir", default="./models", help="Output directory for trained model")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases for logging")
    parser.add_argument("--create_sample_data", action="store_true", help="Create sample training data")
    
    args = parser.parse_args()
    
    if args.create_sample_data:
        create_sample_data()
        return
    
    # Initialize wandb if requested
    if args.use_wandb:
        wandb.init(project="factguard", name=f"{args.task}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    
    # Create output directory
    output_dir = f"{args.output_dir}/{args.task}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize trainer
    trainer = FactGuardTrainer(args.task)
    
    # Train model
    trainer.train(
        data_path=args.data_path,
        output_dir=output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        use_wandb=args.use_wandb
    )
    
    logger.info("Training completed successfully!")

if __name__ == "__main__":
    main()
