import torch
import pandas as pd
from typing import List, Dict, Any, Optional
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    pipeline
)


class BARTRegressionPredictor:
    """
    A utility class for loading and using the trained BART regression model.
    """
    
    def __init__(self, model_path: str, device: str = "auto"):
        """
        Initialize the BART regression predictor.
        
        Args:
            model_path: Path to the trained model directory
            device: Device to run the model on ("auto", "cpu", "cuda", or specific device)
        """
        self.model_path = model_path
        self.device = self._get_device(device)
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        
    def _get_device(self, device: str) -> str:
        """Determine the appropriate device to use."""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return device
    
    def load_model(self) -> None:
        """
        Load the trained model and tokenizer from the specified path.
        """
        try:
            print(f"Loading model from {self.model_path}...")
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            print(f"✓ Tokenizer loaded successfully")
            
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_path,
                num_labels=1,
                problem_type="regression"
            )
            
            self.model.to(self.device)
            self.model.eval()
            print(f"✓ Model loaded successfully on {self.device}")
            
            self.pipeline = pipeline(
                "text-classification",
                model=self.model,
                tokenizer=self.tokenizer,
                device=self.device,
                return_all_scores=True
            )
            print(f"✓ Inference pipeline created")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {str(e)}")
    
    def format_input(self, review_text: str, category: str = "Unknown", title: str = "") -> str:
        """
        Format input text to match the training data format.
        
        Args:
            review_text: The review text
            category: Product category (default: "Unknown")
            title: Review title (default: empty)
        
        Returns:
            Formatted input string
        """
        if title:
            return f"[CAT={category}] [TITLE={title}] {review_text}"
        else:
            return f"[CAT={category}] {review_text}"
    
    def preprocess_text(self, text: str, max_length: int = 1024) -> Dict[str, torch.Tensor]:
        """
        Preprocess text for model input.
        
        Args:
            text: Input text to process
            max_length: Maximum sequence length
        
        Returns:
            Dictionary containing tokenized inputs
        """
        if self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt"
        )
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        return inputs
    
    def predict_single(self, review_text: str, category: str = "Unknown", title: str = "") -> float:
        """
        Predict helpfulness score for a single review.
        
        Args:
            review_text: The review text
            category: Product category
            title: Review title
        
        Returns:
            Predicted helpfulness score
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        formatted_text = self.format_input(review_text, category, title)
        inputs = self.preprocess_text(formatted_text)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            prediction = outputs.logits.squeeze().item()
        
        return prediction
    
    def predict_batch(self, 
                     reviews: List[str], 
                     categories: Optional[List[str]] = None, 
                     titles: Optional[List[str]] = None,
                     batch_size: int = 8) -> List[float]:
        """
        Predict helpfulness scores for multiple reviews.
        
        Args:
            reviews: List of review texts
            categories: List of categories (optional)
            titles: List of titles (optional)
            batch_size: Batch size for processing
        
        Returns:
            List of predicted helpfulness scores
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        if categories is None:
            categories = ["Unknown"] * len(reviews)
        if titles is None:
            titles = [""] * len(reviews)
        
        formatted_texts = [
            self.format_input(review, cat, title) 
            for review, cat, title in zip(reviews, categories, titles)
        ]
        
        predictions = []
        
        # Process in batches
        for i in range(0, len(formatted_texts), batch_size):
            batch_texts = formatted_texts[i:i + batch_size]
            
            # Tokenize batch
            inputs = self.tokenizer(
                batch_texts,
                truncation=True,
                padding=True,
                max_length=1024,
                return_tensors="pt"
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Predict
            with torch.no_grad():
                outputs = self.model(**inputs)
                batch_predictions = outputs.logits.squeeze().tolist()
                
                # Handle single item case
                if isinstance(batch_predictions, float):
                    batch_predictions = [batch_predictions]
                
                predictions.extend(batch_predictions)
        
        return predictions
    
    def predict_from_dataframe(self, 
                              df: pd.DataFrame, 
                              text_column: str = "text",
                              category_column: Optional[str] = "category",
                              title_column: Optional[str] = "title",
                              batch_size: int = 8) -> pd.DataFrame:
        reviews = df[text_column].tolist()
        categories = df[category_column].tolist() if category_column and category_column in df.columns else None
        titles = df[title_column].tolist() if title_column and title_column in df.columns else None
        
        predictions = self.predict_batch(reviews, categories, titles, batch_size)
        
        result_df = df.copy()
        result_df['predicted_helpfulness'] = predictions
        
        return result_df