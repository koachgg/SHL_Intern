# Speech-Based Grammar Score Prediction

## Overview

This repository contains code for predicting grammar scores from speech audio files. The model uses the WavLM-Base-Plus pre-trained model to extract features from speech audio and employs a regression head to predict grammar scores on a scale of 1-5.

## Problem Statement

The goal of this project is to develop a model that can automatically assess the grammatical correctness of spoken English. Given an audio recording of someone speaking English, the model predicts a grammar score between 1 (poor grammar) and 5 (excellent grammar).

## Dataset

The dataset consists of audio recordings (WAV files) along with corresponding grammar scores. The data is split into:
- Training set: Audio files with known grammar scores
- Test set: Audio files for which we need to predict grammar scores

>> For Dataset : Available on [Kaggle](https://www.kaggle.com/competitions/shl-intern-hiring-assessment/overview)

## Methodology

I implemented two main approaches to solve this problem (well i tried a lot of them but with the time constraints found these two the best approaches if time permits i will update this repo): 
> **Note**: Only the **ensemble approach (Approach 1)** has been included in this repository.  
> If you're interested in trying out **Approach 2 (Best Fold Model)**, feel free to **reach out to me**, and I’ll be happy to share the code.

>> Colab Link : 🔗 https://colab.research.google.com/drive/1VReZ3F3jB7H1NP3i6lLhDfmkarJoD8Wx?usp=sharing

### Approach 1: Ensemble Mean (More Robust)

This approach uses K-fold cross-validation to train multiple models and averages their predictions:

1. Split training data into K=3 folds
2. Train a model on each fold, using the remaining data as validation
3. For inference:
   - Generate predictions from all 3 models
   - Average the predictions with equal weights
   - Apply score calibration to match the training distribution

**Advantages:**
- More robust to outliers and model variance
- Generally better generalization to unseen data
- More reliable performance across different data distributions

### Approach 2: Best Fold Model (Better Leaderboard Performance)

This approach selects only the highest-performing fold model:

1. Perform K-fold cross-validation as in Approach 1
2. Select only the best performing model (Fold 3 in my case, with validation score of 0.7364)
3. For inference:
   - Generate predictions using only this single model
   - Apply more aggressive segmentation and test-time augmentation
   - Calibrate scores to match training distribution

**Advantages:**
- Higher peak performance on public test data
- Better leaderboard ranking
- Computationally more efficient at inference time

## Implementation Details

### Model Architecture

```python
class GrammarScoreModel(torch.nn.Module):
    def __init__(self, base_model_name="microsoft/wavlm-base-plus"):
        super(GrammarScoreModel, self).__init__()
        
        # Load base model
        self.base_model = AutoModel.from_pretrained(base_model_name)
        hidden_size = self.base_model.config.hidden_size
        
        # Regression head
        self.regression_head = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, 256),
            torch.nn.GELU(),
            torch.nn.Dropout(0.4),  # Increased dropout
            
            torch.nn.Linear(256, 64),
            torch.nn.GELU(),
            torch.nn.Dropout(0.3),  # Increased dropout
            
            torch.nn.Linear(64, 1)
        )
    
    def forward(self, input_values, attention_mask=None):
        # Process through base model
        outputs = self.base_model(
            input_values=input_values,
            attention_mask=attention_mask
        )
        
        # Get hidden states and apply pooling
        hidden_states = outputs.last_hidden_state
        
        # Handle dimension mismatch between hidden states and attention mask
        if attention_mask is not None:
            # Make sure dimensions match
            seq_len = hidden_states.shape[1]
            mask_len = attention_mask.shape[1]
            
            if mask_len != seq_len:
                # Adjust attention mask as needed
                if mask_len > seq_len:
                    attention_mask = attention_mask[:, :seq_len]
                else:
                    padding = torch.zeros(
                        attention_mask.shape[0], 
                        seq_len - mask_len, 
                        device=attention_mask.device, 
                        dtype=attention_mask.dtype
                    )
                    attention_mask = torch.cat([attention_mask, padding], dim=1)
            
            # Apply mean pooling with mask
            expanded_mask = attention_mask.unsqueeze(-1).float()
            hidden_states = hidden_states * expanded_mask
            pooled = hidden_states.sum(dim=1) / expanded_mask.sum(dim=1).clamp(min=1e-9)
        else:
            # Simple mean pooling
            pooled = hidden_states.mean(dim=1)
        
        # Get score through regression head
        score = self.regression_head(pooled)
        
        # Scale to range [1, 5]
        score = torch.sigmoid(score) * 4.0 + 1.0
        
        return score
```

### Key Techniques

1. **Multi-segment prediction**: Split long audio into multiple segments and average predictions
2. **Test-time augmentation**: Generate multiple predictions with different audio processing
3. **Score calibration**: Adjust prediction distribution to match training data distribution
4. **Cross-validation**: Train multiple models on different data splits
5. **Dropout regularization**: Prevent overfitting with strategic dropout in the regression head

## Results
> 📌 **Important:** The observed difference between the local validation score and the public leaderboard score may be due to the leaderboard being evaluated on **only 30% of the test data**.  
> The **final results will be based on the remaining 70%**, so final rankings may fluctuate.


| Approach | Local Validation Score | Public Leaderboard Score |
|----------|------------------------|--------------------------|
| Ensemble Mean | 0.7208 | 0.691 |
| Best Fold (Fold 3) | 0.7364 | 0.698 |

While the ensemble mean approach is theoretically more robust, the best fold approach performed better on the public leaderboard. This suggests that for this specific competition and test data distribution, the individual fold model (Fold 3) generalized better than the ensemble.

## Future Work

Possible improvements that could further enhance the model performance:

1. Enhanced test-time augmentation with speed perturbation and pitch shifting
2. Attention-based pooling instead of simple mean pooling
3. Quantile calibration for better score distribution matching
4. Alternative model architectures (XLS-R, Whisper embeddings)
5. Voice activity detection to focus on speech segments

## Author

Implemented by [Belo Abhigyan](https://github.com/koachgg)

## License

This project is licensed under the MIT License - see the LICENSE file for details.
```
