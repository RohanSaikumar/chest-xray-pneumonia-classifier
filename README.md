
# Chest X-Ray Pneumonia Classification

Binary chest X-ray classifier (Pneumonia caused by COVID-19 vs Normal) using
transfer learning with a pretrained CNN. The emphasis is on training
stability, reproducibility, and principled design choices, not just metric
maximization.

## Approach
- Model: DenseNet121 CNN pretrained on ImageNet
- Framework: TensorFlow / Keras
- Task: Binary image classification
- Strategy: Frozen training -> Fine-tuning

## Project Structure
- `src/` - data loading, model definition, training, and evaluation code
- `notebooks/` - exploratory data analysis (EDA)
- `app.py` - Streamlit UI for uploading an X-ray and getting a prediction
- `requirements.txt` - project dependencies
- `.gitignore` - ignored files and directories

## Key Design Decisions

### No Data Resampling
Class imbalance was not addressed via over/undersampling, to preserve real
sample diversity and avoid overfitting from duplicated images. Robustness is
instead improved through augmentation and regularization.

### Learning Rate Scheduling
`ReduceLROnPlateau` automatically lowers the learning rate when validation
loss saturates, leading to more stable convergence during fine-tuning.

### Label Smoothing
Label smoothing is applied to reduce overconfident predictions and improve
generalization, which is particularly important in medical imaging tasks
with potential label noise.

## Training Strategy
1. Frozen training of the classification head
2. Fine-tuning of the upper convolutional layers with a reduced learning rate

This balances stability and domain adaptation.

## Results
- **Test Accuracy: 82.34%**

## Running the Project
```
pip install -r requirements.txt
```
Run `notebooks/analysis.ipynb` top to bottom to download the dataset, train,
and evaluate the model. Training saves weights to `model_weights.h5` in the
project root.

### Streamlit App
After training, run a simple UI to upload an X-ray and get a prediction:
```
streamlit run app.py
```

## Author
Rohan Saikumar
