
# Chest X-ray COVID Classification

Binary classification of chest X-ray images (COVID-19 vs Normal) using
transfer learning with DenseNet121.

## Project Structure
- `notebooks/`: exploratory analysis and experiments
- `src/`: reusable training and model code
- `app.py`: Streamlit UI for uploading a chest X-ray and getting a prediction

## Streamlit App
A simple web UI lets you upload a chest X-ray image and see the model's
prediction (COVID vs Normal).

1. Train the model by running `notebooks/analysis.ipynb` through the
   fine-tuning step. This saves trained weights to `model_weights.h5` in
   the project root.
2. Install dependencies: `pip install -r requirements.txt`
3. Run the app from this folder:
   ```
   streamlit run app.py
   ```
4. Upload an X-ray image in the browser tab that opens to see the prediction.

If `model_weights.h5` isn't present yet, the app still runs but shows a
warning instead of a prediction until you train and save the model.

## Status
Work in progress.
