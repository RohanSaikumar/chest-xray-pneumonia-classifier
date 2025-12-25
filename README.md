<h4><b>Chest X-Ray Pneumonia Classification<b><h4>

This project implements a binary chest X-ray classifier (Pneumonia vs Normal) using transfer learning with a pretrained CNN.
The emphasis is on training stability, reproducibility, and principled design choices, not just metric maximization.
 
<h5><b>Approach<b><h5>
<li>Model: DenseNet-based CNN pretrained on ImageNet<li>
<li>Framework:Tensorflow / Keras<li>
<li>Task : Binary Image Classification<li>
<li>Strategy: Frozen Training -> Fine-Tuning<li>

<h5><b>Project Structure<b><h5>

- `src/` – data loading, model definition, training, and evaluation code  
- `notebooks/` – exploratory data analysis (EDA)  
- `requirements.txt` – project dependencies  
- `.gitignore` – ignored files and directories

<h5><b>Key Design Decisions<b><h5>
<h6>No Data Resampling<h6>
Class imbalance was not addressed via over/undersampling to preserve real sample diversity and avoid overfitting from duplicated images.
Instead, robustness is improved through augmentation and regularization.

<h6>Learning Rate Scheduling<h6>
ReduceLROnPlateau is used to automatically lower the learning rate when validation loss saturates, leading to more stable convergence during fine-tuning.

<h6>Label Smoothing<h6>
Label smoothing is applied to reduce overconfident predictions and improve generalization, which is particularly important in medical imaging tasks with potential label noise.

<h5><b>Training Strategies<b><h5>
<ol>
<li>Frozen training of the classification head<li>
<li>Fine-Tuning of the upper convolutional layers with a reduced learning rate<li>
</ol>
This balances stability and domain adaptation.

<h5><b>Results<b><h5>
<ul>
<li><b>Test Accuracy: 82.34%<b><li>

<h5><b>Author<b><h5>
<b>Rohan Saikumar<b>