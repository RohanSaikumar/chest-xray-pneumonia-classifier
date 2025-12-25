<h3><b>Chest X-Ray Pneumonia Classification</b></h3>

<h5>This project implements a binary chest X-ray classifier (Pneumonia vs Normal) using transfer learning with a pretrained CNN.
The emphasis is on training stability, reproducibility, and principled design choices, not just metric maximization.</h5>
 
<h4><b>Approach</b><h4>
<ul>
<li>Model: DenseNet-based CNN pretrained on ImageNet</li>
<li>Framework:Tensorflow / Keras</li>
<li>Task : Binary Image Classification</li>
<li>Strategy: Frozen Training -> Fine-Tuning</li>
</ul>

<h4><b>Project Structure</b></h4>

- `src/` – data loading, model definition, training, and evaluation code  
- `notebooks/` – exploratory data analysis (EDA)  
- `requirements.txt` – project dependencies  
- `.gitignore` – ignored files and directories

<h4><b>Key Design Decisions</b></h4>
<h5>No Data Resampling</h5>
<h6>Class imbalance was not addressed via over/undersampling to preserve real sample diversity and avoid overfitting from duplicated images.
Instead, robustness is improved through augmentation and regularization.</h6>

<h5><b>Learning Rate Scheduling</b></h5>
<h6>ReduceLROnPlateau is used to automatically lower the learning rate when validation loss saturates, leading to more stable convergence during fine-tuning.</h6>

<h5><b>Label Smoothing</b></h5>
<h6>Label smoothing is applied to reduce overconfident predictions and improve generalization, which is particularly important in medical imaging tasks with potential label noise.</h6>

<h4><b>Training Strategies</b></h4>
<ol>
<li>Frozen training of the classification head<li>
<li>Fine-Tuning of the upper convolutional layers with a reduced learning rate<li>
</ol>
<h5>This balances stability and domain adaptation.</h5>

<h4><b>Results</b></h4>
<ul>
<li><b>Test Accuracy: 82.34%<b><li>
</ul>

<h4><b>Author</b></h4>

<b>Rohan Saikumar</b>


