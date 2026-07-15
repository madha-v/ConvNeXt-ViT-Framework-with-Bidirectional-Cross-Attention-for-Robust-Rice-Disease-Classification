🌾 ConvNeXt–ViT Framework with Bidirectional Cross-Attention for Robust Rice Disease Classification

A hybrid deep learning framework combining ConvNeXt and Vision Transformer (ViT) representations through Bidirectional Cross-Attention for robust multi-class rice disease classification.

The framework is evaluated on the Paddy Doctor dataset and achieves 97.79% classification accuracy with a 97.65% Macro F1-Score across 13 paddy disease and healthy classes.

📌 Overview

Rice diseases significantly affect crop health, productivity, and agricultural yield. Manual disease identification generally requires visual inspection and agricultural expertise, which may limit rapid diagnosis in large farming regions.

This project proposes a hybrid computer vision framework that combines:

ConvNeXt for hierarchical local feature extraction.
Vision Transformer (ViT) for global contextual representation.
Bidirectional Cross-Attention for interaction between CNN and transformer features.
Adaptive Feature Fusion for unified disease representation.
Exponential Moving Average (EMA) for stable model evaluation.
Mixed Precision Training (AMP) for efficient GPU training.

The objective is to combine the strong local inductive bias of convolutional architectures with the long-range dependency modelling capability of Vision Transformers.

🧠 Proposed Architecture
                         Input Paddy Image
                                │
                                ▼
                                
                      Image Preprocessing
                                │
                 ┌──────────────┴──────────────┐
                 │                             │
                 ▼                             ▼
        ConvNeXt Feature Branch       Vision Transformer Branch
                 │                             │
                 │                             │
        Local Spatial Features        Global Context Features
                 │                             │
                 └──────────────┬──────────────┘
                                │
                                ▼
                                
                 Bidirectional Cross-Attention
                                │
                  ┌─────────────┴─────────────┐
                  │                           │
          ConvNeXt → ViT Attention    ViT → ConvNeXt Attention
                  │                           │
                  └─────────────┬─────────────┘
                                │
                                ▼
                     Adaptive Feature Fusion
                                │
                                ▼
                         Classification Head
                                │
                                ▼
                      Rice Disease Prediction
Why a Hybrid Architecture?

CNN-based architectures are highly effective at extracting local visual patterns such as:

Leaf lesions
Disease spots
Texture variations
Colour abnormalities
Edge and structural information

Vision Transformers provide stronger modelling of:

Long-range visual dependencies
Global leaf structure
Spatial relationships
Distributed disease patterns

The proposed framework allows both feature representations to interact using Bidirectional Cross-Attention instead of relying only on simple concatenation.

🔄 Bidirectional Cross-Attention

The core component of the framework is the bidirectional interaction between ConvNeXt and ViT features.

ConvNeXt Features ───────► ViT Feature Attention
        ▲                           │
        │                           ▼
ViT Features ◄──────── ConvNeXt Feature Attention


The ConvNeXt representation can attend to global transformer features, while transformer representations can simultaneously interact with local convolutional features.

This two-way feature exchange aims to produce a richer joint representation for disease classification.

📊 Dataset

The framework uses the Paddy Doctor dataset.

Dataset Characteristics
Property	Description
Dataset	Paddy Doctor
Total Images	16,225
Number of Classes	13
Disease Classes	12
Healthy Class	1
Domain	Paddy/Rice Disease Classification
Image Type	RGB Paddy Images
Disease Categories

The dataset contains the following categories:

Bacterial Leaf Blight
Bacterial Leaf Streak
Bacterial Panicle Blight
Black Stem Borer
Blast
Brown Spot
Downy Mildew
Hispa
Leaf Roller
Tungro
White Stem Borer
Yellow Stem Borer
Normal/Healthy
📈 Experimental Results

The proposed ConvNeXt–ViT hybrid framework achieved the following final benchmark results:

Evaluation Metric	Score
Accuracy	97.79%
Macro Precision	97.43%
Macro Recall	97.90%
Macro F1-Score	97.65%
Final Benchmark Verification
📊 --- FINAL BENCHMARK VERIFICATION REPORT ---

Final Accuracy : 97.79%
Macro Precision: 0.9743
Macro Recall   : 0.9790
Macro F1-Score : 0.9765

The strong Macro Precision, Recall, and F1-Score indicate consistent classification performance across disease categories rather than performance being represented only by overall accuracy.

📉 Comparison with Existing Approaches

The original Paddy Doctor benchmark evaluated CNN and transfer-learning architectures including VGG16, MobileNet, Xception, and ResNet34. Its reported best-performing model, ResNet34, achieved an F1-score of 97.50%.

Model / Study	Dataset	Classes	Reported Result
VGG16	Paddy Doctor	13	Benchmark baseline
MobileNet	Paddy Doctor	13	Benchmark baseline
Xception	Paddy Doctor	13	Benchmark baseline
ResNet34	Paddy Doctor	13	97.50% F1
Proposed ConvNeXt–ViT Hybrid	Paddy Doctor	13	97.65% Macro F1
Proposed ConvNeXt–ViT Hybrid	Paddy Doctor	13	97.79% Accuracy

Note: Cross-study comparisons should be interpreted carefully because data splits, preprocessing, augmentation, training strategies, and evaluation protocols may differ. The table is intended to provide benchmark context rather than claim a universally controlled state-of-the-art comparison.

Improvement over the Original Paddy Doctor ResNet34 Benchmark
Original ResNet34 F1 Benchmark : 97.50%
Proposed Hybrid Macro F1       : 97.65%
Absolute Difference            : +0.15 percentage points

The proposed architecture achieves competitive benchmark performance while introducing explicit cross-model feature interaction between convolutional and transformer representations.

📊 Graphical Evaluation

The evaluation pipeline generates four graphical performance analyses.

1. Confusion Matrix

The confusion matrix provides class-level analysis of correct predictions and misclassification patterns.

graphical_results/confusion_matrix.png




2. Accuracy and Macro F1 Across Epochs

The training progression graph visualizes convergence across 40 training epochs.

graphical_results/accuracy_f1_epochs.png




The model demonstrates progressive improvement during training and converges to approximately 97.6% validation performance.

3. Final Metrics Comparison

A comparative visualization of Accuracy, Macro Precision, Macro Recall, and Macro F1-Score.

graphical_results/final_metrics.png




4. Class-Wise Precision, Recall and F1-Score

This visualization analyses classification performance independently for every paddy disease category.

graphical_results/classwise_metrics.png




Class-wise evaluation is particularly important for agricultural disease classification because strong overall accuracy may otherwise hide weak performance on individual disease categories.

⚙️ Training Configuration

The training pipeline uses several optimization strategies.

Component	Configuration
Optimizer	AdamW
Learning Rate Strategy	Layer-wise Learning Rate
LR Scheduler	Cosine Annealing
Loss Function	Cross Entropy Loss
Label Smoothing	Enabled
Gradient Clipping	Max Norm = 1.0
EMA	Enabled
Mixed Precision	AMP
Epochs	40
Model Selection Metric	Macro F1-Score
Layer-Wise Learning Rate

A reduced learning rate is applied to backbone parameters:

backbone_lr = base_lr * 0.1
classifier_lr = base_lr

This allows pretrained feature representations to be fine-tuned conservatively while newly introduced layers adapt at a higher learning rate.


🚀 Training Pipeline

Dataset Loading
       │
       
       ▼
Image Augmentation
       
       ▼ 
ConvNeXt +ViT Forward Pass
       │
       
       ▼
Bidirectional Cross-Attention
       │
       
       ▼
Adaptive Feature Fusion
       │
       
       ▼
Cross Entropy Loss
       │
       
       ▼
       
AMP Backpropagation
       │
       ▼
Gradient Clipping
       │
       ▼
AdamW Optimization
       │
       ▼
EMA Model Update
       │
       
       ▼
Validation
       │
       
       ▼
Macro F1 Evaluation
       │
       
       ▼

       Best Model Checkpoint
📁 Project Structure
completericeproject/
│

├── config/

│   └── config.yaml
│
├── data/
│   └── dataset.py
│
├── models/
│   └── hybrid_model.py
│
├── utils/
│   ├── ema.py
│   └── metrics.py
│
├── graphical_results/
│   ├── confusion_matrix.png
│   ├── accuracy_f1_epochs.png
│   ├── final_metrics.png
│   └── classwise_metrics.png
│
├── train_images/
│
├── train.py
├── evaluate.py
├── requirements.txt
├── paddy_doctor_metadata.csv
└── README.md


🛠️ Installation
Clone the repository:

git clone https://github.com/madha-v/ConvNeXt-ViT-Framework-with-Bidirectional-Cross-Attention-for-Robust-Rice-Disease-Classification.git

Move into the project directory:

cd ConvNeXt-ViT-Framework-with-Bidirectional-Cross-Attention-for-Robust-Rice-Disease-Classification

Install dependencies:

pip install -r requirements.txt
🏋️ Training

Start model training using:

python train.py

The training pipeline evaluates the model after every epoch and saves the checkpoint achieving the highest Macro F1-Score.

best_model.pth

Model checkpoints are excluded from the Git repository because trained weight files may exceed standard GitHub file-size limits.

📊 Evaluation

Run the evaluation pipeline:

python evaluate.py

Example benchmark output:

📊 --- FINAL BENCHMARK VERIFICATION REPORT ---

Final Accuracy : 97.79%
Macro Precision: 0.9743
Macro Recall   : 0.9790
Macro F1-Score : 0.9765

The evaluation script also generates the graphical performance results.

✨ Key Contributions
Hybrid ConvNeXt–Vision Transformer architecture for rice disease classification.
Bidirectional Cross-Attention for CNN–Transformer feature interaction.
Integration of local spatial and global contextual representations.
Adaptive feature fusion for joint disease representation.
Layer-wise learning-rate optimization.
Exponential Moving Average based model evaluation.
Mixed-precision GPU training.
Macro F1 based best-model selection.
Detailed class-wise and graphical performance analysis.
97.79% final classification accuracy.
97.65% Macro F1-Score across 13 classes.
🔬 Future Work

Future extensions of the framework may include:

Explainable AI using Grad-CAM.
Mobile and edge-device deployment.
Real-time disease classification.
Model compression and quantization.
Disease severity estimation.
Integration with agricultural decision-support systems.
Evaluation across additional rice disease datasets and geographic regions.
📚 Dataset Reference

Paddy Doctor: A Visual Image Dataset for Automated Paddy Disease Classification and Benchmarking

Petchiammal A., Briskline Kiruba S., D. Murugan, and Pandarasamy A.

The dataset contains 16,225 annotated paddy images across 13 classes and was introduced for automated paddy disease classification and benchmarking.

👨‍💻 Author

Madhav Joshi

B.Tech Computer Science and Engineering


GitHub: madha-v

⭐ Support

If you find this research framework useful, consider giving the repository a ⭐.

Contributions, research discussions, and improvements are welcome.
