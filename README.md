# A Hybrid Deep Learning Framework for 44-Class Brain Tumor Classification

Official implementation of the paper:

**A Hybrid Deep Learning Framework for 44-Class Brain Tumor Classification**

Published in the **28th International Conference on Computer and Information Technology (ICCIT 2025)**.

**Paper:** https://doi.org/10.1109/ICCIT68739.2025.11491434

**Best Result:** 98.69% accuracy using a hybrid ResNet50 + DenseNet121 architecture.

## Project Overview
This project implements and compares multiple deep learning architectures for Brain Tumor classification. The goal is to accurately classify different types of brain tumors using computer vision techniques, comparing traditional CNN architectures with modern transfer learning approaches and hybrid models.

## Model Architecture
<img width="2530" height="901" alt="times new roman drawio" src="https://github.com/user-attachments/assets/77be0501-f264-4f2d-b319-458680b2f437" />

## Dataset Information
- **Dataset Name:** Brain Tumor MRI Images 44 Classes
- **Source:** `mrnotalent/braint`, derived from the **fernando2rad** dataset on Kaggle and later augmented.
- **Classes:** 44 tumor classes:

1. Astrocitoma T1
2. Astrocitoma T1C+
3. Astrocitoma T2
4. Carcinoma T1
5. Carcinoma T1C+
6. Carcinoma T2
7. Ependimoma T1
8. Ependimoma T1C+
9. Ependimoma T2
10. Ganglioglioma T1
11. Ganglioglioma T1C+
12. Ganglioglioma T2
13. Germinoma T1
14. Germinoma T1C+
15. Germinoma T2
16. Glioblastoma T1
17. Glioblastoma T1C+
18. Glioblastoma T2
19. Granuloma T1
20. Granuloma T1C+
21. Granuloma T2
22. Meduloblastoma T1
23. Meduloblastoma T1C+
24. Meduloblastoma T2
25. Meningioma T1
26. Meningioma T1C+
27. Meningioma T2
28. Neurocitoma T1
29. Neurocitoma T1C+
30. Neurocitoma T2
31. Oligodendroglioma T1
32. Oligodendroglioma T1C+
33. Oligodendroglioma T2
34. Papiloma T1
35. Papiloma T1C+
36. Papiloma T2
37. Schwannoma T1
38. Schwannoma T1C+
39. Schwannoma T2
40. Tuberculoma T1
41. Tuberculoma T1C+
42. Tuberculoma T2
43. _NORMAL T1
44. _NORMAL T2

- **Data Split:** 70% Training, 15% Validation, 15% Testing
- **Total Images:** 16,236
- **Image Resolution:** 224x224, variable

## Execution Environment
- **Platform:** Google Colab Pro / Kaggle
- **Hardware Specifications:**
  - **RAM:** 29 GB / 13 GB
  - **GPU:** NVIDIA Tesla T4 / NVIDIA Tesla P100 / etc.
  - **CPU:** Intel Xeon CPU / AMD Ryzen (if relevant)
  - **Storage:** SSD 512GB (if relevant)

## Experiment Methodology
See [`docs/experiment_methodology.md`](docs/experiment_methodology.md) and [`docs/model_architecture_details.md`](docs/model_architecture_details.md) for full details. Summary:

1. **Data Exploration & Preprocessing**
   - Analyzed class distribution to identify imbalances.
   - Partitioned data into standard train/validation/test splits.
   - Applied augmentation (random rotations, horizontal/vertical flips, zooming/shearing, brightness adjustments) and compared augmented vs. non-augmented performance in the ResNet notebooks.

2. **Baseline Model**
   - `01_Custom_CNN.ipynb`: a CNN built from scratch with convolutional, pooling, and dense layers, used to establish a performance benchmark.

3. **Advanced Models**
   Transfer learning models fine-tuned on the dataset:
   - **DenseNet** - `02_Densenet.ipynb`
   - **EfficientNet** - `03_EfficientNet.ipynb`
   - **GoogleNet** - `04_GoogleNet.ipynb`
   - **Hybrid (DenseNet + ResNet)** - `05_Hybrid_CNN (Densenet + Restnet).ipynb`
   - **ResNet (augmented data)** - `06_RestNet on Augmented Data.ipynb`
   - **ResNet (non-augmented data)** - `07_RestNet on Non-Augmented Data.ipynb`

4. **Model Evaluation**
   - Compared all models on Accuracy, Precision, Recall, and F1-Score to select the final architecture.

## Repository Navigation
- `notebooks/`: numbered notebooks run in order, `01` through `07` (listed above)
- `utils/`: `config.py`, `data_utils.py`, `evaluation_metrices.py`, `main.py`, `modal_architectures.py`, `training_utils.py`, `visualization.py`
- `docs/`: `experiment_methodology.md`, `model_architecture_details.md`
- `results/`
  - `logs/`: `experiment_results.csv`, `training_logs.txt`, `Per Class, Per Model Evaluation.csv`
  - `plots/`: `classification_reports/`, `confusion_matrices/`, `data_visualizations/`, `training_curves/`
- `data/`
  - `raw/`: the 44 class folders listed above
  - `processed/`: see `data/processed/readme.md`

> Note: trained model weights (`.pth` files) are intentionally excluded from version control via `.gitignore` due to file size. Re-run the notebooks to regenerate them, or contact the authors for the trained checkpoint.

## How to Run the Code
1. **Clone Repository:**
   ```bash
   git clone https://github.com/infi9itea/btc44
   cd btc44
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run Experiments:**
   - Start with `notebooks/01_Custom_CNN.ipynb`
   - Follow the numerical order through `07_RestNet on Non-Augmented Data.ipynb`
   - Each notebook is self-contained with clear instructions

4. **Data Setup:**
   - Download the dataset from https://www.kaggle.com/datasets/mrnotalent/braint
   - Place it under `data/raw/` following the class-folder structure above
   - Or modify the data paths in `utils/config.py`

## Key Findings
- **Best Model:** Hybrid ResNet50 + DenseNet121
- **Best Accuracy:** 98.69%
- **Key Insights:**
  - The hybrid ResNet50 + DenseNet121 architecture achieved ~99% accuracy by fusing diverse feature sets.
  - A two-phase fine-tuning strategy (head-first, then full model) proved crucial for effective learning.
  - Extensive data augmentation was key to the model's generalization and to preventing overfitting.

## Final Model Performance
| Metric    | Value  |
| --------- | ------ |
| Accuracy  | 98.69% |
| Precision | 98.72% |
| Recall    | 98.69% |
| F1-Score  | 96.68% |

## Dependencies
- Python 3.8+
- TensorFlow 2.x / PyTorch
- NumPy, Pandas, Matplotlib, Seaborn
- Scikit-learn
- OpenCV
- (See `requirements.txt` for the complete list)

## Presentation Slide
- https://canva.link/npuc3490mes6nqa

## Citation
If you use this work, please cite:
```bibtex
@inproceedings{rubaiyat2025hybrid,
  title     = {A Hybrid Deep Learning Framework for 44-Class Brain Tumor Classification},
  author    = {Rubaiyat, Shafayat Hasnat and Hossain, MD Riyad and Naim, M. Nura Alam and Jerin, Mehejarin Aklima},
  booktitle = {28th International Conference on Computer and Information Technology (ICCIT)},
  year      = {2025},
  doi       = {10.1109/ICCIT68739.2025.11491434}
}
```

## License
This project is licensed under the [MIT License](LICENSE).

## Acknowledgements
We express our gratitude to the following individual for their guidance and support:
- **Dr. Raihan Ul Islam** - project supervision and technical guidance on machine learning methodologies.

## Contributors
All group members contributed equally:
- **Shafayat Hasnat Rubaiyat** - CNN architecture, transfer learning experiments
- **MD Riyad Hossain** - Hyperparameter tuning, data preprocessing
- **M. Nura Alam Naim** - Final integration, evaluation, visualization
- **Mehejarin Aklima Jerin** - Baseline model, documentation
