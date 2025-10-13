# 🧠 CMI-BFRB-Detection

---

### A Precision-Optimized Multimodal CNN for Detecting Body-Focused Repetitive Behaviors

This repository contains the implementation of a sophisticated, precision-optimized PyTorch neural network designed to detect Body-Focused Repetitive Behaviors (BFRBs). The model leverages a multimodal approach, fusing data from various sensors—IMU, thermopile, and Time-of-Flight (ToF)—along with demographic information to achieve high-accuracy classifications.

The primary goal is to provide a robust deep learning framework for identifying specific BFRB gestures, addressing challenges like class imbalance and the need for high precision in a clinical or research context.

---

## ✨ Key Features

*   **🧠 Multimodal Architecture**: Fuses data from four distinct branches (IMU, Thermopile, ToF, Demographic) for a comprehensive understanding of user actions.
*   **🎯 Multi-Task Learning**: Simultaneously performs binary classification (Target vs. Non-target) and fine-grained multiclass classification (18 specific gestures).
*   **⚖️ Advanced Loss Function**: Implements Asymmetric Focal Loss to optimize for precision, which is critical in BFRB detection to minimize false positives.
*   **🔧 Attention Mechanisms**: Utilizes self-attention on fused features and a dedicated attention mechanism for the ToF branch to focus on the most relevant sensor data.
*   **📈 Class Imbalance Handling**: Employs sample weighting to counteract the effects of imbalanced datasets, a common issue in behavioral data.
*   **🔄 Time-Series Data Augmentation**: Enhances model generalization by applying augmentation techniques suitable for sequential sensor data.

---

## 📸 Showcase

> A visual representation of the model's architecture and performance metrics.

*(placeholder for model architecture diagram)*
`![Model Architecture](path/to/architecture_diagram.png)`

*(placeholder for performance plots, e.g., confusion matrix or PR curve)*
`![Performance Metrics](path/to/performance_plot.png)`

---

## 🛠️ Tech Stack & Tools

*   **Language**: Python 3
*   **Core Framework**: PyTorch
*   **Primary Libraries**:
    *   `Jupyter` / `Google Colab`: For model development and experimentation.
    *   `NumPy`: For numerical operations.
    *   `Pandas`: For data manipulation and loading.
    *   `Scikit-learn`: For metrics, data splitting, and utility functions.
    *   `Matplotlib` / `Seaborn`: For data visualization.

---

## 🚀 Getting Started

Follow these instructions to get a local copy of the project up and running for development and testing purposes.

### Prerequisites

*   Python 3.8+
*   `pip` and `venv`
*   An NVIDIA GPU with CUDA and cuDNN is highly recommended for training. This model was developed on a T4 GPU.

### Installation

1.  **Clone the repository:**
    ```sh
    git clone https://github.com/your-username/CMI-BFRB-Detection.git
    cd CMI-BFRB-Detection
    ```

2.  **Create and activate a virtual environment:**
    ```sh
    # For Unix/macOS
    python3 -m venv venv
    source venv/bin/activate

    # For Windows
    python -m venv venv
    .\venv\Scripts\activate
    ```

3.  **Install the required dependencies:**
    *(Note: A `requirements.txt` file is recommended. For now, install key packages manually.)*
    ```sh
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    pip install jupyterlab numpy pandas scikit-learn matplotlib seaborn
    ```

### Running the Project

1.  **Launch Jupyter Lab:**
    ```sh
    jupyter lab
    ```

2.  **Open and run the notebook:**
    *   Navigate to `CNN_Notebook.ipynb` in the Jupyter Lab interface.
    *   Execute the cells sequentially to load data, define the model, train it, and evaluate its performance.

Alternatively, you can upload the `CNN_Notebook.ipynb` to Google Colab to leverage their free GPU resources. Be sure to select a GPU runtime (`Runtime > Change runtime type > T4 GPU`).

---

## 📖 Usage

The `CNN_Notebook.ipynb` serves as the primary entry point and a comprehensive guide to the project. It is structured to be a self-contained workflow:

1.  **Data Loading & Preprocessing**: Contains scripts to load the multimodal sensor data and demographic information, followed by preprocessing steps.
2.  **Model Definition**: The complete PyTorch model, including all branches (IMU, Thermopile, ToF, Demographics) and the fusion mechanism, is defined here.
3.  **Training Loop**: The training procedure, including the custom loss function, optimizer, and threshold optimization logic.
4.  **Evaluation**: Code to evaluate the trained model's performance using metrics like precision, recall, and F1-score.

You can adapt this notebook by replacing the data loading section with your own BFRB dataset, provided it follows a similar structure.

---

## 📂 Project Structure

The repository structure is straightforward:

```
CMI-BFRB-Detection/
├── 📄 CNN_Notebook.ipynb    # Main notebook with all code for model definition, training, and evaluation.
├── 📄 LICENSE               # Project license file.
└── 📄 README.md             # This file.
```

---

## 🤝 Contributing

Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".

1.  Fork the Project
2.  Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3.  Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4.  Push to the Branch (`git push origin feature/AmazingFeature`)
5.  Open a Pull Request

Please see `CONTRIBUTING.md` for more details on our code of conduct and the process for submitting pull requests.

---

## 📜 License

This project is distributed under the terms of the license specified in the `LICENSE` file.

---

## 🙏 Acknowledgements

*   Hat tip to the creators of the libraries and frameworks used in this project.
*   Special thanks to the research community focused on BFRBs and wearable sensor technology.

---

<p align="center">
  Star this repository if you find it useful! ⭐
</p>
