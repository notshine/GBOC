# [AAAI 2026] Finding Time Series Anomalies Using Granular-Ball Vector Data Description

## 📋 Table of Contents

- [Installation](#installation)
- [Dataset](#dataset)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Results](#results)
- [Citation](#citation)

## 🔧 Installation

### Prerequisites

- Python 3.10
- CUDA 11.6 (optional, for GPU acceleration)

### Install Dependencies

```bash
# Clone the repository
git clone https://github.com/notshine/GBOC.git
cd GBOC

# Install required packages
pip install -r requirements.txt
```

### Key Dependencies

```
torch==1.13.1
numpy==1.26.4
pandas==2.2.3
scikit-learn==1.6.1
matplotlib==3.9.4
tqdm==4.67.1
```

## 📊 Dataset

The model is designed for time series anomaly detection and has been tested on:

- **TSB-AD-U**:  <https://www.thedatum.org/datasets/TSB-AD-U.zip>
- **TSB-AD-M**：<https://www.thedatum.org/datasets/TSB-AD-M.zip>

## 🚀 Quick Start

### Basic Usage

```bash
# Run with default parameters
python main.py --AD_Name GBOC --filename 001_NAB_id_1_Facility_tr_1007_1st_2014

# Specify data directory
python main.py --AD_Name GBOC --data_direc TSB-AD-U --filename 001_NAB_id_1_Facility_tr_1007_1st_2014
```

### Advanced Usage with Custom Parameters

```bash
# Override specific hyperparameters
python main.py --AD_Name GBOC \
    --filename 001_NAB_id_1_Facility_tr_1007_1st_2014 \
    --win_size 10 \
    --lr 0.001 \
    --hidden_dim 64 \
    --num_layers 3 \
    --alpha 0.9 \
    --epochs 50
```

## ⚙️ Configuration

### Method 1: HP_list.py Configuration

Edit `HP_list.py` to set default parameters:

```python
Optimal_Uni_algo_HP_dict = {
    'GBOC': {
        'win_size': 5,        # Sliding window size
        'lr': 0.0001,         # Learning rate
        'hidden_dim': 32,     # LSTM hidden dimension
        'num_layers': 3,      # Number of LSTM layers
        'alpha': 0.9,         # Loss combination weight
        'epochs': 30          # Training epochs
    },
}
```

### Method 2: Command-Line Arguments

| Argument       | Type  | Default  | Description                              |
| -------------- | ----- | -------- | ---------------------------------------- |
| `--AD_Name`    | str   | GBOC     | Algorithm name                           |
| `--data_direc` | str   | TSB-AD-U | Dataset directory                        |
| `--filename`   | str   | -        | Input data filename                      |
| `--win_size`   | int   | 5        | Sliding window size                      |
| `--lr`         | float | 0.0001   | Learning rate                            |
| `--hidden_dim` | int   | 32       | LSTM hidden dimension                    |
| `--num_layers` | int   | 3        | Number of LSTM layers                    |
| `--alpha`      | float | 0.9      | Weight for loss combination (α × reconstruction + (1-α) × gb) |
| `--epochs`     | int   | 30       | Number of training epochs                |
| `--save_dir`   | str   | eval     | Directory to save results                |

**Parameter Priority**: Command-line arguments > HP_list.py configuration

## 📊 Results

Results are automatically saved to:

- **Anomaly Scores**: `score/uni/GBOC/{filename}.npy`
- **Evaluation Metrics**: `eval/uni/GBOC.csv`
- **Visualizations**: `plots_results/{filename}_GBOC/`

## 📁 Project Structure

```
GBOC/
├── main.py                          # Main entry point
├── model_wrapper.py                 # Model wrapper functions
├── HP_list.py                       # Hyperparameter configurations
├── data_func.py                     # Data loading utilities
├── plot_ab_sco.py                   # Visualization tools
├── requirements.txt                 # Python dependencies
├── models/
│   ├── GBOC.py                     # Main model implementation
│   └── base.py                     # Base detector class
├── granularball_computing/
│   └── granularball_generation.py  # Granular ball algorithms
├── lib_instance_discrimination/
│   └── ...                         # Instance discrimination modules
├── evaluation/
│   └── metrics.py                  # Evaluation metrics
├── utils/
│   ├── dataset.py                  # Dataset classes
│   ├── torch_utility.py            # PyTorch utilities
│   └── slidingWindows.py           # Sliding window functions
└── TSB-AD-U/                       # Dataset directory
    └── *.csv                       # Time series data files
```

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{GBOC,
  title={Finding Time Series Anomalies Using Granular-Ball Vector Data Description},
  author={Shen, Lifeng and Peng, Liang and Liu, Ruiwen and Xia, Shuyin and Liu, Yi},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2026}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📧 Contact

For questions or issues, please contact:

- Liang Peng : l1angpeng@foxmail.com.

## 🙏 Acknowledgments

We appreciate the following github repos a lot for their valuable code base:

- https://github.com/TheDatumOrg/TSB-AD

---

**Note**: This is a research project. For production use, please conduct thorough testing and validation.
