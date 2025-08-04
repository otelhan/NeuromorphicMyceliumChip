# Neuromorphic Mycelium Chip

We introduce a neuromorphic computing substrate based on PEDOT:PSS-infused mycelium, a biofabricated, morphologically tunable material that can be engineered into electrically active components including resistors, capacitors, and diode-like elements. Leveraging the principles of physical reservoir computing, we demonstrate that mycelium networks grown under controlled environmental conditions can transform time-varying inputs into nonlinear, high-dimensional state trajectories, enabling machine learning tasks such as NARMA-10 sequence prediction. 

## Overview

<img src="./images/mycelium_compute_chip_v2.jpg" alt="Mycelium Computer Chip" height="300" width="400">

This project implements neuromorphic computing systems using mycelium networks as biological substrates for:
- **NARMA-10 Reservoir Computing**: Time-series prediction using mycelium as a reservoir
- **Temporal Memory Analysis**: Characterization of mycelium's memory capabilities

## 🔬 Research Contributions

### Neuromorphic Computing with Mycelium
- First demonstration of mycelium-based reservoir computing
- Temporal memory characterization in mycelium networks
- Hardware-software integration with Digilent devices

### Key Achievements
- **NARMA-10 Benchmark**: Successfully trained mycelium networks for time-series prediction
- **Memory Analysis**: Quantified temporal memory effects in mycelium
- **Hardware Integration**: Real-time voltage control and measurement

## 📁 Repository Structure

```
NeuromorphicMyceliumChip/
├── src/                          # Core implementation
│   ├── mycelium_narma10.py      # NARMA-10 reservoir computing
│   ├── mycelium_memory_test.py  # Memory analysis tools
│   └── cleanup_devices.py       # Device management utilities
├── data/                         # Sample data files
│   └── sample_narma_results.csv # NARMA-10 experimental results
├── docs/                         # Documentation
├── tests/                        # Test files
├── examples/                     # Usage examples
├── requirements.txt              # Python dependencies
└── run_predictor.sh             # Environment setup script
```

## 🚀 Quick Start

### Prerequisites
- Python 3.12+
- Digilent WaveForms framework
- Digilent Analog Discovery 2 (AD2) or similar devices

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/NeuromorphicMyceliumChip.git
cd NeuromorphicMyceliumChip
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Setup Digilent environment**
```bash
# Install Digilent WaveForms from https://digilent.com/reference/software/waveforms/
# Connect your Digilent devices
chmod +x run_predictor.sh
```

### Running Experiments

#### NARMA-10 Reservoir Computing
```bash
./run_predictor.sh src/mycelium_narma10.py
```



#### Memory Analysis
```bash
./run_predictor.sh src/mycelium_memory_test.py
```

## 🔧 Hardware Requirements

### Digilent Devices
- **Device 1**: Analog Discovery 2 (AD2) for voltage control and measurement
- **Device 2**: Second Digilent device for additional voltage control
- **Connection**: USB connection to both devices

### Mycelium Setup
- **Substrate**: Mycelium network on appropriate growth medium
- **Electrodes**: Conductive electrodes for voltage application and measurement
- **Environment**: Controlled humidity and temperature conditions

## 📊 Results

### NARMA-10 Performance
- **Training Samples**: 1,000 samples per model
- **Test Performance**: 200+ test samples
- **Model Optimization**: Multiple iterations with parameter tuning
- **Results**: 
  - **Ridge Regression**: Training RMSE 0.119, Test RMSE 0.102 (best performance)
  - **Random Forest**: Training RMSE 0.096, Test RMSE 0.197
- **Method**: Ridge regression with normalization and Random Forest with nonlinear feature transformations
- **Features**: Reservoir state, squared state, and trigonometric expansions (sin(state×3), cos(state×2))



### Memory Characterization
- **Temporal Memory**: R² improvement of 0.1-0.3 when historical inputs included
- **Response Dynamics**: Step response and autocorrelation analysis
- **State Prediction**: Predictive modeling with input history
- **Settling Time**: ~100ms (one sample interval at optimal timing)
- **Autocorrelation**: Strong correlation (r > 0.5) at lags of 1-3 samples
- **Cross-correlation**: Significant correlation (|r| > 0.3) between past inputs and current states

## 🛠️ Technical Details

### Software Architecture
- **GUI Framework**: PyQt6 for user interfaces
- **Machine Learning**: scikit-learn for classification and regression
- **Data Analysis**: pandas, numpy, matplotlib for analysis
- **Hardware Interface**: ctypes for Digilent device communication

### Key Algorithms
- **Reservoir Computing**: Mycelium as nonlinear dynamical system
- **Ridge Regression**: For NARMA-10 prediction
- **t-SNE Visualization**: High-dimensional state analysis

## 📈 Experimental Protocol

### NARMA-10 Training
1. Generate NARMA-10 input sequences
2. Apply voltage signals to mycelium network
3. Record mycelium state responses
4. Train readout layer (Ridge regression)
5. Evaluate prediction performance



### Memory Analysis
1. Apply step response tests
2. Measure autocorrelation functions
3. Analyze cross-correlation patterns
4. Characterize temporal memory effects

## 🔬 Research Significance

This project demonstrates that mycelium networks can serve as effective substrates for neuromorphic computing, providing:
- **Biological Computing**: Natural nonlinear dynamics
- **Temporal Memory**: Inherent memory capabilities
- **Scalability**: Network growth and adaptation
- **Sustainability**: Biodegradable computing substrates

## 📚 References

- Reservoir Computing principles
- NARMA-10 benchmark task
- Mycelium network properties
- Neuromorphic computing architectures

## 🤝 Contributing

This is a research project. For questions or collaboration, please contact the research team.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- DARPA for research funding
- Digilent for hardware support
- Research collaborators and advisors

---

**Note**: This project requires specialized hardware (Digilent devices) and biological materials (mycelium chips). Full replication requires access to these resources. 