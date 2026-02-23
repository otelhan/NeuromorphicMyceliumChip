# Neuromorphic Mycelium Chip

We introduce a neuromorphic computing substrate based on PEDOT:PSS-infused mycelium, a biofabricated, morphologically tunable material that can be engineered into electrically active components including resistors, capacitors, and diode-like elements. Leveraging the principles of physical reservoir computing, we demonstrate that mycelium networks grown under controlled environmental conditions can transform time-varying inputs into nonlinear, high-dimensional state trajectories, enabling machine learning tasks such as NARMA-10 sequence prediction.

*For detailed experimental methods and results, see the primary research paper: [Morphologically Tunable Mycelium Chips for Physical Reservoir Computing](https://www.biorxiv.org/content/10.1101/2025.08.20.671348v1) (bioRxiv, 2025)* 

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
├── data/                         # Experimental data
│   ├── Original Submission (Aug 4)/
│   │   ├── NARMA-10 data/       # Original NARMA-10 training data
│   │   ├── Memory test data/    # Original memory characterization
│   │   └── *.csv                 # Summary data files
│   └── Peer-Review Revision (Feb 22)/
│       ├── Feb 14 Memory Test - k10 (Chip 39)/
│       ├── Feb 14 Step Test (Chip 39)/
│       ├── Feb 15 Tests (Chips 43 and 47)/
│       ├── Feb 17 Power Consumption Test (Chip 39)/
│       ├── Feb 20 Sine & Step tests (Chip 39)/
│       ├── MC Random Input Comparison ( Chips 39  43 47)/
│       └── Table 1 revision/
├── images/                       # Figures and diagrams
├── requirements.txt              # Python dependencies
├── run_narma10.sh               # NARMA-10 experiment launcher
├── run_memory_test.sh           # Memory test launcher
├── check_digilent_setup.sh     # Device diagnostic script
└── test_device_access.py        # Device connectivity test
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
chmod +x run_narma10.sh run_memory_test.sh check_digilent_setup.sh
```

### Running Experiments

#### NARMA-10 Reservoir Computing
```bash
./run_narma10.sh
```

#### Memory Analysis
```bash
./run_memory_test.sh
```

## 🔧 Hardware Requirements

### Digilent Devices
- **Device 1**: Analog Discovery 2 (AD2) for voltage control and measurement
- **Device 2**: Analog Discovery 3 for additional voltage control
- **Connection**: USB connection to both devices
- **Channels**: 16 independent input channels operating in parallel

### Mycelium Setup
- **Substrate**: Mycelium network on appropriate growth medium
- **Electrodes**: Conductive electrodes for voltage application and measurement
- **Environment**: Controlled humidity and temperature conditions
- **Carrier Board**: Analog interface with 4x gain amplification (0-5V to 16-18V)
- **Reservoirs**: 3 reservoirs (locations A1, A4, A6) across 3 chips

## 📊 Results

## Available Experimental Data

This repository includes experimental datasets underlying the published neuromorphic mycelium chip study, including:

**NARMA-10 Reservoir Computing**
- `narma10_results_*.csv`: Input, reservoir state, target output, and model predictions
- Ridge regression readout with nonlinear feature expansion
- Multiple independent experimental runs

**Temporal Memory Characterization**
- `memory_test_step_response_test_*.csv`: Step-response dynamics (time, input, state)
- `memory_test_random_input_*.csv`: Random input stimulation
- `memory_test_sine_wave_test_*.csv`: Sinusoidal drive experiments
- Lag-resolved memory capacity analysis

All datasets contain timestamped voltage inputs and corresponding reservoir state measurements recorded from PEDOT:PSS-infused mycelium substrates.

---

## NARMA-10 Reservoir Performance

We evaluated the mycelium substrate using the standard NARMA-10 benchmark task under the following conditions:

- **Sampling interval:** dt = 300 ms (3.33 Hz)
- **Input range:** 1–4 V control (mapped to nonlinear substrate bias regime)
- **Training samples:** 1,000 per run
- **Readout:** Ridge regression
- **Feature set:**  
  - Raw reservoir state  
  - Squared state  
  - `sin(3·state)`  
  - `cos(2·state)`

### Performance (Chip #39 baseline)

- **RMSE:** 0.102–0.106  
- **NRMSE:** 1.01–1.07  

Performance was reproducible across repeated runs using identical hardware and stimulation settings.

While absolute accuracy is modest relative to optimized electronic reservoirs, these results demonstrate that a biodegradable, morphologically engineered substrate can support nonlinear temporal transformation and short-horizon memory sufficient for NARMA-10 reconstruction.

The reported performance reflects:
- Limited linear memory capacity (MC ≤ 0.21)
- Single-scalar readout aggregation
- Moderate signal-to-noise ratio (5–8 dB)

This benchmark establishes proof-of-concept temporal computing in a biologically grown, biodegradable substrate.

---

## Memory Capacity and Temporal Dynamics

### Linear Memory Capacity (MC)

Lag-resolved linear memory capacity was measured up to K = 10 (3 s lookback) across three independently fabricated chips:

- **Chip #39:** 0.209 ± 0.067  
- **Chip #47:** 0.052 ± 0.023  
- **Chip #43:** 0.048 ± 0.016  

Each value represents mean ± SD across five independent runs (n = 5 per chip).

All devices exhibit:

- Monotonic MC decay with increasing lag  
- Finite fading-memory behavior  
- Preserved decay structure across morphologies  

Normalized MC curves confirm conserved temporal decay profiles despite amplitude scaling differences.

---

### Step-Response Dynamics

Repeated step-response measurements (n = 5 per chip) demonstrate:

- Fast initial relaxation within ~1–2 sampling intervals  
- Sub-second effective response channel  
- Superimposed slower drift component  
- Negligible within-session drift (~10⁻⁶ V/s)

Signal-to-noise ratio (SNR), computed from plateau amplitude relative to steady-state noise:

- **Chip #39:** 5.1 ± 0.15 dB  
- **Chip #47:** 8.0 ± 0.25 dB  
- **Chip #43:** 7.8 ± 0.32 dB  

These results confirm reproducible short-term temporal dynamics across independently grown substrates.

---

## Session-to-Session Stability (Chip #39)

Lag-resolved memory capacity was re-measured approximately eight months after initial characterization.

Key observations:

- Preserved fading-memory structure  
- Redistribution of lag weighting toward k = 1  
- Stable total memory magnitude within expected variability  

This indicates that substrate-level temporal behavior is stable over extended ambient storage under controlled laboratory reactivation.

---

## Board-Level Energy (Prototype Interface)

During random-input memory testing:

- **Supply voltage:** 24 V  
- **Mean current:** ~0.113 A  
- **Board-level power:** ~2.7 W  
- **Energy per sample:** ~0.81 J/sample  

Power consumption is dominated by high-voltage analog conditioning electronics (OPA552/OPA454 chain), not intrinsic substrate dissipation.

These measurements reflect prototype interface energy rather than optimized substrate-level energy scaling.

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
1. Generate 1,000 NARMA-10 training samples
2. Apply voltage signals (1–4 V control range, mapped to nonlinear substrate bias regime) to 3 reservoirs (A1, A4, A6)
3. Record mycelium state responses from summed output channels at 300 ms sampling intervals (3.33 Hz)
4. Create feature representations (raw state, squared state, trigonometric expansions: sin(3·state), cos(2·state))
5. Train Ridge regression model
6. Evaluate prediction performance across multiple independent experimental runs



### Memory Analysis
1. Apply three input types: step signals, random pulses, sine waves
2. Measure autocorrelation and cross-correlation metrics
3. Analyze temporal dynamics across three reservoirs
4. Quantify settling time and memory retention
5. Evaluate R² improvement with historical inputs

## 🔬 Research Significance

This project demonstrates that mycelium networks can serve as effective substrates for neuromorphic computing, providing:
- **Reservoir Computing**: PEDOT:PSS-based nonlinear dynamics
- **Temporal Memory**: Inherent memory capabilities
- **Scalability**: Network growth and adaptation
- **Sustainability**: Biodegradable computing substrates

## 📚 References

### Primary Research Paper
- **Telhan, O. et al.** (2025). Morphologically Tunable Mycelium Chips for Physical Reservoir Computing. *bioRxiv* [https://www.biorxiv.org/content/10.1101/2025.08.20.671348v1](https://www.biorxiv.org/content/10.1101/2025.08.20.671348v1)

### Background Literature
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

---

**Note**: This project requires specialized hardware (Digilent devices) and biological materials (mycelium chips). Full replication requires access to these resources. 