#!/usr/bin/env python3
"""
Mycelium Color Classification System
Compares single sample vs sequential presentation methods

Sequential Variation Modes:
1. Progressive Intensity: Gradually increases stimulus strength from weak to full intensity
   - Tests mycelium's ability to integrate temporal intensity changes
   - Sample 1: 33% intensity, Sample 2: 67% intensity, Sample 3: 100% intensity
   - More biologically realistic than identical repetitions
   
2. NARMA-10 Noise Progression: Gradually increases noise level from 0% to max
   - Tests robustness to temporal noise patterns
   - Sample 1: clean signal, Sample 2: medium noise, Sample 3: full noise
   - Based on successful NARMA-10 reservoir computing approach
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
import pandas as pd
from datetime import datetime
import time
import os
from ctypes import c_int, c_double, c_uint, byref, create_string_buffer, cdll, c_bool, c_void_p
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.model_selection import train_test_split
import pickle
import random
import csv

try:
    from PyQt6.QtWidgets import *
    from PyQt6.QtCore import *
    from PyQt6.QtGui import *
except ImportError:
    print("PyQt6 not found. Installing...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "PyQt6"])
    from PyQt6.QtWidgets import *
    from PyQt6.QtCore import *
    from PyQt6.QtGui import *

class ColorClassifierApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Mycelium Color Classification - Single vs Sequential")
        self.setMinimumSize(1200, 900)
        
        # Parameters based on memory test and NARMA-10 results
        self.single_duration = 50  # ms - Brief stimulation for direct response (minimize temporal memory)
        self.reset_delay = 3000  # ms - Longer delay for better color separation (based on NARMA-10 optimization)
        self.sequence_length = 10  # Default 10 samples per color - matches NARMA-10 success (now configurable)
        self.memory_window = 1.0  # seconds - NARMA-10 optimal timing (10 × 100ms = 1s total)
        
        # Calculate derived parameter
        self.sample_interval = self.memory_window / self.sequence_length  # seconds per sample
        
        # Color definitions (RGB values 0-255)
        self.colors = {
            'RED': (255, 0, 0),
            'ORANGE': (255, 165, 0),
            'YELLOW': (255, 255, 0),
            'GREEN': (0, 255, 0),
            'BLUE': (0, 0, 255),
            'INDIGO': (75, 0, 130),
            'VIOLET': (148, 0, 211)
        }
        
        # Data storage
        self.single_data = {'states': [], 'labels': [], 'noise_levels': []}
        self.sequential_data = {'states': [], 'labels': [], 'noise_levels': []}
        
        # Models
        self.single_model = None
        self.sequential_model = None
        
        # Device handles - using same pattern as NARMA-10 app
        self.dwf = None
        self.hdwf1 = None
        self.hdwf2 = None
        
        # UI state
        self.is_collecting = False
        self.current_color_index = 0
        self.current_trial = 0
        self.total_trials_per_color = 50
        self.current_mode = "single"  # "single" or "sequential"
        self.noise_level = 0.0  # 0.0 to 0.3 (0-30% noise)
        
        # Setup UI first
        self.setup_ui()
        
        # Initialize Digilent devices - MANDATORY
        device_init_success = self.setup_digilent()
        if device_init_success and self.dwf:
            self.status_label.setText("Devices initialized successfully - Ready to collect")
            self.collect_btn.setEnabled(True)
            self.train_single_btn.setEnabled(True)
            self.train_sequential_btn.setEnabled(True)
            self.compare_btn.setEnabled(True)
        else:
            self.status_label.setText("❌ DEVICE INITIALIZATION FAILED - Hardware required")
            self.status_label.setStyleSheet("color: red; font-weight: bold;")
            # Disable all controls
            self.collect_btn.setEnabled(False)
            self.stop_btn.setEnabled(False)
            self.reset_btn.setEnabled(False)
            self.train_single_btn.setEnabled(False)
            self.train_sequential_btn.setEnabled(False)
            self.compare_btn.setEnabled(False)
            self.save_data_btn.setEnabled(False)
            self.save_analysis_btn.setEnabled(False)
            
            # Show error dialog
            error_dialog = QMessageBox()
            error_dialog.setIcon(QMessageBox.Icon.Critical)
            error_dialog.setWindowTitle("Hardware Error")
            error_dialog.setText("Failed to initialize Digilent devices!")
            error_dialog.setInformativeText(
                "This application requires both Digilent devices to be connected.\n\n"
                "Please check:\n"
                "• Both devices are connected via USB\n"
                "• Drivers are properly installed\n"
                "• No other applications are using the devices\n"
                "• Devices are powered on"
            )
            error_dialog.exec()

    def setup_ui(self):
        """Create the user interface"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Create tabs
        tabs = QTabWidget()
        main_layout.addWidget(tabs)
        
        # Collection Tab
        collection_tab = QWidget()
        collection_layout = QVBoxLayout(collection_tab)
        tabs.addTab(collection_tab, "Data Collection")
        
        # Parameters Group
        param_group = QGroupBox("Collection Parameters")
        param_layout = QGridLayout()
        param_group.setLayout(param_layout)
        
        # Presentation mode selection
        param_layout.addWidget(QLabel("Presentation Mode:"), 0, 0)
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Single Sample", "Sequential (3x)"])
        self.mode_combo.currentTextChanged.connect(self.update_mode)
        param_layout.addWidget(self.mode_combo, 0, 1)
        
        # Voltage range for color mapping
        param_layout.addWidget(QLabel("Max Color Voltage (V):"), 1, 0)
        self.max_voltage_spin = QDoubleSpinBox()
        self.max_voltage_spin.setRange(2.0, 5.0)
        self.max_voltage_spin.setValue(3.5)  # Default to NARMA-10 optimal
        self.max_voltage_spin.setSingleStep(0.1)
        self.max_voltage_spin.setToolTip("Maximum voltage for color mapping (3.5V optimal from NARMA-10, 4V for max coupling)")
        param_layout.addWidget(self.max_voltage_spin, 1, 1)
        
        # Min voltage for color mapping
        param_layout.addWidget(QLabel("Min Color Voltage (V):"), 1, 2)
        self.min_voltage_spin = QDoubleSpinBox()
        self.min_voltage_spin.setRange(0.0, 2.0)
        self.min_voltage_spin.setValue(0.0)  # Match NARMA-10 range (0V to 3.5V)
        self.min_voltage_spin.setSingleStep(0.1)
        self.min_voltage_spin.setToolTip("Minimum voltage for color mapping (0V from NARMA-10 setup)")
        param_layout.addWidget(self.min_voltage_spin, 1, 3)
        
        # Memory window for sequential mode
        param_layout.addWidget(QLabel("Memory Window (s):"), 2, 0)
        self.memory_window_spin = QDoubleSpinBox()
        self.memory_window_spin.setRange(1.0, 10.0)
        self.memory_window_spin.setValue(self.memory_window)  # Default to NARMA-10 optimal 1s
        self.memory_window_spin.setSingleStep(0.1)
        self.memory_window_spin.setToolTip("Total duration for sequential presentations (1s optimal from NARMA-10: 10×100ms)")
        param_layout.addWidget(self.memory_window_spin, 2, 1)
        
        # Number of samples for sequential mode
        param_layout.addWidget(QLabel("Samples in Sequence:"), 2, 2)
        self.sequence_length_spin = QSpinBox()
        self.sequence_length_spin.setRange(2, 10)  # 2-10 samples in sequence
        self.sequence_length_spin.setValue(self.sequence_length)  # Default to 10 (NARMA-10 optimal)
        self.sequence_length_spin.setSingleStep(1)
        self.sequence_length_spin.setToolTip("Number of samples in sequential presentation (10 optimal from NARMA-10: 100ms intervals)")
        param_layout.addWidget(self.sequence_length_spin, 2, 3)
        
        # Sequential variation type
        param_layout.addWidget(QLabel("Sequential Variation:"), 3, 0)
        self.variation_combo = QComboBox()
        self.variation_combo.addItems(["Progressive Intensity", "NARMA-10 Noise Progression"])
        self.variation_combo.setToolTip("Progressive Intensity: gradually increase stimulus strength 0% to 100%; NARMA-10: progressive noise 0% to max_noise%")
        param_layout.addWidget(self.variation_combo, 3, 1)
        
        # Reset delay
        param_layout.addWidget(QLabel("Reset Delay (ms):"), 4, 0)
        self.reset_delay_spin = QSpinBox()
        self.reset_delay_spin.setRange(500, 10000)
        self.reset_delay_spin.setValue(self.reset_delay)  # Default to 3000ms
        self.reset_delay_spin.setSingleStep(500)
        self.reset_delay_spin.setToolTip("Delay between different colors to clear temporal memory (3s for better separation)")
        param_layout.addWidget(self.reset_delay_spin, 4, 1)

        # Single sample duration
        param_layout.addWidget(QLabel("Single Duration (ms):"), 4, 2)
        self.single_duration_spin = QSpinBox()
        self.single_duration_spin.setRange(10, 2000)  # Allow very short stimulations for direct response
        self.single_duration_spin.setValue(self.single_duration)  # Default to 50ms
        self.single_duration_spin.setSingleStep(10)  # Finer control for short durations
        self.single_duration_spin.setToolTip("Stimulation duration for single sample mode (50ms for direct response, minimize memory effects)")
        param_layout.addWidget(self.single_duration_spin, 4, 3)

        # Trials per color
        param_layout.addWidget(QLabel("Trials per Color:"), 5, 0)
        self.trials_spin = QSpinBox()
        self.trials_spin.setRange(10, 200)
        self.trials_spin.setValue(self.total_trials_per_color)
        param_layout.addWidget(self.trials_spin, 5, 1)
        
        # Noise level
        param_layout.addWidget(QLabel("Noise Level (0-30%):"), 6, 0)
        self.noise_spin = QDoubleSpinBox()
        self.noise_spin.setRange(0.0, 30.0)
        self.noise_spin.setValue(self.noise_level * 100)  # Convert from decimal to percentage
        self.noise_spin.setSingleStep(5.0)
        self.noise_spin.setSuffix("%")
        param_layout.addWidget(self.noise_spin, 6, 1)
        
        collection_layout.addWidget(param_group)
        
        # Collection Controls
        control_group = QGroupBox("Collection Controls")
        control_layout = QHBoxLayout()
        control_group.setLayout(control_layout)
        
        self.collect_btn = QPushButton("Start Collection")
        self.collect_btn.clicked.connect(self.start_collection)
        self.collect_btn.setEnabled(False)
        control_layout.addWidget(self.collect_btn)
        
        self.stop_btn = QPushButton("Stop Collection")
        self.stop_btn.clicked.connect(self.stop_collection)
        self.stop_btn.setEnabled(False)
        control_layout.addWidget(self.stop_btn)
        
        self.reset_btn = QPushButton("Reset Data")
        self.reset_btn.clicked.connect(self.reset_data)
        control_layout.addWidget(self.reset_btn)
        
        collection_layout.addWidget(control_group)
        
        # Progress and Status
        progress_group = QGroupBox("Collection Progress")
        progress_layout = QVBoxLayout()
        progress_group.setLayout(progress_layout)
        
        self.progress_bar = QProgressBar()
        progress_layout.addWidget(self.progress_bar)
        
        self.current_color_label = QLabel("Ready to collect")
        progress_layout.addWidget(self.current_color_label)
        
        self.status_label = QLabel("Initializing devices...")
        progress_layout.addWidget(self.status_label)
        
        # Data summary
        self.data_summary = QTextEdit()
        self.data_summary.setMaximumHeight(100)
        self.data_summary.setReadOnly(True)
        progress_layout.addWidget(self.data_summary)
        
        collection_layout.addWidget(progress_group)
        
        # Training Tab
        training_tab = QWidget()
        training_layout = QVBoxLayout(training_tab)
        tabs.addTab(training_tab, "Training & Comparison")
        
        # Training Controls
        train_group = QGroupBox("Model Training")
        train_layout = QHBoxLayout()
        train_group.setLayout(train_layout)
        
        self.train_single_btn = QPushButton("Train Single Sample Model")
        self.train_single_btn.clicked.connect(self.train_single_model)
        train_layout.addWidget(self.train_single_btn)
        
        self.train_sequential_btn = QPushButton("Train Sequential Model")
        self.train_sequential_btn.clicked.connect(self.train_sequential_model)
        train_layout.addWidget(self.train_sequential_btn)
        
        self.compare_btn = QPushButton("Compare Models")
        self.compare_btn.clicked.connect(self.compare_models)
        train_layout.addWidget(self.compare_btn)
        
        self.save_data_btn = QPushButton("Save Training Data (CSV)")
        self.save_data_btn.clicked.connect(self.save_training_data_csv)
        train_layout.addWidget(self.save_data_btn)
        
        self.save_analysis_btn = QPushButton("Save Analysis Report (PDF)")
        self.save_analysis_btn.clicked.connect(self.save_analysis_pdf)
        train_layout.addWidget(self.save_analysis_btn)
        
        training_layout.addWidget(train_group)
        
        # Results display
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        training_layout.addWidget(self.results_text)
        
        # Model Evaluation Tab
        eval_tab = QWidget()
        eval_layout = QVBoxLayout(eval_tab)
        tabs.addTab(eval_tab, "Model Evaluation")
        
        # Evaluation Controls
        eval_control_group = QGroupBox("Noise Robustness Testing")
        eval_control_layout = QGridLayout()
        eval_control_group.setLayout(eval_control_layout)
        
        # Progressive noise testing options
        eval_control_layout.addWidget(QLabel("Evaluation Mode:"), 0, 0)
        self.eval_mode_combo = QComboBox()
        self.eval_mode_combo.addItems(["Single Noise Level", "Progressive Noise (0-30%)", "Custom Noise Levels"])
        eval_control_layout.addWidget(self.eval_mode_combo, 0, 1)
        
        # Test noise level (for single noise evaluation)
        eval_control_layout.addWidget(QLabel("Test Noise Level:"), 1, 0)
        self.test_noise_spin = QDoubleSpinBox()
        self.test_noise_spin.setRange(0.0, 50.0)
        self.test_noise_spin.setValue(20.0)  # Default 20%
        self.test_noise_spin.setSingleStep(5.0)
        self.test_noise_spin.setSuffix("%")
        eval_control_layout.addWidget(self.test_noise_spin, 1, 1)
        
        # Test samples per noise level
        eval_control_layout.addWidget(QLabel("Samples per Test:"), 2, 0)
        self.test_samples_spin = QSpinBox()
        self.test_samples_spin.setRange(5, 100)
        self.test_samples_spin.setValue(20)
        eval_control_layout.addWidget(self.test_samples_spin, 2, 1)
        
        # Evaluation buttons
        eval_button_layout = QHBoxLayout()
        
        self.eval_single_btn = QPushButton("Evaluate Single Model")
        self.eval_single_btn.clicked.connect(self.evaluate_single_model)
        eval_button_layout.addWidget(self.eval_single_btn)
        
        self.eval_sequential_btn = QPushButton("Evaluate Sequential Model")
        self.eval_sequential_btn.clicked.connect(self.evaluate_sequential_model)
        eval_button_layout.addWidget(self.eval_sequential_btn)
        
        self.eval_compare_btn = QPushButton("Compare Noise Robustness")
        self.eval_compare_btn.clicked.connect(self.compare_noise_robustness)
        eval_button_layout.addWidget(self.eval_compare_btn)
        
        eval_control_layout.addLayout(eval_button_layout, 3, 0, 1, 2)
        eval_layout.addWidget(eval_control_group)
        
        # Evaluation results display
        self.eval_results_text = QTextEdit()
        self.eval_results_text.setReadOnly(True)
        eval_layout.addWidget(self.eval_results_text)
        
        # Evaluation plots
        self.eval_figure = Figure(figsize=(12, 6))
        self.eval_canvas = FigureCanvas(self.eval_figure)
        eval_layout.addWidget(self.eval_canvas)
        
        # Visualization Tab
        viz_tab = QWidget()
        viz_layout = QVBoxLayout(viz_tab)
        tabs.addTab(viz_tab, "t-SNE Visualization")
        
        # Visualization controls
        viz_control_group = QGroupBox("Visualization Controls")
        viz_control_layout = QHBoxLayout()
        viz_control_group.setLayout(viz_control_layout)
        
        self.tsne_single_btn = QPushButton("t-SNE Single Sample")
        self.tsne_single_btn.clicked.connect(self.generate_tsne_single)
        viz_control_layout.addWidget(self.tsne_single_btn)
        
        self.tsne_sequential_btn = QPushButton("t-SNE Sequential")
        self.tsne_sequential_btn.clicked.connect(self.generate_tsne_sequential)
        viz_control_layout.addWidget(self.tsne_sequential_btn)
        
        self.tsne_compare_btn = QPushButton("Compare Both")
        self.tsne_compare_btn.clicked.connect(self.generate_tsne_comparison)
        viz_control_layout.addWidget(self.tsne_compare_btn)
        
        self.save_plots_btn = QPushButton("Save Plots (PNG)")
        self.save_plots_btn.clicked.connect(self.save_plots_png)
        viz_control_layout.addWidget(self.save_plots_btn)
        
        viz_layout.addWidget(viz_control_group)
        
        # t-SNE plot area
        self.tsne_figure = Figure(figsize=(12, 6))
        self.tsne_canvas = FigureCanvas(self.tsne_figure)
        viz_layout.addWidget(self.tsne_canvas)
        
    def setup_digilent(self):
        """Initialize Digilent devices - REQUIRED for operation"""
        try:
            # Load the dwf library
            if sys.platform.startswith("win"):
                self.dwf = cdll.dwf
            elif sys.platform.startswith("darwin"):
                self.dwf = cdll.LoadLibrary("/Library/Frameworks/dwf.framework/dwf")
            else:
                self.dwf = cdll.LoadLibrary("libdwf.so")
            
            print("DWF library loaded successfully")
            
            # Check how many devices are available
            cDevice = c_int()
            self.dwf.FDwfEnum(c_int(0), byref(cDevice))
            print(f"Number of Digilent devices found: {cDevice.value}")
            
            if cDevice.value < 2:
                print(f"ERROR: Need 2 devices, only found {cDevice.value}")
                return False
            
            # Variables for device handles
            self.hdwf1 = c_int()
            self.hdwf2 = c_int()
            
            # Open first device
            print("Opening first device...")
            self.dwf.FDwfDeviceOpen(c_int(0), byref(self.hdwf1))
            if self.hdwf1.value == 0:
                print("ERROR: Failed to open first device")
                return False
            print(f"First device opened with handle: {self.hdwf1.value}")
            
            # Open second device
            print("Opening second device...")
            self.dwf.FDwfDeviceOpen(c_int(1), byref(self.hdwf2))
            if self.hdwf2.value == 0:
                print("ERROR: Failed to open second device") 
                # Close first device if second fails
                self.dwf.FDwfDeviceClose(self.hdwf1)
                return False
            print(f"Second device opened with handle: {self.hdwf2.value}")
            
            # Configure analog output for both devices
            print("Configuring analog outputs...")
            for hdwf in [self.hdwf1, self.hdwf2]:
                channels = [0, 1] if hdwf == self.hdwf1 else [0]
                for channel in channels:
                    self.dwf.FDwfAnalogOutNodeEnableSet(hdwf, c_int(channel), c_int(0), c_bool(True))
                    self.dwf.FDwfAnalogOutNodeFunctionSet(hdwf, c_int(channel), c_int(0), c_int(1))  # DC
                    self.dwf.FDwfAnalogOutNodeFrequencySet(hdwf, c_int(channel), c_int(0), c_double(1000))
                    self.dwf.FDwfAnalogOutNodeAmplitudeSet(hdwf, c_int(channel), c_int(0), c_double(0))
                    self.dwf.FDwfAnalogOutNodeOffsetSet(hdwf, c_int(channel), c_int(0), c_double(2.5))
                    self.dwf.FDwfAnalogOutConfigure(hdwf, c_int(channel), c_bool(True))
            
            # Configure analog input on first device
            print("Configuring analog input...")
            self.dwf.FDwfAnalogInChannelEnableSet(self.hdwf1, c_int(0), c_bool(True))
            self.dwf.FDwfAnalogInChannelRangeSet(self.hdwf1, c_int(0), c_double(10.0))
            self.dwf.FDwfAnalogInAcquisitionModeSet(self.hdwf1, c_int(3))  # Single acquisition
            self.dwf.FDwfAnalogInFrequencySet(self.hdwf1, c_double(1000.0))
            self.dwf.FDwfAnalogInBufferSizeSet(self.hdwf1, c_int(1))
            self.dwf.FDwfAnalogInConfigure(self.hdwf1, c_bool(False), c_bool(True))
            
            print("✅ All devices initialized successfully")
            return True
            
        except Exception as e:
            print(f"❌ Device initialization failed: {e}")
            return False
    
    def rgb_to_voltage(self, rgb, noise_level=0.0):
        """Convert RGB values to voltages with optional noise"""
        r, g, b = rgb
        
        # Add noise if specified
        if noise_level > 0:
            noise_r = random.uniform(-noise_level, noise_level) * 255
            noise_g = random.uniform(-noise_level, noise_level) * 255  
            noise_b = random.uniform(-noise_level, noise_level) * 255
            
            r = max(0, min(255, r + noise_r))
            g = max(0, min(255, g + noise_g))
            b = max(0, min(255, b + noise_b))
        
        # Get voltage range from GUI controls
        min_voltage = self.min_voltage_spin.value()
        max_voltage = self.max_voltage_spin.value()
        voltage_range = max_voltage - min_voltage
        
        # Scale to configurable voltage range with optimal offset for mycelium
        # Based on NARMA-10 findings: 2V-10V is optimal range for mycelium
        # Use 0.5V baseline to stay within 5V hardware limit
        baseline_voltage = 0.5  # Minimum voltage for each channel (hardware limit consideration)
        max_channel_voltage = min(5.0, min_voltage + voltage_range)  # Ensure we don't exceed 5V
        effective_range = max_channel_voltage - (min_voltage + baseline_voltage)
        
        voltage_r = (min_voltage + baseline_voltage) + (r / 255.0) * effective_range
        voltage_g = (min_voltage + baseline_voltage) + (g / 255.0) * effective_range
        voltage_b = (min_voltage + baseline_voltage) + (b / 255.0) * effective_range
        
        return voltage_r, voltage_g, voltage_b
    
    def set_voltages(self, voltage_r, voltage_g, voltage_b):
        """Set voltages on the three pins (same as NARMA-10 setup)"""
        if not (self.dwf and self.hdwf1.value != 0 and self.hdwf2.value != 0):
            raise RuntimeError("Devices not properly initialized")
        
        # Set voltages: hdwf1 channels 0&1, hdwf2 channel 0
        self.dwf.FDwfAnalogOutNodeOffsetSet(self.hdwf1, c_int(0), c_int(0), c_double(voltage_r))
        self.dwf.FDwfAnalogOutNodeOffsetSet(self.hdwf1, c_int(1), c_int(0), c_double(voltage_g)) 
        self.dwf.FDwfAnalogOutNodeOffsetSet(self.hdwf2, c_int(0), c_int(0), c_double(voltage_b))
    
    def read_mycelium_state(self):
        """Read mycelium state voltage"""
        if not (self.dwf and self.hdwf1.value != 0):
            raise RuntimeError("Device not properly initialized")
        
        # Start acquisition
        self.dwf.FDwfAnalogInConfigure(self.hdwf1, c_bool(False), c_bool(True))
        
        # Wait for acquisition
        time.sleep(0.01)
        
        # Read data
        sts = c_int()
        while True:
            self.dwf.FDwfAnalogInStatus(self.hdwf1, c_int(1), byref(sts))
            if sts.value == 2:  # Done
                break
            time.sleep(0.001)
        
        # Get voltage reading
        voltage = c_double()
        self.dwf.FDwfAnalogInStatusSample(self.hdwf1, c_int(0), byref(voltage))
        return voltage.value
    
    def update_mode(self, mode_text):
        """Update presentation mode and suggest optimal voltage range"""
        self.current_mode = "single" if "Single" in mode_text else "sequential"
        
        # Suggest mode-appropriate voltage ranges based on memory test findings
        if self.current_mode == "single":
            # Single mode: could benefit from higher voltage for max input coupling
            if self.max_voltage_spin.value() == 3.5:  # Only auto-adjust if at default
                self.max_voltage_spin.setValue(4.0)  # Higher for max coupling (4-5V range from memory tests)
        else:
            # Sequential mode: use NARMA-10 proven optimal range
            if self.max_voltage_spin.value() == 4.0:  # Only auto-adjust if at single-mode setting
                self.max_voltage_spin.setValue(3.5)  # NARMA-10 optimal for temporal memory
    
    def start_collection(self):
        """Start data collection process"""
        # Update parameters from UI
        self.total_trials_per_color = self.trials_spin.value()
        self.noise_level = self.noise_spin.value() / 100.0  # Convert from percentage to decimal
        self.memory_window = self.memory_window_spin.value()  # Update memory window
        self.sequence_length = self.sequence_length_spin.value()  # Update number of samples in sequence
        self.reset_delay = self.reset_delay_spin.value()  # Update reset delay from GUI
        self.single_duration = self.single_duration_spin.value()  # Update single stimulation duration
        
        # Calculate sample interval for sequential mode (derived from memory window)
        self.sample_interval = self.memory_window / self.sequence_length  # seconds per sample
        
        # Calculate estimated time
        total_samples = len(self.colors) * self.total_trials_per_color
        
        if self.current_mode == "single":
            # Single mode: configurable single duration + reset delay
            time_per_sample = (self.single_duration / 1000.0) + (self.reset_delay / 1000.0)  # Single duration + reset
        else:
            # Sequential mode: memory window + reset delay  
            time_per_sample = self.memory_window + (self.reset_delay / 1000.0)  # Sequential mode timing
        
        estimated_total_time = total_samples * time_per_sample
        estimated_minutes = int(estimated_total_time // 60)
        estimated_seconds = int(estimated_total_time % 60)
        
        # Show time estimate to user
        if estimated_minutes > 0:
            time_str = f"{estimated_minutes}m {estimated_seconds}s"
        else:
            time_str = f"{estimated_seconds}s"
        
        self.status_label.setText(f"Starting collection - Estimated time: {time_str}")
        
        # Initialize collection state
        self.is_collecting = True
        self.current_color_index = 0
        self.current_trial = 0
        
        # Setup progress bar
        self.progress_bar.setMaximum(total_samples)
        self.progress_bar.setValue(0)
        
        # Disable controls
        self.collect_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        
        # Start collection timer
        self.collection_timer = QTimer()
        self.collection_timer.timeout.connect(self.collect_next_sample)
        self.collection_timer.start(100)  # Start immediately
    
    def collect_next_sample(self):
        """Collect next sample in the sequence"""
        if not self.is_collecting:
            return
        
        color_names = list(self.colors.keys())
        
        if self.current_color_index >= len(color_names):
            # Collection complete
            self.stop_collection()
            self.status_label.setText("Collection completed!")
            return
        
        current_color_name = color_names[self.current_color_index]
        current_color_rgb = self.colors[current_color_name]
        
        # Update UI
        self.current_color_label.setText(
            f"Collecting {current_color_name} - Trial {self.current_trial + 1}/{self.total_trials_per_color} "
            f"({self.current_mode} mode)"
        )
        
        try:
            if self.current_mode == "single":
                # Single sample presentation
                voltage_r, voltage_g, voltage_b = self.rgb_to_voltage(current_color_rgb, self.noise_level)
                self.set_voltages(voltage_r, voltage_g, voltage_b)
                time.sleep(self.single_duration / 1000.0)  # Use configurable single duration
                final_state = self.read_mycelium_state()
                
                # Store data
                self.single_data['states'].append(final_state)
                self.single_data['labels'].append(current_color_name)
                self.single_data['noise_levels'].append(self.noise_level)
                
                # Schedule next sample after reset delay
                self.collection_timer.setInterval(self.reset_delay)
                
            else:
                # Sequential presentation (configurable samples spread across memory window)
                states = []
                
                # Get variation mode
                use_narma_progression = "NARMA-10" in self.variation_combo.currentText()
                
                for seq_sample in range(self.sequence_length):
                    if use_narma_progression:
                        # NARMA-10 style: progressive noise from 0% to max_noise%
                        # Sample 1: (1/n) * max_noise, Sample 2: (2/n) * max_noise, etc.
                        progressive_noise = ((seq_sample + 1) / self.sequence_length) * self.noise_level
                        # Use full intensity RGB values
                        scaled_rgb = current_color_rgb
                    else:
                        # Progressive intensity: gradually increase stimulus strength from 0% to 100%
                        # Sample 1: (1/n) * full_intensity, Sample 2: (2/n) * full_intensity, etc.
                        intensity_factor = (seq_sample + 1) / self.sequence_length
                        scaled_rgb = tuple(int(c * intensity_factor) for c in current_color_rgb)
                        progressive_noise = self.noise_level  # Use base noise level
                    
                    voltage_r, voltage_g, voltage_b = self.rgb_to_voltage(scaled_rgb, progressive_noise)
                    self.set_voltages(voltage_r, voltage_g, voltage_b)
                    
                    # Use memory window-based timing (memory_window / num_samples)
                    time.sleep(self.sample_interval)  # seconds per sample
                    
                    state = self.read_mycelium_state()
                    states.append(state)
                
                # Use final state as representative (like NARMA-10 uses current state)
                final_state = states[-1]
                
                # Store data
                self.sequential_data['states'].append(final_state)
                self.sequential_data['labels'].append(current_color_name)
                self.sequential_data['noise_levels'].append(self.noise_level)
                
                # Schedule next sample after reset delay
                self.collection_timer.setInterval(self.reset_delay)
            
            # Update progress
            progress_value = self.current_color_index * self.total_trials_per_color + self.current_trial + 1
            self.progress_bar.setValue(progress_value)
            
            # Move to next trial/color
            self.current_trial += 1
            if self.current_trial >= self.total_trials_per_color:
                self.current_trial = 0
                self.current_color_index += 1
            
            # Update data summary
            self.update_data_summary()
            
        except Exception as e:
            self.status_label.setText(f"Collection error: {e}")
            self.stop_collection()
    
    def stop_collection(self):
        """Stop data collection"""
        self.is_collecting = False
        if hasattr(self, 'collection_timer'):
            self.collection_timer.stop()
        
        # Re-enable controls
        self.collect_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        
        # Reset voltages to neutral
        self.set_voltages(2.5, 2.5, 2.5)
    
    def reset_data(self):
        """Reset all collected data"""
        self.single_data = {'states': [], 'labels': [], 'noise_levels': []}
        self.sequential_data = {'states': [], 'labels': [], 'noise_levels': []}
        self.single_model = None
        self.sequential_model = None
        self.update_data_summary()
        self.results_text.clear()
    
    def update_data_summary(self):
        """Update data collection summary"""
        summary = []
        summary.append(f"Single Sample Data: {len(self.single_data['states'])} samples")
        summary.append(f"Sequential Data: {len(self.sequential_data['states'])} samples")
        
        if len(self.single_data['states']) > 0:
            single_colors = set(self.single_data['labels'])
            summary.append(f"Single Sample Colors: {len(single_colors)}")
        
        if len(self.sequential_data['states']) > 0:
            sequential_colors = set(self.sequential_data['labels'])
            summary.append(f"Sequential Colors: {len(sequential_colors)}")
        
        self.data_summary.setText('\n'.join(summary))
    
    def train_single_model(self):
        """Train classifier on single sample data"""
        if len(self.single_data['states']) < 10:
            self.results_text.append("Not enough single sample data for training!")
            return
        
        self.results_text.append("Training single sample model... (estimated <10 seconds)")
        QApplication.processEvents()  # Update UI immediately
        
        try:
            # Get raw states
            X_raw = np.array(self.single_data['states']).reshape(-1, 1)
            y = np.array(self.single_data['labels'])
            
            # Create features with nonlinear transformations (same as NARMA-10)
            X = np.column_stack([
                X_raw.reshape(-1, 1),                    # Original state
                X_raw.reshape(-1, 1)**2,                 # Squared
                np.sin(X_raw.reshape(-1, 1) * 3),        # Sine transform
                np.cos(X_raw.reshape(-1, 1) * 2)         # Cosine transform
            ])
            
            # 70/30 train/test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42, stratify=y
            )
            
            self.single_model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.single_model.fit(X_train, y_train)
            
            # Calculate both training and test accuracy
            train_accuracy = self.single_model.score(X_train, y_train)
            test_accuracy = self.single_model.score(X_test, y_test)
            
            # Store test data for comparison
            self.single_test_data = (X_test, y_test)
            
            self.results_text.append(f"\nSingle Sample Model Trained!")
            self.results_text.append(f"Training Accuracy: {train_accuracy:.3f}")
            self.results_text.append(f"Test Accuracy: {test_accuracy:.3f}")
            self.results_text.append(f"Training Samples: {len(X_train)}")
            self.results_text.append(f"Test Samples: {len(X_test)}")
            
        except Exception as e:
            self.results_text.append(f"Single model training failed: {e}")
    
    def train_sequential_model(self):
        """Train classifier on sequential data"""
        if len(self.sequential_data['states']) < 10:
            self.results_text.append("Not enough sequential data for training!")
            return
        
        self.results_text.append("Training sequential model... (estimated <10 seconds)")
        QApplication.processEvents()  # Update UI immediately
        
        try:
            # Get raw states
            X_raw = np.array(self.sequential_data['states']).reshape(-1, 1)
            y = np.array(self.sequential_data['labels'])
            
            # Create features with nonlinear transformations (same as NARMA-10)
            X = np.column_stack([
                X_raw.reshape(-1, 1),                    # Original state
                X_raw.reshape(-1, 1)**2,                 # Squared
                np.sin(X_raw.reshape(-1, 1) * 3),        # Sine transform
                np.cos(X_raw.reshape(-1, 1) * 2)         # Cosine transform
            ])
            
            # 70/30 train/test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42, stratify=y
            )
            
            self.sequential_model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.sequential_model.fit(X_train, y_train)
            
            # Calculate both training and test accuracy
            train_accuracy = self.sequential_model.score(X_train, y_train)
            test_accuracy = self.sequential_model.score(X_test, y_test)
            
            # Store test data for comparison
            self.sequential_test_data = (X_test, y_test)
            
            self.results_text.append(f"\nSequential Model Trained!")
            self.results_text.append(f"Training Accuracy: {train_accuracy:.3f}")
            self.results_text.append(f"Test Accuracy: {test_accuracy:.3f}")
            self.results_text.append(f"Training Samples: {len(X_train)}")
            self.results_text.append(f"Test Samples: {len(X_test)}")
            
        except Exception as e:
            self.results_text.append(f"Sequential model training failed: {e}")
    
    def compare_models(self):
        """Compare performance of both models"""
        if self.single_model is None or self.sequential_model is None:
            self.results_text.append("Both models must be trained before comparison!")
            return
        
        if not hasattr(self, 'single_test_data') or not hasattr(self, 'sequential_test_data'):
            self.results_text.append("Test data not available - retrain models!")
            return
        
        try:
            # Get test data
            X_single_test, y_single_test = self.single_test_data
            X_sequential_test, y_sequential_test = self.sequential_test_data
            
            # Calculate test accuracies
            single_test_accuracy = self.single_model.score(X_single_test, y_single_test)
            sequential_test_accuracy = self.sequential_model.score(X_sequential_test, y_sequential_test)
            
            # Get predictions for confusion matrices
            single_pred = self.single_model.predict(X_single_test)
            sequential_pred = self.sequential_model.predict(X_sequential_test)
            
            # Calculate cluster quality if enough samples
            if len(set(y_single_test)) > 1 and len(X_single_test) > len(set(y_single_test)):
                single_silhouette = silhouette_score(X_single_test, y_single_test)
                sequential_silhouette = silhouette_score(X_sequential_test, y_sequential_test)
            else:
                single_silhouette = 0
                sequential_silhouette = 0
            
            # Display comparison results
            self.results_text.append("\n" + "="*50)
            self.results_text.append("MODEL COMPARISON RESULTS (Test Accuracy)")
            self.results_text.append("="*50)
            
            self.results_text.append(f"\nTEST ACCURACY COMPARISON:")
            self.results_text.append(f"Single Sample:  {single_test_accuracy:.3f}")
            self.results_text.append(f"Sequential:     {sequential_test_accuracy:.3f}")
            self.results_text.append(f"Temporal Benefit: {sequential_test_accuracy - single_test_accuracy:.3f}")
            
            if single_silhouette > 0:
                self.results_text.append(f"\nCLUSTER QUALITY (Silhouette Score):")
                self.results_text.append(f"Single Sample:  {single_silhouette:.3f}")
                self.results_text.append(f"Sequential:     {sequential_silhouette:.3f}")
                self.results_text.append(f"Clustering Improvement: {sequential_silhouette - single_silhouette:.3f}")
            
            # Confusion matrix analysis
            single_cm = confusion_matrix(y_single_test, single_pred)
            sequential_cm = confusion_matrix(y_sequential_test, sequential_pred)
            
            self.results_text.append(f"\nCONFUSION MATRIX ANALYSIS:")
            self.results_text.append("(Focus on similar color pairs)")
            
            # Check for similar color confusions
            similar_pairs = [('RED', 'ORANGE'), ('BLUE', 'INDIGO'), ('YELLOW', 'GREEN')]
            for color1, color2 in similar_pairs:
                if color1 in y_single_test and color2 in y_single_test:
                    # Find confusion between these colors
                    labels_single = list(set(y_single_test))
                    if color1 in labels_single and color2 in labels_single:
                        idx1_s = labels_single.index(color1)
                        idx2_s = labels_single.index(color2)
                        single_confusion = single_cm[idx1_s][idx2_s] + single_cm[idx2_s][idx1_s]
                        
                        labels_seq = list(set(y_sequential_test))
                        if color1 in labels_seq and color2 in labels_seq:
                            idx1_seq = labels_seq.index(color1)
                            idx2_seq = labels_seq.index(color2)
                            seq_confusion = sequential_cm[idx1_seq][idx2_seq] + sequential_cm[idx2_seq][idx1_seq]
                            
                            self.results_text.append(f"{color1}↔{color2}: Single={single_confusion}, Sequential={seq_confusion}")
            
            # Overall conclusion
            self.results_text.append(f"\nCONCLUSION:")
            if sequential_test_accuracy > single_test_accuracy + 0.05:
                self.results_text.append("✅ Sequential presentation provides significant benefit!")
                self.results_text.append("✅ Temporal memory enhances color classification!")
            elif sequential_test_accuracy > single_test_accuracy:
                self.results_text.append("✅ Sequential presentation shows modest improvement")
            else:
                self.results_text.append("❌ No clear temporal benefit detected")
                self.results_text.append("Consider adjusting timing or collecting more data")
            
        except Exception as e:
            self.results_text.append(f"Comparison failed: {e}")
    
    def generate_tsne_single(self):
        """Generate t-SNE plot for single sample data"""
        if len(self.single_data['states']) < 10:
            return
        
        try:
            X = np.array(self.single_data['states']).reshape(-1, 1)
            y = self.single_data['labels']
            
            # Create color map for visualization
            unique_colors = list(set(y))
            color_map = plt.cm.Set1(np.linspace(0, 1, len(unique_colors)))
            
            if X.shape[1] == 1:
                # For 1D data, create 2D t-SNE by adding noise dimension
                X_tsne = np.column_stack([X.flatten(), np.random.normal(0, 0.1, len(X))])
            else:
                tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X)//4))
                X_tsne = tsne.fit_transform(X)
            
            # Clear figure and plot
            self.tsne_figure.clear()
            ax = self.tsne_figure.add_subplot(121)
            
            for i, color in enumerate(unique_colors):
                mask = np.array(y) == color
                ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1], 
                          c=[color_map[i]], label=color, alpha=0.7, s=50)
            
            ax.set_title('Single Sample t-SNE')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            self.tsne_canvas.draw()
            
        except Exception as e:
            print(f"Single t-SNE generation failed: {e}")
    
    def generate_tsne_sequential(self):
        """Generate t-SNE plot for sequential data"""
        if len(self.sequential_data['states']) < 10:
            return
        
        try:
            X = np.array(self.sequential_data['states']).reshape(-1, 1)
            y = self.sequential_data['labels']
            
            # Create color map for visualization
            unique_colors = list(set(y))
            color_map = plt.cm.Set1(np.linspace(0, 1, len(unique_colors)))
            
            if X.shape[1] == 1:
                # For 1D data, create 2D t-SNE by adding noise dimension  
                X_tsne = np.column_stack([X.flatten(), np.random.normal(0, 0.1, len(X))])
            else:
                tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X)//4))
                X_tsne = tsne.fit_transform(X)
            
            # Clear figure and plot
            self.tsne_figure.clear()
            ax = self.tsne_figure.add_subplot(122)
            
            for i, color in enumerate(unique_colors):
                mask = np.array(y) == color
                ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1],
                          c=[color_map[i]], label=color, alpha=0.7, s=50)
            
            ax.set_title('Sequential t-SNE')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            self.tsne_canvas.draw()
            
        except Exception as e:
            print(f"Sequential t-SNE generation failed: {e}")
    
    def generate_tsne_comparison(self):
        """Generate side-by-side t-SNE comparison"""
        if len(self.single_data['states']) < 10 or len(self.sequential_data['states']) < 10:
            return
        
        try:
            # Prepare single sample data
            X_single = np.array(self.single_data['states']).reshape(-1, 1)
            y_single = self.single_data['labels']
            
            # Prepare sequential data
            X_sequential = np.array(self.sequential_data['states']).reshape(-1, 1)
            y_sequential = self.sequential_data['labels']
            
            # Create unified color map
            all_colors = list(set(y_single + y_sequential))
            color_map = plt.cm.Set1(np.linspace(0, 1, len(all_colors)))
            color_dict = {color: color_map[i] for i, color in enumerate(all_colors)}
            
            # Clear figure
            self.tsne_figure.clear()
            
            # Single sample plot
            ax1 = self.tsne_figure.add_subplot(121)
            if X_single.shape[1] == 1:
                X_single_plot = np.column_stack([X_single.flatten(), np.random.normal(0, 0.1, len(X_single))])
            else:
                tsne_single = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X_single)//4))
                X_single_plot = tsne_single.fit_transform(X_single)
            
            for color in all_colors:
                if color in y_single:
                    mask = np.array(y_single) == color
                    ax1.scatter(X_single_plot[mask, 0], X_single_plot[mask, 1],
                              c=[color_dict[color]], label=color, alpha=0.7, s=50)
            
            ax1.set_title('Single Sample t-SNE')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Sequential plot
            ax2 = self.tsne_figure.add_subplot(122)
            if X_sequential.shape[1] == 1:
                X_sequential_plot = np.column_stack([X_sequential.flatten(), np.random.normal(0, 0.1, len(X_sequential))])
            else:
                tsne_sequential = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X_sequential)//4))
                X_sequential_plot = tsne_sequential.fit_transform(X_sequential)
            
            for color in all_colors:
                if color in y_sequential:
                    mask = np.array(y_sequential) == color
                    ax2.scatter(X_sequential_plot[mask, 0], X_sequential_plot[mask, 1],
                              c=[color_dict[color]], label=color, alpha=0.7, s=50)
            
            ax2.set_title('Sequential t-SNE')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            self.tsne_figure.tight_layout()
            self.tsne_canvas.draw()
            
        except Exception as e:
            print(f"Comparison t-SNE generation failed: {e}")
    
    def save_training_data_csv(self):
        """Save both single and sequential training data to CSV files"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Save single sample data
            if len(self.single_data['states']) > 0:
                single_filename = f"single_sample_data_{timestamp}.csv"
                with open(single_filename, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(['Trial', 'Color', 'State_Voltage', 'Noise_Level', 'Voltage_Setting', 'Single_Duration_ms', 'Reset_Delay_ms'])
                    
                    for i, (state, label, noise) in enumerate(zip(
                        self.single_data['states'],
                        self.single_data['labels'], 
                        self.single_data['noise_levels']
                    )):
                        writer.writerow([i+1, label, state, noise, f"{self.min_voltage_spin.value()}-{self.max_voltage_spin.value()}V", 
                                       self.single_duration, self.reset_delay])
                
                self.results_text.append(f"Single sample data saved: {single_filename}")
            
            # Save sequential data
            if len(self.sequential_data['states']) > 0:
                sequential_filename = f"sequential_data_{timestamp}.csv"
                with open(sequential_filename, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(['Trial', 'Color', 'Final_State_Voltage', 'Noise_Level', 
                                   'Voltage_Setting', 'Sequence_Length', 'Memory_Window_s', 'Sample_Interval_ms', 'Reset_Delay_ms', 'Variation_Type'])
                    
                    for i, (state, label, noise) in enumerate(zip(
                        self.sequential_data['states'],
                        self.sequential_data['labels'],
                        self.sequential_data['noise_levels']
                    )):
                        variation_type = self.variation_combo.currentText()
                        writer.writerow([i+1, label, state, noise, f"{self.min_voltage_spin.value()}-{self.max_voltage_spin.value()}V", 
                                       self.sequence_length, self.memory_window, self.sample_interval * 1000, self.reset_delay, variation_type])
                
                self.results_text.append(f"Sequential data saved: {sequential_filename}")
            
            # Save combined summary
            summary_filename = f"experiment_summary_{timestamp}.csv"
            with open(summary_filename, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['Parameter', 'Value'])
                writer.writerow(['Voltage_Range', f"{self.min_voltage_spin.value()}-{self.max_voltage_spin.value()}V"])
                writer.writerow(['Memory_Window_s', self.memory_window])
                writer.writerow(['Single_Duration_ms', self.single_duration])
                writer.writerow(['Sample_Interval_ms', self.sample_interval * 1000])
                writer.writerow(['Reset_Delay_ms', self.reset_delay])
                writer.writerow(['Sequence_Length', self.sequence_length])
                writer.writerow(['Trials_Per_Color', self.total_trials_per_color])
                writer.writerow(['Noise_Level', self.noise_level])
                writer.writerow(['Total_Colors', len(self.colors)])
                writer.writerow(['Single_Samples', len(self.single_data['states'])])
                writer.writerow(['Sequential_Samples', len(self.sequential_data['states'])])
                writer.writerow(['Sequence_Length', self.sequence_length])
                writer.writerow(['Sequential_Variation', self.variation_combo.currentText()])
                
                if self.single_model and self.sequential_model:
                    X_single = np.array(self.single_data['states']).reshape(-1, 1)
                    y_single = np.array(self.single_data['labels'])
                    X_sequential = np.array(self.sequential_data['states']).reshape(-1, 1)
                    y_sequential = np.array(self.sequential_data['labels'])
                    
                    single_accuracy = self.single_model.score(X_single, y_single)
                    sequential_accuracy = self.sequential_model.score(X_sequential, y_sequential)
                    
                    writer.writerow(['Single_Model_Accuracy', single_accuracy])
                    writer.writerow(['Sequential_Model_Accuracy', sequential_accuracy])
                    writer.writerow(['Temporal_Benefit', sequential_accuracy - single_accuracy])
            
            self.results_text.append(f"Experiment summary saved: {summary_filename}")
            
        except Exception as e:
            self.results_text.append(f"Error saving training data: {e}")
    
    def save_analysis_pdf(self):
        """Save comprehensive analysis report with visuals as PDF"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"color_classification_analysis_{timestamp}.pdf"
            
            with PdfPages(filename) as pdf:
                # Title Page
                fig, ax = plt.subplots(figsize=(8.5, 11))
                ax.axis('off')
                
                ax.text(0.5, 0.9, 'Mycelium Color Classification Analysis', 
                        ha='center', va='center', fontsize=18, fontweight='bold')
                ax.text(0.5, 0.85, 'Single vs Sequential Presentation Comparison', 
                        ha='center', va='center', fontsize=14, style='italic')
                ax.text(0.5, 0.8, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 
                        ha='center', va='center', fontsize=12)
                
                # Experiment parameters
                param_text = f"""
EXPERIMENT PARAMETERS:

Hardware Configuration:
• Voltage Range: {self.min_voltage_spin.value()}-{self.max_voltage_spin.value()}V
• Memory Window: {self.memory_window}s (sequential mode)
• Single Duration: {self.single_duration}ms (single mode)
• Sample Interval: {self.sample_interval * 1000:.1f}ms (memory_window / num_samples)
• Reset Delay: {self.reset_delay}ms (between colors)
• Sequence Length: {self.sequence_length} samples
• Sequential Variation: {self.variation_combo.currentText()}

Data Collection:
• Trials per Color: {self.total_trials_per_color}
• Noise Level: {self.noise_level:.1%}
• Total Colors: {len(self.colors)}
• Colors: {', '.join(self.colors.keys())}

Sample Counts:
• Single Sample Data: {len(self.single_data['states'])} samples
• Sequential Data: {len(self.sequential_data['states'])} samples
                """
                
                ax.text(0.1, 0.65, param_text, ha='left', va='top', fontsize=11,
                        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.7))
                
                pdf.savefig(fig, bbox_inches='tight')
                plt.close()
                
                # Model Performance Page
                if self.single_model and self.sequential_model:
                    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(11, 8.5))
                    fig.suptitle('Model Performance Analysis', fontsize=16, fontweight='bold')
                    
                    # Prepare data
                    X_single = np.array(self.single_data['states']).reshape(-1, 1)
                    y_single = np.array(self.single_data['labels'])
                    X_sequential = np.array(self.sequential_data['states']).reshape(-1, 1)
                    y_sequential = np.array(self.sequential_data['labels'])
                    
                    single_accuracy = self.single_model.score(X_single, y_single)
                    sequential_accuracy = self.sequential_model.score(X_sequential, y_sequential)
                    temporal_benefit = sequential_accuracy - single_accuracy
                    
                    # Accuracy comparison
                    categories = ['Single Sample', 'Sequential']
                    accuracies = [single_accuracy, sequential_accuracy]
                    colors = ['lightcoral', 'lightgreen']
                    
                    bars = ax1.bar(categories, accuracies, color=colors, alpha=0.7)
                    ax1.set_ylabel('Accuracy')
                    ax1.set_title('Classification Accuracy Comparison')
                    ax1.set_ylim(0, 1)
                    
                    for bar, acc in zip(bars, accuracies):
                        height = bar.get_height()
                        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                                f'{acc:.3f}', ha='center', va='bottom')
                    
                    # Temporal benefit visualization
                    ax2.bar(['Temporal Benefit'], [temporal_benefit], 
                           color='gold' if temporal_benefit > 0 else 'lightcoral', alpha=0.7)
                    ax2.set_ylabel('Accuracy Improvement')
                    ax2.set_title('Sequential vs Single Benefit')
                    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                    ax2.text(0, temporal_benefit + 0.01, f'{temporal_benefit:.3f}', 
                            ha='center', va='bottom')
                    
                    # State distribution comparison
                    ax3.hist(self.single_data['states'], bins=20, alpha=0.6, 
                            label='Single', color='lightcoral', density=True)
                    ax3.hist(self.sequential_data['states'], bins=20, alpha=0.6, 
                            label='Sequential', color='lightgreen', density=True)
                    ax3.set_xlabel('State Voltage (V)')
                    ax3.set_ylabel('Density')
                    ax3.set_title('State Distribution Comparison')
                    ax3.legend()
                    ax3.grid(True, alpha=0.3)
                    
                    # Results summary
                    ax4.axis('off')
                    results_text = f"""
RESULTS SUMMARY:

Accuracy Metrics:
• Single Sample: {single_accuracy:.3f}
• Sequential: {sequential_accuracy:.3f}
• Temporal Benefit: {temporal_benefit:.3f}

Performance Assessment:
{"✅ Significant temporal benefit!" if temporal_benefit > 0.05 
 else "✅ Modest improvement" if temporal_benefit > 0 
 else "❌ No temporal benefit"}

Data Quality:
• Single samples: {len(self.single_data['states'])}
• Sequential samples: {len(self.sequential_data['states'])}
• Colors tested: {len(set(self.single_data['labels']))}

Conclusion:
{"Sequential presentation enhances classification through temporal memory effects." if temporal_benefit > 0.02
 else "Limited evidence of temporal memory benefits."}
                    """
                    
                    ax4.text(0.05, 0.95, results_text, ha='left', va='top', fontsize=11,
                            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.7))
                    
                    pdf.savefig(fig, bbox_inches='tight')
                    plt.close()
                
                # t-SNE Visualization Page
                if len(self.single_data['states']) >= 10 and len(self.sequential_data['states']) >= 10:
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 8.5))
                    fig.suptitle('t-SNE Cluster Analysis', fontsize=16, fontweight='bold')
                    
                    # Single sample t-SNE
                    X_single = np.array(self.single_data['states']).reshape(-1, 1)
                    y_single = self.single_data['labels']
                    
                    unique_colors = list(set(y_single))
                    color_map = plt.cm.Set1(np.linspace(0, 1, len(unique_colors)))
                    
                    X_single_plot = np.column_stack([X_single.flatten(), 
                                                   np.random.normal(0, 0.1, len(X_single))])
                    
                    for i, color in enumerate(unique_colors):
                        mask = np.array(y_single) == color
                        ax1.scatter(X_single_plot[mask, 0], X_single_plot[mask, 1],
                                  c=[color_map[i]], label=color, alpha=0.7, s=50)
                    
                    ax1.set_title('Single Sample Clustering')
                    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                    ax1.grid(True, alpha=0.3)
                    
                    # Sequential t-SNE
                    X_sequential = np.array(self.sequential_data['states']).reshape(-1, 1)
                    y_sequential = self.sequential_data['labels']
                    
                    X_sequential_plot = np.column_stack([X_sequential.flatten(),
                                                       np.random.normal(0, 0.1, len(X_sequential))])
                    
                    for i, color in enumerate(unique_colors):
                        if color in y_sequential:
                            mask = np.array(y_sequential) == color
                            ax2.scatter(X_sequential_plot[mask, 0], X_sequential_plot[mask, 1],
                                      c=[color_map[i]], label=color, alpha=0.7, s=50)
                    
                    ax2.set_title('Sequential Clustering')
                    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                    ax2.grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    pdf.savefig(fig, bbox_inches='tight')
                    plt.close()
                
                # Data Distribution Page
                if len(self.single_data['states']) > 0 and len(self.sequential_data['states']) > 0:
                    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(11, 8.5))
                    fig.suptitle('Data Distribution Analysis', fontsize=16, fontweight='bold')
                    
                    # Color distribution for single samples
                    single_color_counts = pd.Series(self.single_data['labels']).value_counts()
                    single_color_counts.plot(kind='bar', ax=ax1, color='lightcoral', alpha=0.7)
                    ax1.set_title('Single Sample Color Distribution')
                    ax1.set_ylabel('Count')
                    ax1.tick_params(axis='x', rotation=45)
                    
                    # Color distribution for sequential samples
                    sequential_color_counts = pd.Series(self.sequential_data['labels']).value_counts()
                    sequential_color_counts.plot(kind='bar', ax=ax2, color='lightgreen', alpha=0.7)
                    ax2.set_title('Sequential Sample Color Distribution')
                    ax2.set_ylabel('Count')
                    ax2.tick_params(axis='x', rotation=45)
                    
                    # Voltage range analysis
                    ax3.boxplot([self.single_data['states'], self.sequential_data['states']], 
                               tick_labels=['Single', 'Sequential'])
                    ax3.set_title('State Voltage Distribution')
                    ax3.set_ylabel('Voltage (V)')
                    ax3.grid(True, alpha=0.3)
                    
                    # Timeline of collection
                    ax4.plot(range(len(self.single_data['states'])), self.single_data['states'], 
                            'o-', alpha=0.6, label='Single', markersize=3)
                    ax4.plot(range(len(self.sequential_data['states'])), self.sequential_data['states'], 
                            'o-', alpha=0.6, label='Sequential', markersize=3)
                    ax4.set_title('Collection Timeline')
                    ax4.set_xlabel('Sample Number')
                    ax4.set_ylabel('State Voltage (V)')
                    ax4.legend()
                    ax4.grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    pdf.savefig(fig, bbox_inches='tight')
                    plt.close()
            
            self.results_text.append(f"\nAnalysis report saved: {filename}")
            
        except Exception as e:
            self.results_text.append(f"Error saving analysis PDF: {e}")
    
    def save_plots_png(self):
        """Save current t-SNE plots as PNG files"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Save the current figure
            if hasattr(self, 'tsne_figure'):
                plot_filename = f"tsne_plots_{timestamp}.png"
                self.tsne_figure.savefig(plot_filename, dpi=300, bbox_inches='tight')
                self.results_text.append(f"Plots saved: {plot_filename}")
            
        except Exception as e:
            self.results_text.append(f"Error saving plots: {e}")
    
    def closeEvent(self, event):
        """Clean up when closing"""
        self.stop_collection()
        
        # Close devices
        if self.dwf and hasattr(self, 'hdwf1') and hasattr(self, 'hdwf2'):
            if self.hdwf1.value != 0:
                self.dwf.FDwfDeviceClose(self.hdwf1)
            if self.hdwf2.value != 0:
                self.dwf.FDwfDeviceClose(self.hdwf2)
        
        event.accept()

    def collect_test_data(self, noise_levels, samples_per_level, mode):
        """Collect test data at specified noise levels"""
        test_data = {'states': [], 'labels': [], 'noise_levels': []}
        color_names = list(self.colors.keys())
        
        self.status_label.setText("Collecting evaluation data...")
        
        try:
            for noise_level in noise_levels:
                for color_name in color_names:
                    color_rgb = self.colors[color_name]
                    
                    for sample in range(samples_per_level):
                        self.status_label.setText(
                            f"Evaluating {color_name} at {noise_level:.1%} noise - {sample+1}/{samples_per_level}"
                        )
                        QApplication.processEvents()  # Update UI
                        
                        if mode == "single":
                            # Single sample presentation
                            voltage_r, voltage_g, voltage_b = self.rgb_to_voltage(color_rgb, noise_level)
                            self.set_voltages(voltage_r, voltage_g, voltage_b)
                            time.sleep(self.single_duration / 1000.0)  # Use configurable single duration
                            final_state = self.read_mycelium_state()
                            
                        else:
                            # Sequential presentation (configurable samples spread across memory window)
                            states = []
                            
                            # Get variation mode
                            use_narma_progression = "NARMA-10" in self.variation_combo.currentText()
                            
                            for seq_sample in range(self.sequence_length):
                                if use_narma_progression:
                                    # NARMA-10 style: progressive noise from 0% to max_noise%
                                    # Sample 1: (1/n) * max_noise, Sample 2: (2/n) * max_noise, etc.
                                    progressive_noise = ((seq_sample + 1) / self.sequence_length) * noise_level
                                    # Use full intensity RGB values
                                    scaled_rgb = color_rgb
                                else:
                                    # Progressive intensity: gradually increase stimulus strength from 0% to 100%
                                    # Sample 1: (1/n) * full_intensity, Sample 2: (2/n) * full_intensity, etc.
                                    intensity_factor = (seq_sample + 1) / self.sequence_length
                                    scaled_rgb = tuple(int(c * intensity_factor) for c in color_rgb)
                                    progressive_noise = noise_level  # Use base noise level
                                
                                voltage_r, voltage_g, voltage_b = self.rgb_to_voltage(scaled_rgb, progressive_noise)
                                self.set_voltages(voltage_r, voltage_g, voltage_b)
                                
                                # Use memory window-based timing (memory_window / num_samples)
                                time.sleep(self.sample_interval)  # seconds per sample
                                
                                state = self.read_mycelium_state()
                                states.append(state)
                            
                            # Use final state as representative (like NARMA-10 uses current state)
                            final_state = states[-1]
                        
                        test_data['states'].append(final_state)
                        test_data['labels'].append(color_name)
                        test_data['noise_levels'].append(noise_level)
                        
                        # Reset delay between colors
                        time.sleep(self.reset_delay / 1000.0)
            
            self.status_label.setText("Evaluation data collection completed")
            return test_data
            
        except Exception as e:
            self.status_label.setText(f"Evaluation error: {e}")
            return None
    
    def evaluate_single_model(self):
        """Evaluate single sample model against noise"""
        if self.single_model is None:
            self.eval_results_text.append("Single sample model not trained!")
            return
        
        if not hasattr(self, 'single_test_data'):
            self.eval_results_text.append("No test data available - retrain model!")
            return
        
        # Update parameters from GUI for evaluation
        self.memory_window = self.memory_window_spin.value()
        self.sequence_length = self.sequence_length_spin.value()
        self.reset_delay = self.reset_delay_spin.value()
        self.single_duration = self.single_duration_spin.value()
        self.sample_interval = self.memory_window / self.sequence_length
        
        # Get existing test data
        X_test, y_test = self.single_test_data
        
        # Determine evaluation mode
        eval_mode = self.eval_mode_combo.currentText()
        
        if "Progressive" in eval_mode:
            # For progressive noise, need to collect new data at different noise levels
            noise_levels = [0.0, 0.1, 0.2, 0.3]
            samples_per_level = self.test_samples_spin.value()
            
            self.eval_results_text.append("Collecting new data for progressive noise testing...")
            # Calculate estimated time for progressive noise testing
            total_eval_samples = len(noise_levels) * len(self.colors) * samples_per_level
            # Use single mode timing for single model evaluation
            time_per_sample = (self.single_duration / 1000.0) + (self.reset_delay / 1000.0)
            
            estimated_time = total_eval_samples * time_per_sample
            estimated_minutes = int(estimated_time // 60)
            if estimated_minutes > 0:
                time_str = f"~{estimated_minutes}m {int(estimated_time % 60)}s"
            else:
                time_str = f"~{int(estimated_time)}s"
            
            self.eval_results_text.append(f"Estimated time: {time_str}")
            QApplication.processEvents()  # Update UI
            
            test_data = self.collect_test_data(noise_levels, samples_per_level, "single")
            if test_data is None:
                return
            
            # Evaluate model on new data
            X_eval = np.array(test_data['states']).reshape(-1, 1)
            y_eval = np.array(test_data['labels'])
            noise_eval = np.array(test_data['noise_levels'])
            
            # Calculate accuracy for each noise level
            results = []
            for noise_level in noise_levels:
                mask = noise_eval == noise_level
                if np.sum(mask) > 0:
                    X_subset = X_eval[mask]
                    y_subset = y_eval[mask]
                    accuracy = self.single_model.score(X_subset, y_subset)
                    results.append((noise_level, accuracy))
            
            # Display results
            self.eval_results_text.append("\n=== SINGLE MODEL PROGRESSIVE NOISE EVALUATION ===")
            for noise_level, accuracy in results:
                self.eval_results_text.append(f"Noise {noise_level:.1%}: Accuracy {accuracy:.3f}")
                
        else:
            # Use existing test data (fast evaluation)
            test_accuracy = self.single_model.score(X_test, y_test)
            
            # Display results
            self.eval_results_text.append("\n=== SINGLE MODEL TEST EVALUATION ===")
            self.eval_results_text.append(f"Test Accuracy: {test_accuracy:.3f}")
            self.eval_results_text.append(f"Test Samples: {len(X_test)}")
            results = [(0.0, test_accuracy)]  # Return format for consistency
        
        return results
    
    def evaluate_sequential_model(self):
        """Evaluate sequential model against noise"""
        if self.sequential_model is None:
            self.eval_results_text.append("Sequential model not trained!")
            return
        
        if not hasattr(self, 'sequential_test_data'):
            self.eval_results_text.append("No test data available - retrain model!")
            return
        
        # Update parameters from GUI for evaluation
        self.memory_window = self.memory_window_spin.value()
        self.sequence_length = self.sequence_length_spin.value()
        self.reset_delay = self.reset_delay_spin.value()
        self.single_duration = self.single_duration_spin.value()
        self.sample_interval = self.memory_window / self.sequence_length
        
        # Get existing test data
        X_test, y_test = self.sequential_test_data
        
        # Determine evaluation mode
        eval_mode = self.eval_mode_combo.currentText()
        
        if "Progressive" in eval_mode:
            # For progressive noise, need to collect new data at different noise levels
            noise_levels = [0.0, 0.1, 0.2, 0.3]
            samples_per_level = self.test_samples_spin.value()
            
            self.eval_results_text.append("Collecting new data for progressive noise testing...")
            # Calculate estimated time for progressive noise testing
            total_eval_samples = len(noise_levels) * len(self.colors) * samples_per_level
            # Use sequential mode timing for sequential model evaluation
            time_per_sample = self.memory_window + (self.reset_delay / 1000.0)
            
            estimated_time = total_eval_samples * time_per_sample
            estimated_minutes = int(estimated_time // 60)
            if estimated_minutes > 0:
                time_str = f"~{estimated_minutes}m {int(estimated_time % 60)}s"
            else:
                time_str = f"~{int(estimated_time)}s"
            
            self.eval_results_text.append(f"Estimated time: {time_str}")
            QApplication.processEvents()  # Update UI
            
            test_data = self.collect_test_data(noise_levels, samples_per_level, "sequential")
            if test_data is None:
                return
            
            # Evaluate model on new data
            X_eval = np.array(test_data['states']).reshape(-1, 1)
            y_eval = np.array(test_data['labels'])
            noise_eval = np.array(test_data['noise_levels'])
            
            # Calculate accuracy for each noise level
            results = []
            for noise_level in noise_levels:
                mask = noise_eval == noise_level
                if np.sum(mask) > 0:
                    X_subset = X_eval[mask]
                    y_subset = y_eval[mask]
                    accuracy = self.sequential_model.score(X_subset, y_subset)
                    results.append((noise_level, accuracy))
            
            # Display results
            self.eval_results_text.append("\n=== SEQUENTIAL MODEL PROGRESSIVE NOISE EVALUATION ===")
            for noise_level, accuracy in results:
                self.eval_results_text.append(f"Noise {noise_level:.1%}: Accuracy {accuracy:.3f}")
                
        else:
            # Use existing test data (fast evaluation)
            test_accuracy = self.sequential_model.score(X_test, y_test)
            
            # Display results
            self.eval_results_text.append("\n=== SEQUENTIAL MODEL TEST EVALUATION ===")
            self.eval_results_text.append(f"Test Accuracy: {test_accuracy:.3f}")
            self.eval_results_text.append(f"Test Samples: {len(X_test)}")
            results = [(0.0, test_accuracy)]  # Return format for consistency
        
        return results
    
    def compare_noise_robustness(self):
        """Compare both models across noise levels"""
        if self.single_model is None or self.sequential_model is None:
            self.eval_results_text.append("Both models must be trained for comparison!")
            return
        
        if not hasattr(self, 'single_test_data') or not hasattr(self, 'sequential_test_data'):
            self.eval_results_text.append("No test data available - retrain models!")
            return
        
        # Update parameters from GUI for evaluation
        self.memory_window = self.memory_window_spin.value()
        self.sequence_length = self.sequence_length_spin.value()
        self.reset_delay = self.reset_delay_spin.value()
        self.single_duration = self.single_duration_spin.value()
        self.sample_interval = self.memory_window / self.sequence_length
        
        # Determine evaluation mode
        eval_mode = self.eval_mode_combo.currentText()
        
        if "Progressive" in eval_mode:
            # For progressive noise, need to collect new data
            noise_levels = [0.0, 0.1, 0.2, 0.3]
            samples_per_level = self.test_samples_spin.value()
            
            self.eval_results_text.append("Collecting new data for progressive noise comparison...")
            
            # Collect test data for both modes
            single_test_data = self.collect_test_data(noise_levels, samples_per_level, "single")
            sequential_test_data = self.collect_test_data(noise_levels, samples_per_level, "sequential")
            
            if single_test_data is None or sequential_test_data is None:
                return
            
            # Evaluate both models
            single_results = []
            sequential_results = []
            
            for noise_level in noise_levels:
                # Single model evaluation
                mask_single = np.array(single_test_data['noise_levels']) == noise_level
                X_single = np.array(single_test_data['states'])[mask_single].reshape(-1, 1)
                y_single = np.array(single_test_data['labels'])[mask_single]
                single_acc = self.single_model.score(X_single, y_single)
                single_results.append(single_acc)
                
                # Sequential model evaluation
                mask_seq = np.array(sequential_test_data['noise_levels']) == noise_level
                X_seq = np.array(sequential_test_data['states'])[mask_seq].reshape(-1, 1)
                y_seq = np.array(sequential_test_data['labels'])[mask_seq]
                seq_acc = self.sequential_model.score(X_seq, y_seq)
                sequential_results.append(seq_acc)
            
            # Plot comparison
            self.eval_figure.clear()
            ax = self.eval_figure.add_subplot(111)
            
            noise_percentages = [n * 100 for n in noise_levels]
            ax.plot(noise_percentages, single_results, 'o-', label='Single Sample', linewidth=2, markersize=8)
            ax.plot(noise_percentages, sequential_results, 's-', label='Sequential', linewidth=2, markersize=8)
            
            ax.set_xlabel('Noise Level (%)')
            ax.set_ylabel('Classification Accuracy')
            ax.set_title('Noise Robustness Comparison')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1)
            
            self.eval_canvas.draw()
            
        else:
            # Use existing test data (fast comparison)
            X_single_test, y_single_test = self.single_test_data
            X_sequential_test, y_sequential_test = self.sequential_test_data
            
            single_accuracy = self.single_model.score(X_single_test, y_single_test)
            sequential_accuracy = self.sequential_model.score(X_sequential_test, y_sequential_test)
            
            single_results = [single_accuracy]
            sequential_results = [sequential_accuracy]
            noise_levels = [self.noise_spin.value() / 100.0]  # Training noise level
        
        # Display comparison results
        self.eval_results_text.append("\n" + "="*60)
        self.eval_results_text.append("NOISE ROBUSTNESS COMPARISON")
        self.eval_results_text.append("="*60)
        
        self.eval_results_text.append(f"\n{'Noise Level':<12} {'Single':<10} {'Sequential':<12} {'Benefit':<10}")
        self.eval_results_text.append("-" * 50)
        
        for i, noise_level in enumerate(noise_levels):
            benefit = sequential_results[i] - single_results[i]
            noise_pct = f"{noise_level:.1%}"
            self.eval_results_text.append(
                f"{noise_pct:<12} {single_results[i]:<10.3f} "
                f"{sequential_results[i]:<12.3f} {benefit:+<10.3f}"
            )
        
        # Analysis
        avg_benefit = np.mean([sequential_results[i] - single_results[i] for i in range(len(noise_levels))])
        self.eval_results_text.append(f"\nAVERAGE TEMPORAL BENEFIT: {avg_benefit:+.3f}")
        
        if avg_benefit > 0.05:
            self.eval_results_text.append("✅ Sequential presentation provides robust benefits across noise levels!")
        elif avg_benefit > 0.02:
            self.eval_results_text.append("✅ Sequential presentation shows modest but consistent benefits")
        else:
            self.eval_results_text.append("❌ Limited temporal memory benefits under noise conditions")

def main():
    app = QApplication(sys.argv)
    window = ColorClassifierApp()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main() 