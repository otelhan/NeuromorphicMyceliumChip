#!/usr/bin/env python3
"""
Mycelium Temporal Memory Analysis Tool
Tests for memory effects in mycelium-based reservoir computing

This application provides various tests to analyze temporal memory:
1. Autocorrelation analysis
2. Step response test  
3. Cross-correlation analysis
4. State prediction test
5. Response decay analysis
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
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

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

class MyceliumMemoryApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Mycelium Temporal Memory Analysis")
        self.setMinimumSize(1000, 800)
        
        # Test parameters
        self.sample_rate = 1000  # Hz
        self.delay_ms = 300      # ms between samples
        self.test_duration = 60  # seconds
        self.input_amplitude = 4.0  # Direct voltage amplitude (1-5V)
        
        # Memory analysis parameters
        self.max_lag = 20
        self.step_duration = 10  # samples per step
        
        # Device handles - using same pattern as NARMA-10 app
        self.dwf = None
        self.hdwf1 = None
        self.hdwf2 = None
        
        # Data storage
        self.timestamps = []
        self.inputs = []
        self.states = []
        self.test_type = "none"
        
        # Analysis results
        self.autocorr_results = []
        self.cross_corr_results = []
        self.decay_results = []
        
        # UI state
        self.is_testing = False
        self.current_sample = 0
        
        # Setup UI and timers
        self.setup_ui()
        self.monitor_timer = QTimer()
        self.monitor_timer.timeout.connect(self.update_monitoring)
        
        # Initialize Digilent like NARMA-10 app
        device_init_success = self.setup_digilent()
        
        # Update UI based on device initialization result
        if device_init_success and self.dwf:
            self.status_label.setText("Devices initialized successfully")
            self.start_btn.setEnabled(True)
            self.connect_btn.setText("Reconnect Devices")
        else:
            self.status_label.setText("Device initialization failed - running in simulation mode")
            self.start_btn.setEnabled(True)  # Allow testing in simulation mode
            self.connect_btn.setText("Connect Devices")

    def setup_ui(self):
        """Create the user interface"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Create tabs
        tabs = QTabWidget()
        main_layout.addWidget(tabs)
        
        # Setup tab
        setup_tab = QWidget()
        setup_layout = QVBoxLayout(setup_tab)
        tabs.addTab(setup_tab, "Setup")
        
        # Test parameters
        param_group = QGroupBox("Test Parameters")
        param_layout = QGridLayout()
        param_group.setLayout(param_layout)
        
        # Sample delay
        param_layout.addWidget(QLabel("Sample Delay (ms):"), 0, 0)
        self.delay_spin = QSpinBox()
        self.delay_spin.setRange(50, 2000)
        self.delay_spin.setValue(self.delay_ms)
        param_layout.addWidget(self.delay_spin, 0, 1)
        
        # Test duration
        param_layout.addWidget(QLabel("Test Duration (s):"), 1, 0)
        self.duration_spin = QSpinBox()
        self.duration_spin.setRange(30, 300)
        self.duration_spin.setValue(self.test_duration)
        param_layout.addWidget(self.duration_spin, 1, 1)
        
        # Input amplitude
        param_layout.addWidget(QLabel("Input Amplitude (1-5V):"), 2, 0)
        self.amplitude_spin = QDoubleSpinBox()
        self.amplitude_spin.setRange(1.0, 5.0)
        self.amplitude_spin.setSingleStep(0.2)
        self.amplitude_spin.setValue(self.input_amplitude)
        param_layout.addWidget(self.amplitude_spin, 2, 1)
        
        # Max lag for correlation
        param_layout.addWidget(QLabel("Max Correlation Lag:"), 3, 0)
        self.lag_spin = QSpinBox()
        self.lag_spin.setRange(5, 50)
        self.lag_spin.setValue(self.max_lag)
        param_layout.addWidget(self.lag_spin, 3, 1)
        
        setup_layout.addWidget(param_group)
        
        # Memory test selection
        test_group = QGroupBox("Memory Tests")
        test_layout = QVBoxLayout()
        test_group.setLayout(test_layout)
        
        # Test type selection
        self.test_combo = QComboBox()
        self.test_combo.addItems([
            "Random Input (Baseline)",
            "Step Response Test", 
            "Pulse Train Test",
            "Sine Wave Test",
            "Custom Pattern Test"
        ])
        test_layout.addWidget(QLabel("Test Type:"))
        test_layout.addWidget(self.test_combo)
        
        # Control buttons
        button_layout = QHBoxLayout()
        
        self.connect_btn = QPushButton("Connect Devices")
        self.connect_btn.clicked.connect(self.connect_devices)
        button_layout.addWidget(self.connect_btn)
        
        self.start_btn = QPushButton("Start Memory Test")
        self.start_btn.clicked.connect(self.start_memory_test)
        self.start_btn.setEnabled(False)
        button_layout.addWidget(self.start_btn)
        
        self.stop_btn = QPushButton("Stop Test")
        self.stop_btn.clicked.connect(self.stop_test)
        self.stop_btn.setEnabled(False)
        button_layout.addWidget(self.stop_btn)
        
        self.analyze_btn = QPushButton("Analyze Memory")
        self.analyze_btn.clicked.connect(self.analyze_memory_effects)
        self.analyze_btn.setEnabled(False)
        button_layout.addWidget(self.analyze_btn)
        
        test_layout.addLayout(button_layout)
        setup_layout.addWidget(test_group)
        
        # Status
        self.status_label = QLabel("Ready - Connect devices to begin")
        setup_layout.addWidget(self.status_label)
        
        # Monitoring tab
        monitor_tab = QWidget()
        monitor_layout = QVBoxLayout(monitor_tab)
        tabs.addTab(monitor_tab, "Real-time Monitoring")
        
        # Monitoring plots
        self.monitor_figure = Figure(figsize=(12, 8))
        self.monitor_canvas = FigureCanvas(self.monitor_figure)
        monitor_layout.addWidget(self.monitor_canvas)
        
        # Analysis tab
        analysis_tab = QWidget()
        analysis_layout = QVBoxLayout(analysis_tab)
        tabs.addTab(analysis_tab, "Memory Analysis")
        
        # Analysis plots
        self.analysis_figure = Figure(figsize=(12, 10))
        self.analysis_canvas = FigureCanvas(self.analysis_figure)
        analysis_layout.addWidget(self.analysis_canvas)
        
        # Results display
        results_group = QGroupBox("Memory Analysis Results")
        results_layout = QVBoxLayout()
        results_group.setLayout(results_layout)
        
        self.results_text = QTextEdit()
        self.results_text.setMaximumHeight(150)
        results_layout.addWidget(self.results_text)
        
        analysis_layout.addWidget(results_group)

    def setup_digilent(self):
        """Initialize Digilent devices with retry (copied from working NARMA-10 app)"""
        max_retries = 3
        retry_delay = 2
        
        for attempt in range(max_retries):
            print(f"\n=== Device Setup Attempt {attempt + 1}/{max_retries} ===")
            
            try:
                if hasattr(self, 'dwf') and self.dwf:
                    print("Cleaning up previous initialization...")
                    try:
                        if hasattr(self, 'hdwf1'):
                            self.dwf.FDwfDeviceClose(self.hdwf1)
                        if hasattr(self, 'hdwf2'):
                            self.dwf.FDwfDeviceClose(self.hdwf2)
                    except:
                        pass
                    time.sleep(1)
                
                print("\n=== Starting Digilent Device Setup ===")
                try:
                    # Check if running in x86_64 mode
                    import platform
                    print(f"Current architecture: {platform.machine()}")
                    print(f"Current Python executable: {sys.executable}")
                    
                    if sys.platform.startswith("darwin"):
                        print("macOS platform detected, loading dwf framework...")
                        try:
                            framework_path = "/Library/Frameworks/dwf.framework/dwf"
                            if not os.path.exists(framework_path):
                                print(f"Error: Framework not found at {framework_path}")
                                raise FileNotFoundError(f"Framework not found at {framework_path}")
                            
                            print(f"Framework exists at: {framework_path}")
                            self.dwf = cdll.LoadLibrary(framework_path)
                            print("Successfully loaded framework")
                            
                            # First try to close any open devices
                            print("\nChecking for open devices...")
                            deviceCount = c_int()
                            self.dwf.FDwfEnum(c_int(0), byref(deviceCount))
                            for i in range(deviceCount.value):
                                hdwf = c_int()
                                self.dwf.FDwfDeviceOpen(c_int(i), byref(hdwf))
                                if hdwf.value != 0:
                                    print(f"Closing device {i}")
                                    self.dwf.FDwfDeviceClose(hdwf)
                            print("Cleaned up any open devices")
                            
                            # Wait a moment for devices to reset
                            time.sleep(2.0)
                            
                            # Now try to enumerate devices again
                            self.dwf.FDwfEnum(c_int(0), byref(deviceCount))
                            print(f"\nNumber of devices found: {deviceCount.value}")
                            
                            if deviceCount.value < 2:
                                raise Exception(f"Need 2 devices, but found only {deviceCount.value}")
                            
                            # Store handles for both devices
                            self.hdwf1 = c_int()  # First device for input
                            self.hdwf2 = c_int()  # Second device for reading state
                            
                            # Open and configure first device
                            print("\nOpening first device...")
                            result1 = self.dwf.FDwfDeviceOpen(c_int(0), byref(self.hdwf1))
                            if self.hdwf1.value == 0:
                                error_msg = self.get_error_message()
                                raise Exception(f"Failed to open first device: {error_msg}")
                            
                            # Open and configure second device
                            print("\nOpening second device...")
                            result2 = self.dwf.FDwfDeviceOpen(c_int(1), byref(self.hdwf2))
                            if self.hdwf2.value == 0:
                                error_msg = self.get_error_message()
                                self.dwf.FDwfDeviceClose(self.hdwf1)  # Clean up first device
                                raise Exception(f"Failed to open second device: {error_msg}")
                            
                            # Configure first device for analog output (both channels)
                            self.dwf.FDwfDeviceAutoConfigureSet(self.hdwf1, c_int(1))
                            self.dwf.FDwfAnalogOutNodeEnableSet(self.hdwf1, c_int(0), c_int(0), c_int(1))  # Channel 0 (Red)
                            self.dwf.FDwfAnalogOutNodeFunctionSet(self.hdwf1, c_int(0), c_int(0), c_int(1))
                            self.dwf.FDwfAnalogOutNodeEnableSet(self.hdwf1, c_int(1), c_int(0), c_int(1))  # Channel 1 (Green)
                            self.dwf.FDwfAnalogOutNodeFunctionSet(self.hdwf1, c_int(1), c_int(0), c_int(1))
                            
                            # Setup analog input for first device
                            self.dwf.FDwfAnalogInChannelEnableSet(self.hdwf1, c_int(0), c_int(1))  # Enable channel 1
                            self.dwf.FDwfAnalogInChannelRangeSet(self.hdwf1, c_int(0), c_double(5))  # Set range to 5V
                            self.dwf.FDwfAnalogInConfigure(self.hdwf1, c_int(0), c_int(0))
                            
                            # Configure second device for analog output and input
                            self.dwf.FDwfDeviceAutoConfigureSet(self.hdwf2, c_int(1))
                            self.dwf.FDwfAnalogOutNodeEnableSet(self.hdwf2, c_int(0), c_int(0), c_int(1))  # Channel 0 (Blue)
                            self.dwf.FDwfAnalogOutNodeFunctionSet(self.hdwf2, c_int(0), c_int(0), c_int(1))
                            self.dwf.FDwfAnalogInChannelEnableSet(self.hdwf2, c_int(0), c_int(1))  # Enable channel 1
                            self.dwf.FDwfAnalogInChannelRangeSet(self.hdwf2, c_int(0), c_double(5))  # Set range to 5V
                            self.dwf.FDwfAnalogInConfigure(self.hdwf2, c_int(0), c_int(0))
                            
                            print("Both devices configured successfully!")
                            
                        except OSError as e:
                            print(f"Failed to load library: {e}")
                            raise
                        
                    print("\n=== Device initialization completed successfully! ===")
                    
                except Exception as e:
                    print(f"\n!!! Error initializing Digilent !!!")
                    print(f"Error details: {str(e)}")
                    print("Stack trace:")
                    import traceback
                    traceback.print_exc()
                    self.dwf = None
                    self.hdwf1 = None
                    self.hdwf2 = None
                    return False
                
                return True
                
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt < max_retries - 1:
                    print(f"Waiting {retry_delay} seconds before retry...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Increase delay for next retry
                else:
                    print("All attempts failed")
                    self.dwf = None
                    return False
        
        return False

    def get_error_message(self):
        """Get the last error message from the device"""
        error_msg = create_string_buffer(512)
        self.dwf.FDwfGetLastErrorMsg(error_msg)
        return error_msg.value.decode()

    def scale_to_voltage(self, value, min_val=0, max_val=0.5, min_out=1.0, max_out=5.0):
        """Scale normalized values to voltage range (1-5V like NARMA-10 app)"""
        return min_out + (value - min_val) * (max_out - min_out) / (max_val - min_val)

    def read_voltage(self):
        """Read voltage with averaging (copied from working NARMA-10 app)"""
        if not hasattr(self, 'dwf') or not self.dwf:
            print("DEBUG: No dwf library - returning None")
            return None
        
        if not hasattr(self, 'hdwf1') or not self.hdwf1:
            print("DEBUG: No hdwf1 device handle - returning None")
            return None
        
        try:
            v_avg = 0
            num_samples = 5  # Average 5 readings
            
            print(f"DEBUG: Reading from hdwf1 device (like NARMA-10), 5 samples...")
            for i in range(num_samples):
                self.dwf.FDwfAnalogInStatus(self.hdwf1, c_int(False), None)
                voltage = c_double()
                self.dwf.FDwfAnalogInStatusSample(self.hdwf1, c_int(0), byref(voltage))
                v_avg += voltage.value
                print(f"DEBUG: Sample {i+1}: {voltage.value:.6f}V")
                time.sleep(0.01)  # Small delay between readings
            
            v_avg = v_avg / num_samples
            print(f"DEBUG: Average voltage: {v_avg:.6f}V")
            return v_avg
            
        except Exception as e:
            print(f"DEBUG: Error reading voltage: {e}")
            import traceback
            traceback.print_exc()
            return None

    def connect_devices(self):
        """Connect to Digilent devices"""
        device_init_success = self.setup_digilent()
        if device_init_success and self.dwf:
            self.status_label.setText("Devices connected successfully")
            self.start_btn.setEnabled(True)
            self.connect_btn.setText("Reconnect Devices")
        else:
            self.status_label.setText("Device connection failed - running in simulation mode")
            self.start_btn.setEnabled(True)  # Allow testing in simulation mode
            self.connect_btn.setText("Connect Devices")

    def update_params_from_ui(self):
        """Update parameters from UI controls"""
        self.delay_ms = self.delay_spin.value()
        self.test_duration = self.duration_spin.value()
        self.input_amplitude = self.amplitude_spin.value()
        self.max_lag = self.lag_spin.value()

    def start_memory_test(self):
        """Start the memory test sequence"""
        self.update_params_from_ui()
        
        # Clear previous data
        self.timestamps.clear()
        self.inputs.clear()
        self.states.clear()
        self.current_sample = 0
        
        self.test_type = self.test_combo.currentText()
        self.is_testing = True
        
        # Update UI
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.analyze_btn.setEnabled(False)
        self.status_label.setText(f"Running {self.test_type}...")
        
        # Start monitoring
        self.monitor_timer.start(self.delay_ms)

    def generate_test_input(self, sample_index):
        """Generate input voltage based on selected test type (direct 1-5V range)"""
        if self.test_type == "Random Input (Baseline)":
            return np.random.uniform(1.0, self.input_amplitude)
        
        elif self.test_type == "Step Response Test":
            # Alternating steps: low for step_duration, high for step_duration
            cycle_length = 2 * self.step_duration
            position_in_cycle = sample_index % cycle_length
            if position_in_cycle < self.step_duration:
                return 1.2  # Low level (1.2V)
            else:
                return self.input_amplitude  # High level (user-specified)
        
        elif self.test_type == "Pulse Train Test":
            # Single sample pulses with gaps
            if sample_index % 5 == 0:  # Pulse every 5 samples
                return self.input_amplitude
            else:
                return 1.2  # Low baseline (1.2V)
        
        elif self.test_type == "Sine Wave Test":
            # Slow sine wave to test frequency response
            period = 40  # samples
            return 1.2 + (self.input_amplitude - 1.2) * (1 + np.sin(2 * np.pi * sample_index / period)) / 2
        
        elif self.test_type == "Custom Pattern Test":
            # Predefined pattern for specific memory testing (direct voltages)
            pattern = [1.2, 1.2, self.input_amplitude, 1.2, 2.0, 3.0, self.input_amplitude, 1.2]
            return pattern[sample_index % len(pattern)]
        
        return 1.2  # Default (1.2V baseline)

    def update_monitoring(self):
        """Update monitoring during test"""
        if not self.is_testing:
            return
        
        try:
            # Generate input voltage directly (1-5V range)
            input_voltage = self.generate_test_input(self.current_sample)
            
            # Apply input to all three pins (like NARMA-10 app)
            if self.hdwf1:
                # Set all three channels to the same voltage to maximize signal
                self.dwf.FDwfAnalogOutNodeOffsetSet(self.hdwf1, c_int(0), c_int(0), c_double(input_voltage))  # R - Red
                self.dwf.FDwfAnalogOutNodeOffsetSet(self.hdwf1, c_int(1), c_int(0), c_double(input_voltage))  # G - Green
            if self.hdwf2:
                self.dwf.FDwfAnalogOutNodeOffsetSet(self.hdwf2, c_int(0), c_int(0), c_double(input_voltage))  # B - Blue
            
            # Read state from device 2 using robust method
            state_value = self.read_voltage()
            if state_value is None:
                # No simulation - if hardware fails, stop the test
                self.status_label.setText("ERROR: Cannot read voltage from hardware - stopping test")
                self.stop_test()
                return
            
            print(f"DEBUG: Read from hardware - state_value: {state_value:.6f}V")
            
            # Store data
            self.timestamps.append(time.time())
            self.inputs.append(input_voltage)
            self.states.append(state_value)
            self.current_sample += 1
            
            # Update plots
            self.update_monitoring_plots()
            
            # Check if test is complete
            max_samples = int(self.test_duration * 1000 / self.delay_ms)
            if self.current_sample >= max_samples:
                self.finish_test()
            
        except Exception as e:
            self.status_label.setText(f"Monitoring error: {str(e)}")
            self.stop_test()

    def update_monitoring_plots(self):
        """Update real-time monitoring plots"""
        self.monitor_figure.clear()
        
        if len(self.inputs) < 2:
            return
        
        # Create subplots
        ax1 = self.monitor_figure.add_subplot(3, 1, 1)
        ax2 = self.monitor_figure.add_subplot(3, 1, 2)
        ax3 = self.monitor_figure.add_subplot(3, 1, 3)
        
        time_axis = np.arange(len(self.inputs)) * self.delay_ms / 1000
        
        # Input signal
        ax1.plot(time_axis, self.inputs, 'b-', linewidth=1)
        ax1.set_ylabel('Input (V)')
        ax1.set_title(f'{self.test_type} - Sample {self.current_sample}')
        ax1.grid(True)
        
        # State signal
        ax2.plot(time_axis, self.states, 'r-', linewidth=1)
        ax2.set_ylabel('State (V)')
        ax2.grid(True)
        
        # Input-State overlay (recent samples)
        recent_samples = min(100, len(self.inputs))
        ax3.plot(time_axis[-recent_samples:], self.inputs[-recent_samples:], 'b-', 
                label='Input', alpha=0.7)
        ax3.plot(time_axis[-recent_samples:], self.states[-recent_samples:], 'r-', 
                label='State', alpha=0.7)
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Voltage (V)')
        ax3.legend()
        ax3.grid(True)
        
        self.monitor_figure.tight_layout()
        self.monitor_canvas.draw()

    def stop_test(self):
        """Stop the current test"""
        self.is_testing = False
        self.monitor_timer.stop()
        
        # Update UI
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        
        if len(self.states) > 0:
            # Save data when stopping
            self.save_test_data()
            self.analyze_btn.setEnabled(True)
            self.status_label.setText(f"Test stopped. Collected {len(self.states)} samples - Data saved")
        else:
            self.status_label.setText("Test stopped - no data collected")
        
        # Re-enable start button after a brief delay
        QTimer.singleShot(1000, lambda: self.start_btn.setEnabled(True))

    def finish_test(self):
        """Complete the test and enable analysis"""
        self.stop_test()
        self.status_label.setText(f"Test complete! Collected {len(self.states)} samples")
        
        # Save data
        self.save_test_data()

    def save_test_data(self):
        """Save test data to CSV"""
        if len(self.states) == 0:
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        test_name = self.test_type.replace(" ", "_").lower()
        filename = f"memory_test_{test_name}_{timestamp}.csv"
        
        df = pd.DataFrame({
            'sample': range(len(self.inputs)),
            'time_s': np.arange(len(self.inputs)) * self.delay_ms / 1000,
            'input': self.inputs,
            'state': self.states
        })
        
        df.to_csv(filename, index=False)
        self.status_label.setText(f"Data saved to {filename}")

    def analyze_memory_effects(self):
        """Perform comprehensive memory analysis"""
        if len(self.states) < self.max_lag * 2:
            self.results_text.setText("Not enough data for memory analysis")
            return
        
        self.status_label.setText("Analyzing temporal memory effects...")
        
        # Perform analyses
        autocorr = self.analyze_autocorrelation()
        cross_corr = self.analyze_cross_correlation()
        prediction_scores = self.analyze_state_prediction()
        decay_analysis = self.analyze_response_decay()
        
        # Update results display
        self.display_analysis_results(autocorr, cross_corr, prediction_scores, decay_analysis)
        
        # Update analysis plots
        self.update_analysis_plots(autocorr, cross_corr, prediction_scores)
        
        # Generate report
        self.generate_memory_report(autocorr, cross_corr, prediction_scores, decay_analysis)
        
        self.status_label.setText("Memory analysis complete!")

    def analyze_autocorrelation(self):
        """Analyze temporal correlations in state data"""
        states = np.array(self.states)
        autocorr = []
        lags = range(1, min(self.max_lag + 1, len(states) // 2))
        
        for lag in lags:
            corr = np.corrcoef(states[:-lag], states[lag:])[0, 1]
            autocorr.append(corr)
        
        return {'lags': list(lags), 'correlations': autocorr}

    def analyze_cross_correlation(self):
        """Analyze correlation between past inputs and current states"""
        inputs = np.array(self.inputs)
        states = np.array(self.states)
        cross_corr = []
        lags = range(1, min(self.max_lag + 1, len(inputs) - 1))
        
        for lag in lags:
            corr = np.corrcoef(inputs[:-lag], states[lag:])[0, 1]
            cross_corr.append(corr)
        
        return {'lags': list(lags), 'correlations': cross_corr}

    def analyze_state_prediction(self):
        """Test prediction accuracy using temporal features"""
        if len(self.states) < 20:
            return {'current_only': 0, 'with_history': 0, 'improvement': 0}
        
        states = np.array(self.states)
        inputs = np.array(self.inputs)
        
        # Prepare data for prediction (predict state from inputs)
        X_current = inputs[:-1].reshape(-1, 1)  # Current input only
        X_history = []  # Current + past inputs
        
        history_length = min(5, len(inputs) - 1)
        for i in range(history_length, len(inputs) - 1):
            features = inputs[i-history_length:i+1]  # Include current + past
            X_history.append(features)
        
        X_history = np.array(X_history)
        y_current = states[1:]  # Target states
        y_history = states[history_length+1:]  # Target states (aligned with history features)
        
        # Train models
        model_current = LinearRegression()
        model_history = LinearRegression()
        
        model_current.fit(X_current, y_current)
        model_history.fit(X_history, y_history)
        
        # Evaluate predictions
        score_current = model_current.score(X_current, y_current)
        score_history = model_history.score(X_history, y_history)
        
        improvement = score_history - score_current
        
        return {
            'current_only': score_current,
            'with_history': score_history,
            'improvement': improvement
        }

    def analyze_response_decay(self):
        """Analyze how long states take to stabilize after input changes"""
        if self.test_type != "Step Response Test":
            return {'message': 'Response decay analysis requires Step Response Test'}
        
        inputs = np.array(self.inputs)
        states = np.array(self.states)
        
        # Find step transitions
        transitions = []
        for i in range(1, len(inputs)):
            if abs(inputs[i] - inputs[i-1]) > 0.1:  # Significant input change
                transitions.append(i)
        
        decay_times = []
        for trans in transitions:
            if trans + 10 < len(states):  # Need enough samples after transition
                pre_state = np.mean(states[max(0, trans-3):trans])
                post_states = states[trans:trans+10]
                
                # Find when state reaches 90% of final value
                final_state = np.mean(states[trans+7:trans+10])
                threshold = pre_state + 0.9 * (final_state - pre_state)
                
                decay_time = 0
                for j, state in enumerate(post_states):
                    if abs(state - final_state) < abs(threshold - final_state):
                        decay_time = j * self.delay_ms / 1000  # Convert to seconds
                        break
                
                decay_times.append(decay_time)
        
        avg_decay = np.mean(decay_times) if decay_times else 0
        
        return {
            'transitions': len(transitions),
            'decay_times': decay_times,
            'average_decay': avg_decay
        }

    def display_analysis_results(self, autocorr, cross_corr, prediction, decay):
        """Display analysis results in text widget"""
        results = []
        results.append("=== TEMPORAL MEMORY ANALYSIS RESULTS ===\n")
        
        # Autocorrelation analysis
        if autocorr['correlations']:
            max_autocorr = max(autocorr['correlations'])
            max_lag = autocorr['lags'][np.argmax(autocorr['correlations'])]
            results.append(f"AUTOCORRELATION:")
            results.append(f"  Max correlation: {max_autocorr:.4f} at lag {max_lag}")
            results.append(f"  Memory persistence: {'Strong' if max_autocorr > 0.5 else 'Moderate' if max_autocorr > 0.2 else 'Weak'}")
        
        # Cross-correlation analysis
        if cross_corr['correlations']:
            max_cross_corr = max([abs(c) for c in cross_corr['correlations']])
            max_lag = cross_corr['lags'][np.argmax([abs(c) for c in cross_corr['correlations']])]
            results.append(f"\nCROSS-CORRELATION:")
            results.append(f"  Max input-state correlation: {max_cross_corr:.4f} at lag {max_lag}")
            results.append(f"  Input memory effect: {'Strong' if max_cross_corr > 0.3 else 'Moderate' if max_cross_corr > 0.1 else 'Weak'}")
        
        # Prediction analysis
        results.append(f"\nPREDICTION ANALYSIS:")
        results.append(f"  Current input only R²: {prediction['current_only']:.4f}")
        results.append(f"  With input history R²: {prediction['with_history']:.4f}")
        results.append(f"  Improvement: {prediction['improvement']:.4f}")
        results.append(f"  Temporal benefit: {'Significant' if prediction['improvement'] > 0.1 else 'Moderate' if prediction['improvement'] > 0.02 else 'Minimal'}")
        
        # Decay analysis
        if 'average_decay' in decay:
            results.append(f"\nRESPONSE DECAY:")
            results.append(f"  Average settling time: {decay['average_decay']:.3f} seconds")
            results.append(f"  Number of transitions: {decay['transitions']}")
        else:
            results.append(f"\nRESPONSE DECAY: {decay['message']}")
        
        # Overall assessment
        memory_score = 0
        if autocorr['correlations'] and max(autocorr['correlations']) > 0.2:
            memory_score += 1
        if cross_corr['correlations'] and max([abs(c) for c in cross_corr['correlations']]) > 0.1:
            memory_score += 1
        if prediction['improvement'] > 0.02:
            memory_score += 1
        
        results.append(f"\n=== OVERALL MEMORY ASSESSMENT ===")
        results.append(f"Memory score: {memory_score}/3")
        if memory_score >= 2:
            results.append("CONCLUSION: Strong evidence of temporal memory effects")
        elif memory_score == 1:
            results.append("CONCLUSION: Moderate evidence of temporal memory effects")
        else:
            results.append("CONCLUSION: Limited evidence of temporal memory effects")
        
        self.results_text.setText('\n'.join(results))

    def update_analysis_plots(self, autocorr, cross_corr, prediction):
        """Update analysis visualization plots"""
        self.analysis_figure.clear()
        
        # Create subplots
        ax1 = self.analysis_figure.add_subplot(2, 3, 1)
        ax2 = self.analysis_figure.add_subplot(2, 3, 2)
        ax3 = self.analysis_figure.add_subplot(2, 3, 3)
        ax4 = self.analysis_figure.add_subplot(2, 3, 4)
        ax5 = self.analysis_figure.add_subplot(2, 3, 5)
        ax6 = self.analysis_figure.add_subplot(2, 3, 6)
        
        # Autocorrelation plot
        if autocorr['correlations']:
            ax1.plot(autocorr['lags'], autocorr['correlations'], 'bo-')
            ax1.axhline(y=0, color='k', linestyle='--', alpha=0.5)
            ax1.set_xlabel('Lag (samples)')
            ax1.set_ylabel('Autocorrelation')
            ax1.set_title('State Autocorrelation')
            ax1.grid(True)
        
        # Cross-correlation plot
        if cross_corr['correlations']:
            ax2.plot(cross_corr['lags'], cross_corr['correlations'], 'ro-')
            ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
            ax2.set_xlabel('Lag (samples)')
            ax2.set_ylabel('Cross-correlation')
            ax2.set_title('Input-State Cross-correlation')
            ax2.grid(True)
        
        # Prediction comparison
        categories = ['Current Only', 'With History']
        scores = [prediction['current_only'], prediction['with_history']]
        ax3.bar(categories, scores, color=['blue', 'orange'])
        ax3.set_ylabel('R² Score')
        ax3.set_title('Prediction Performance')
        ax3.set_ylim(0, max(1, max(scores) * 1.1))
        
        # Time series overview
        if len(self.states) > 0:
            time_axis = np.arange(len(self.inputs)) * self.delay_ms / 1000
            ax4.plot(time_axis, self.inputs, 'b-', alpha=0.7, label='Input')
            ax4.plot(time_axis, self.states, 'r-', alpha=0.7, label='State')
            ax4.set_xlabel('Time (s)')
            ax4.set_ylabel('Voltage (V)')
            ax4.set_title('Full Time Series')
            ax4.legend()
            ax4.grid(True)
        
        # State distribution
        if len(self.states) > 0:
            ax5.hist(self.states, bins=20, alpha=0.7, color='red')
            ax5.set_xlabel('State Voltage (V)')
            ax5.set_ylabel('Frequency')
            ax5.set_title('State Distribution')
            ax5.grid(True)
        
        # Input-state scatter
        if len(self.states) > 1:
            ax6.scatter(self.inputs[:-1], self.states[1:], alpha=0.6, s=10)
            ax6.set_xlabel('Input (t)')
            ax6.set_ylabel('State (t+1)')
            ax6.set_title('Input-State Relationship')
            ax6.grid(True)
        
        self.analysis_figure.tight_layout()
        self.analysis_canvas.draw()

    def generate_memory_report(self, autocorr, cross_corr, prediction, decay):
        """Generate comprehensive PDF report of memory analysis"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"mycelium_memory_analysis_report_{timestamp}.pdf"
        
        with PdfPages(filename) as pdf:
            # Page 1: Title and Test Configuration
            fig = plt.figure(figsize=(8.5, 11))
            fig.suptitle('Mycelium Temporal Memory Analysis Report', fontsize=18, fontweight='bold', y=0.95)
            
            # Test parameters section
            ax_config = fig.add_subplot(111)
            ax_config.axis('off')
            
            # Calculate some derived parameters for the report
            total_test_time = len(self.states) * self.delay_ms / 1000  # seconds
            max_lag_time = self.max_lag * self.delay_ms / 1000  # seconds
            
            config_text = f"""
EXPERIMENT OVERVIEW
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Date & Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Report Generated: {filename}

TEST CONFIGURATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Test Type: {self.test_type}
Sample Delay: {self.delay_ms} ms ({self.delay_ms/1000:.1f} seconds per sample)
Test Duration Setting: {self.test_duration} seconds
Actual Test Duration: {total_test_time:.1f} seconds
Input Amplitude: {self.input_amplitude:.1f} V (voltage range applied to mycelium)
Max Correlation Lag: {self.max_lag} samples ({max_lag_time:.1f} seconds lookback)
Total Data Points: {len(self.states)} samples

DATA COLLECTION SUMMARY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Input Voltage Range: {min(self.inputs):.2f}V to {max(self.inputs):.2f}V
State Voltage Range: {min(self.states):.3f}V to {max(self.states):.3f}V
State Voltage Mean: {np.mean(self.states):.3f}V ± {np.std(self.states):.3f}V
Data Quality: {"✓ Good" if len(self.states) > 50 else "⚠ Limited"}

HARDWARE CONFIGURATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Device Mode: {"Real Hardware" if hasattr(self, 'dwf') and self.dwf else "Simulation"}
Input Device: {"Digilent Device 1" if hasattr(self, 'dwf') and self.dwf else "Simulated"}
Measurement Device: {"Digilent Device 2" if hasattr(self, 'dwf') and self.dwf else "Simulated"}
            """
            
            ax_config.text(0.05, 0.95, config_text, transform=ax_config.transAxes, 
                          verticalalignment='top', fontsize=10, fontfamily='monospace')
            
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
            
            # Page 2: Analysis Results Summary
            fig = plt.figure(figsize=(8.5, 11))
            fig.suptitle('Temporal Memory Analysis Results', fontsize=16, fontweight='bold', y=0.95)
            
            ax_results = fig.add_subplot(111)
            ax_results.axis('off')
            
            # Calculate memory score
            memory_score = 0
            if autocorr['correlations'] and max(autocorr['correlations']) > 0.2:
                memory_score += 1
            if cross_corr['correlations'] and max([abs(c) for c in cross_corr['correlations']]) > 0.1:
                memory_score += 1
            if prediction['improvement'] > 0.02:
                memory_score += 1
            
            # Detailed results text
            if autocorr['correlations']:
                max_autocorr = max(autocorr['correlations'])
                autocorr_lag = autocorr['lags'][np.argmax(autocorr['correlations'])]
                autocorr_time = autocorr_lag * self.delay_ms / 1000
            else:
                max_autocorr = 0
                autocorr_lag = 0
                autocorr_time = 0
                
            if cross_corr['correlations']:
                max_cross_corr = max([abs(c) for c in cross_corr['correlations']])
                cross_corr_lag = cross_corr['lags'][np.argmax([abs(c) for c in cross_corr['correlations']])]
                cross_corr_time = cross_corr_lag * self.delay_ms / 1000
            else:
                max_cross_corr = 0
                cross_corr_lag = 0
                cross_corr_time = 0
            
            results_text = f"""
TEMPORAL MEMORY ANALYSIS RESULTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

AUTOCORRELATION ANALYSIS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Maximum Correlation: {max_autocorr:.4f}
Optimal Lag: {autocorr_lag} samples ({autocorr_time:.1f} seconds)
Memory Persistence: {'Strong' if max_autocorr > 0.5 else 'Moderate' if max_autocorr > 0.2 else 'Weak'}
Interpretation: {'States show strong self-similarity over time' if max_autocorr > 0.5 else 'States show moderate self-similarity' if max_autocorr > 0.2 else 'States show weak temporal structure'}

CROSS-CORRELATION ANALYSIS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Maximum Input-State Correlation: {max_cross_corr:.4f}
Optimal Lag: {cross_corr_lag} samples ({cross_corr_time:.1f} seconds)
Input Memory Effect: {'Strong' if max_cross_corr > 0.3 else 'Moderate' if max_cross_corr > 0.1 else 'Weak'}
Interpretation: {'Past inputs strongly influence current states' if max_cross_corr > 0.3 else 'Past inputs moderately influence current states' if max_cross_corr > 0.1 else 'Past inputs weakly influence current states'}

STATE PREDICTION ANALYSIS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Current Input Only R²: {prediction['current_only']:.4f}
With Input History R²: {prediction['with_history']:.4f}
Temporal Improvement: {prediction['improvement']:.4f}
Temporal Benefit: {'Significant' if prediction['improvement'] > 0.1 else 'Moderate' if prediction['improvement'] > 0.02 else 'Minimal'}
Interpretation: {'Input history significantly improves state prediction' if prediction['improvement'] > 0.1 else 'Input history moderately improves prediction' if prediction['improvement'] > 0.02 else 'Input history provides minimal predictive benefit'}
            """
            
            # Add decay analysis if available
            if 'average_decay' in decay:
                results_text += f"""
RESPONSE DECAY ANALYSIS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Number of Transitions: {decay['transitions']}
Average Settling Time: {decay['average_decay']:.3f} seconds
Interpretation: {'Fast response dynamics' if decay['average_decay'] < 1.0 else 'Moderate response dynamics' if decay['average_decay'] < 3.0 else 'Slow response dynamics'}
"""
            else:
                results_text += f"""
RESPONSE DECAY ANALYSIS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Status: {decay.get('message', 'Not performed')}
"""
            
            # Overall assessment
            results_text += f"""
OVERALL MEMORY ASSESSMENT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Memory Score: {memory_score}/3
{'★★★' if memory_score >= 2 else '★★☆' if memory_score == 1 else '★☆☆'}

CONCLUSION: {'🧠 Strong evidence of temporal memory effects' if memory_score >= 2 else '⚡ Moderate evidence of temporal memory effects' if memory_score == 1 else '📊 Limited evidence of temporal memory effects'}

RESERVOIR COMPUTING ASSESSMENT:
{'✓ This mycelium sample shows excellent reservoir computing potential' if memory_score >= 2 else '⚠ This mycelium sample shows some reservoir computing potential' if memory_score == 1 else '❌ This mycelium sample shows limited reservoir computing potential'}
            """
            
            ax_results.text(0.05, 0.95, results_text, transform=ax_results.transAxes, 
                           verticalalignment='top', fontsize=9, fontfamily='monospace')
            
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
            
            # Page 3: Real-time monitoring plots
            if hasattr(self, 'monitor_figure'):
                pdf.savefig(self.monitor_figure, bbox_inches='tight')
            
            # Page 4: Analysis plots
            pdf.savefig(self.analysis_figure, bbox_inches='tight')
            
            # Page 5: Data tables (if we have enough data)
            if len(self.states) > 0:
                fig = plt.figure(figsize=(8.5, 11))
                fig.suptitle('Data Summary Tables', fontsize=16, fontweight='bold')
                
                # Create data summary table
                ax_table = fig.add_subplot(111)
                ax_table.axis('off')
                
                # Statistical summary
                stats_data = [
                    ['Parameter', 'Input (V)', 'State (V)'],
                    ['Mean', f'{np.mean(self.inputs):.3f}', f'{np.mean(self.states):.3f}'],
                    ['Std Dev', f'{np.std(self.inputs):.3f}', f'{np.std(self.states):.3f}'],
                    ['Min', f'{np.min(self.inputs):.3f}', f'{np.min(self.states):.3f}'],
                    ['Max', f'{np.max(self.inputs):.3f}', f'{np.max(self.states):.3f}'],
                    ['Range', f'{np.max(self.inputs)-np.min(self.inputs):.3f}', f'{np.max(self.states)-np.min(self.states):.3f}']
                ]
                
                table = ax_table.table(cellText=stats_data[1:], colLabels=stats_data[0],
                                     cellLoc='center', loc='upper center',
                                     bbox=[0.1, 0.7, 0.8, 0.25])
                table.auto_set_font_size(False)
                table.set_fontsize(10)
                table.scale(1, 2)
                
                # Add correlation results table
                if autocorr['correlations'] and cross_corr['correlations']:
                    corr_data = [
                        ['Lag (samples)', 'Lag (seconds)', 'Autocorr', 'Cross-corr'],
                    ]
                    
                    # Show first 10 lags
                    for i in range(min(10, len(autocorr['lags']))):
                        lag_samples = autocorr['lags'][i]
                        lag_seconds = lag_samples * self.delay_ms / 1000
                        auto_val = autocorr['correlations'][i] if i < len(autocorr['correlations']) else 0
                        cross_val = cross_corr['correlations'][i] if i < len(cross_corr['correlations']) else 0
                        corr_data.append([
                            f'{lag_samples}',
                            f'{lag_seconds:.1f}',
                            f'{auto_val:.3f}',
                            f'{cross_val:.3f}'
                        ])
                    
                    table2 = ax_table.table(cellText=corr_data[1:], colLabels=corr_data[0],
                                          cellLoc='center', loc='lower center',
                                          bbox=[0.1, 0.1, 0.8, 0.5])
                    table2.auto_set_font_size(False)
                    table2.set_fontsize(9)
                    table2.scale(1, 1.5)
                
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)
        
        self.status_label.setText(f"Comprehensive report saved to {filename}")
        print(f"Generated comprehensive PDF report: {filename}")

    def closeEvent(self, event):
        """Clean up when closing the application"""
        if self.is_testing:
            self.stop_test()
        
        # Close device connections
        if self.dwf and self.hdwf1:
            self.dwf.FDwfDeviceClose(self.hdwf1)
        if self.dwf and self.hdwf2:
            self.dwf.FDwfDeviceClose(self.hdwf2)
        
        event.accept()

def main():
    app = QApplication(sys.argv)
    window = MyceliumMemoryApp()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main() 