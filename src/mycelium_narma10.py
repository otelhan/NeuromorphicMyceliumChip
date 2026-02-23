import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QLabel, QPushButton, QSlider, 
                            QGroupBox, QSpinBox, QDoubleSpinBox, QProgressBar,
                            QTabWidget, QGridLayout, QCheckBox, QFileDialog, QComboBox)
from PyQt6.QtCore import Qt, QTimer
import sys
import time
import os
from ctypes import c_int, c_double, c_uint, byref, create_string_buffer, cdll
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pickle
from datetime import datetime
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import learning_curve

class NarmaCanvas(FigureCanvas):
    """Canvas for displaying NARMA plots"""
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = plt.Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)
        
        # Create subplots
        self.input_ax = self.fig.add_subplot(311)
        self.state_ax = self.fig.add_subplot(312)
        self.output_ax = self.fig.add_subplot(313)
        
        # Setup plot labels
        self.input_ax.set_title('Input Signal')
        self.input_ax.set_ylabel('Input Value')
        
        self.state_ax.set_title('Reservoir State')
        self.state_ax.set_ylabel('Voltage (V)')
        
        self.output_ax.set_title('Target vs Prediction')
        self.output_ax.set_ylabel('Output')
        self.output_ax.set_xlabel('Time Step')
        
        # Initialize data storage
        self.input_line, = self.input_ax.plot([], [], 'b-', label='Input')
        self.state_line, = self.state_ax.plot([], [], 'g-', label='State')
        self.target_line, = self.output_ax.plot([], [], 'r-', label='Target')
        self.pred_line, = self.output_ax.plot([], [], 'c--', label='Prediction')
        
        # Add legends
        self.input_ax.legend()
        self.state_ax.legend()
        self.output_ax.legend()
        
        self.fig.tight_layout()
        
    def update_plots(self, inputs, states, targets, predictions):
        """Update all plots with new data"""
        # Get time axis
        t = np.arange(len(inputs))
        
        # Update data
        self.input_line.set_data(t, inputs)
        self.state_line.set_data(t, states)
        self.target_line.set_data(t, targets)
        self.pred_line.set_data(t, predictions)
        
        # Adjust axes limits
        for ax in [self.input_ax, self.state_ax, self.output_ax]:
            ax.relim()
            ax.autoscale_view()
        
        # Draw canvas
        self.fig.tight_layout()
        self.draw()

class MyceliumNarma10App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Mycelium NARMA-10 Experiment")
        self.setMinimumSize(900, 700)
        
        # Setup variables
        self.order = 10
        self.train_len = 1000
        self.test_len = 200
        self.washout = 100
        self.alpha = 0.3
        self.delay_ms = 300
        
        # Random Forest parameters
        self.n_estimators = 100
        self.max_depth = 10
        
        # Extended readout parameters (defaults preserve baseline behavior)
        self.use_state_taps = False
        self.n_state_taps = 10
        self.input_v_min = 1.0  # Default matches current behavior
        self.input_v_max = 5.0  # Default matches current behavior
        
        # Data storage
        self.inputs = []
        self.states = []
        self.targets = []
        self.predictions = []
        self.model = None
        self.is_training = False
        self.is_testing = False
        self.current_step = 0
        self.max_steps = 0
        
        # Initialize Digilent
        self.setup_digilent()
        
        # Setup UI
        self.setup_ui()
        
        # Setup timer for monitoring
        self.monitor_timer = QTimer()
        self.monitor_timer.timeout.connect(self.update_monitoring)
        self.monitor_timer.setInterval(100)  # Update every 100ms
        
    def setup_ui(self):
        """Create the user interface"""
        # Central widget and layout
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
        
        # Parameter settings
        param_group = QGroupBox("NARMA-10 Parameters")
        param_layout = QGridLayout()
        param_group.setLayout(param_layout)
        
        # Training length
        param_layout.addWidget(QLabel("Training Samples:"), 0, 0)
        self.train_spin = QSpinBox()
        self.train_spin.setRange(100, 5000)
        self.train_spin.setValue(self.train_len)
        self.train_spin.setSingleStep(100)
        param_layout.addWidget(self.train_spin, 0, 1)
        
        # Testing length
        param_layout.addWidget(QLabel("Testing Samples:"), 1, 0)
        self.test_spin = QSpinBox()
        self.test_spin.setRange(10, 1000)
        self.test_spin.setValue(self.test_len)
        self.test_spin.setSingleStep(10)
        param_layout.addWidget(self.test_spin, 1, 1)
        
        # Washout period
        param_layout.addWidget(QLabel("Washout Period:"), 2, 0)
        self.washout_spin = QSpinBox()
        self.washout_spin.setRange(0, 500)
        self.washout_spin.setValue(self.washout)
        self.washout_spin.setSingleStep(10)
        param_layout.addWidget(self.washout_spin, 2, 1)
        
        # Alpha (regularization)
        param_layout.addWidget(QLabel("Regularization (α):"), 3, 0)
        self.alpha_spin = QDoubleSpinBox()
        self.alpha_spin.setRange(0.0001, 10.0)
        self.alpha_spin.setValue(self.alpha)
        self.alpha_spin.setSingleStep(0.05)
        self.alpha_spin.setDecimals(4)
        param_layout.addWidget(self.alpha_spin, 3, 1)

        # Ridge Regression extra settings
        param_layout.addWidget(QLabel("Ridge fit_intercept:"), 3, 2)
        self.ridge_fit_intercept_check = QCheckBox()
        self.ridge_fit_intercept_check.setChecked(True)
        param_layout.addWidget(self.ridge_fit_intercept_check, 3, 3)

        param_layout.addWidget(QLabel("Ridge normalize:"), 4, 2)
        self.ridge_normalize_check = QCheckBox()
        self.ridge_normalize_check.setChecked(False)
        param_layout.addWidget(self.ridge_normalize_check, 4, 3)

        param_layout.addWidget(QLabel("Ridge solver:"), 5, 2)
        self.ridge_solver_combo = QComboBox()
        self.ridge_solver_combo.addItems(["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga", "lbfgs"])
        self.ridge_solver_combo.setCurrentText("auto")
        param_layout.addWidget(self.ridge_solver_combo, 5, 3)
        
        # Delay between samples
        param_layout.addWidget(QLabel("Sample Delay (ms):"), 6, 0)
        self.delay_spin = QSpinBox()
        self.delay_spin.setRange(10, 1000)
        self.delay_spin.setValue(self.delay_ms)  # Default: 300ms
        self.delay_spin.setSingleStep(10)
        param_layout.addWidget(self.delay_spin, 6, 1)
        
        # Channel Selection
        param_layout.addWidget(QLabel("Active Channels:"), 7, 0, 1, 2)
        
        # Channel checkboxes
        channel_layout = QHBoxLayout()
        self.channel_1_check = QCheckBox("Channel 1")
        self.channel_2_check = QCheckBox("Channel 2")  
        self.channel_3_check = QCheckBox("Channel 3")
        
        # Default to all channels enabled
        self.channel_1_check.setChecked(True)
        self.channel_2_check.setChecked(True)
        self.channel_3_check.setChecked(True)
        
        # Connect to validation function
        self.channel_1_check.toggled.connect(self.validate_channels)
        self.channel_2_check.toggled.connect(self.validate_channels)
        self.channel_3_check.toggled.connect(self.validate_channels)
        
        channel_layout.addWidget(self.channel_1_check)
        channel_layout.addWidget(self.channel_2_check)
        channel_layout.addWidget(self.channel_3_check)
        
        channel_widget = QWidget()
        channel_widget.setLayout(channel_layout)
        param_layout.addWidget(channel_widget, 8, 0, 1, 2)
        
        # Add Random Forest parameters
        param_layout.addWidget(QLabel("Random Forest Parameters:"), 9, 0, 1, 2)
        
        # Number of estimators
        param_layout.addWidget(QLabel("Number of Trees:"), 10, 0)
        self.n_estimators_spin = QSpinBox()
        self.n_estimators_spin.setRange(10, 500)
        self.n_estimators_spin.setValue(self.n_estimators)
        self.n_estimators_spin.setSingleStep(10)
        param_layout.addWidget(self.n_estimators_spin, 10, 1)
        
        # Max depth
        param_layout.addWidget(QLabel("Max Tree Depth:"), 11, 0)
        self.max_depth_spin = QSpinBox()
        self.max_depth_spin.setRange(3, 30)
        self.max_depth_spin.setValue(self.max_depth)
        self.max_depth_spin.setSingleStep(1)
        param_layout.addWidget(self.max_depth_spin, 11, 1)
        
        setup_layout.addWidget(param_group)
        
        # Extended Readout Settings (Advanced - at bottom to preserve layout)
        extended_group = QGroupBox("Extended Readout Settings (Advanced)")
        extended_layout = QGridLayout()
        extended_group.setLayout(extended_layout)
        
        # Use state taps checkbox
        extended_layout.addWidget(QLabel("Use State Taps:"), 0, 0)
        self.use_state_taps_check = QCheckBox()
        self.use_state_taps_check.setChecked(False)  # Default OFF - preserves baseline
        extended_layout.addWidget(self.use_state_taps_check, 0, 1)
        
        # Number of state taps
        extended_layout.addWidget(QLabel("Number of State Taps (L):"), 1, 0)
        self.n_state_taps_spin = QSpinBox()
        self.n_state_taps_spin.setRange(1, 50)
        self.n_state_taps_spin.setValue(self.n_state_taps)
        self.n_state_taps_spin.setSingleStep(1)
        extended_layout.addWidget(self.n_state_taps_spin, 1, 1)
        
        # Input voltage range
        extended_layout.addWidget(QLabel("Input Voltage Min (V):"), 2, 0)
        self.input_v_min_spin = QDoubleSpinBox()
        self.input_v_min_spin.setRange(0.0, 5.0)
        self.input_v_min_spin.setValue(self.input_v_min)
        self.input_v_min_spin.setSingleStep(0.1)
        self.input_v_min_spin.setDecimals(1)
        extended_layout.addWidget(self.input_v_min_spin, 2, 1)
        
        extended_layout.addWidget(QLabel("Input Voltage Max (V):"), 3, 0)
        self.input_v_max_spin = QDoubleSpinBox()
        self.input_v_max_spin.setRange(0.0, 5.0)
        self.input_v_max_spin.setValue(self.input_v_max)
        self.input_v_max_spin.setSingleStep(0.1)
        self.input_v_max_spin.setDecimals(1)
        extended_layout.addWidget(self.input_v_max_spin, 3, 1)
        
        setup_layout.addWidget(extended_group)
        
        # Run controls
        run_group = QGroupBox("Experiment Controls")
        run_layout = QHBoxLayout()
        run_group.setLayout(run_layout)
        
        self.train_btn = QPushButton("Train Model")
        self.train_btn.clicked.connect(self.start_training)
        run_layout.addWidget(self.train_btn)
        
        self.test_btn = QPushButton("Test Model")
        self.test_btn.clicked.connect(self.start_testing)
        self.test_btn.setEnabled(False)
        run_layout.addWidget(self.test_btn)
        
        self.save_btn = QPushButton("Save Model")
        self.save_btn.clicked.connect(self.save_model)
        self.save_btn.setEnabled(False)
        run_layout.addWidget(self.save_btn)
        
        self.load_btn = QPushButton("Load Model")
        self.load_btn.clicked.connect(self.load_model)
        run_layout.addWidget(self.load_btn)
        
        setup_layout.addWidget(run_group)
        
        # Progress group
        progress_group = QGroupBox("Progress")
        progress_layout = QVBoxLayout()
        progress_group.setLayout(progress_layout)
        
        self.status_label = QLabel("Ready")
        progress_layout.addWidget(self.status_label)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        progress_layout.addWidget(self.progress_bar)
        
        self.rmse_label = QLabel("RMSE: N/A")
        progress_layout.addWidget(self.rmse_label)
        
        setup_layout.addWidget(progress_group)
        
        # Monitoring tab
        monitor_tab = QWidget()
        monitor_layout = QVBoxLayout(monitor_tab)
        tabs.addTab(monitor_tab, "Monitoring")
        
        # Add plotting canvas
        self.plot_canvas = NarmaCanvas(width=8, height=6)
        monitor_layout.addWidget(self.plot_canvas)
        
        # Add monitoring controls
        monitor_control_layout = QHBoxLayout()
        
        self.monitor_btn = QPushButton("Start Monitoring")
        self.monitor_btn.clicked.connect(self.toggle_monitoring)
        monitor_control_layout.addWidget(self.monitor_btn)
        
        self.clear_btn = QPushButton("Clear Plots")
        self.clear_btn.clicked.connect(self.clear_plots)
        monitor_control_layout.addWidget(self.clear_btn)
        
        monitor_layout.addLayout(monitor_control_layout) 

        # Analysis tab
        analysis_tab = QWidget()
        analysis_layout = QVBoxLayout(analysis_tab)
        tabs.addTab(analysis_tab, "Analysis")

        # Add matplotlib figures for analysis
        self.analysis_fig = plt.Figure(figsize=(12, 8))
        self.analysis_canvas = FigureCanvas(self.analysis_fig)
        analysis_layout.addWidget(self.analysis_canvas)

        # Save analysis button
        self.save_analysis_btn = QPushButton("Save Analysis to PDF")
        self.save_analysis_btn.clicked.connect(self.save_analysis_pdf)
        analysis_layout.addWidget(self.save_analysis_btn)

    def setup_digilent(self):
        """Initialize Digilent devices with retry"""
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
                            self.hdwf1 = c_int()  # First device for Red
                            self.hdwf2 = c_int()  # Second device for Green and Blue
                            
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
                            
                            # Configure first device
                            self.dwf.FDwfDeviceAutoConfigureSet(self.hdwf1, c_int(1))
                            self.dwf.FDwfAnalogOutNodeEnableSet(self.hdwf1, c_int(0), c_int(0), c_int(1))
                            self.dwf.FDwfAnalogOutNodeFunctionSet(self.hdwf1, c_int(0), c_int(0), c_int(1))
                            
                            # Setup analog input for first device
                            self.dwf.FDwfAnalogInChannelEnableSet(self.hdwf1, c_int(0), c_int(1))  # Enable channel 1
                            self.dwf.FDwfAnalogInChannelRangeSet(self.hdwf1, c_int(0), c_double(5))  # Set range to 5V
                            self.dwf.FDwfAnalogInConfigure(self.hdwf1, c_int(0), c_int(0))
                            
                            # Configure second device
                            self.dwf.FDwfDeviceAutoConfigureSet(self.hdwf2, c_int(1))
                            self.dwf.FDwfAnalogOutNodeEnableSet(self.hdwf2, c_int(0), c_int(0), c_int(1))  # W1
                            self.dwf.FDwfAnalogOutNodeEnableSet(self.hdwf2, c_int(1), c_int(0), c_int(1))  # W2
                            self.dwf.FDwfAnalogOutNodeFunctionSet(self.hdwf2, c_int(0), c_int(0), c_int(1))
                            self.dwf.FDwfAnalogOutNodeFunctionSet(self.hdwf2, c_int(1), c_int(0), c_int(1))
                            
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
        
        # Update UI status
        self.status_label.setText("Devices initialized successfully")
    
    def validate_channels(self):
        """Ensure at least one channel is always selected"""
        if not (self.channel_1_check.isChecked() or 
                self.channel_2_check.isChecked() or 
                self.channel_3_check.isChecked()):
            # If no channels selected, re-enable the one that was just unchecked
            sender = self.sender()
            sender.setChecked(True)
            self.status_label.setText("Error: At least one channel must be selected")
        else:
            # Update status with current channel configuration
            active_channels = []
            if self.channel_1_check.isChecked():
                active_channels.append("Ch1")
            if self.channel_2_check.isChecked():
                active_channels.append("Ch2")
            if self.channel_3_check.isChecked():
                active_channels.append("Ch3")
            self.status_label.setText(f"Active channels: {', '.join(active_channels)}")
        
    def get_error_message(self):
        """Get the last error message from the device"""
        error_msg = create_string_buffer(512)
        self.dwf.FDwfGetLastErrorMsg(error_msg)
        return error_msg.value.decode()
    
    def build_state_tap_matrix(self, x: np.ndarray, L: int) -> np.ndarray:
        """
        Build tapped delay matrix from state vector.
        
        Args:
            x: State vector of shape (T,)
            L: Number of taps (returns L+1 columns: x(t), x(t-1), ..., x(t-L))
        
        Returns:
            Matrix of shape (T-L, L+1) with columns [x(t), x(t-1), ..., x(t-L)]
        """
        T = len(x)
        if T <= L:
            raise ValueError(f"State vector length {T} must be > {L} taps")
        
        # Create matrix with tapped delays
        X_taps = np.zeros((T - L, L + 1))
        for i in range(L + 1):
            X_taps[:, i] = x[L - i:T - i]
        
        return X_taps
        
    def generate_narma10_data(self):
        """Generate NARMA-10 dataset"""
        # Update parameters from UI
        self.train_len = self.train_spin.value()
        self.test_len = self.test_spin.value()
        self.washout = self.washout_spin.value()
        self.alpha = self.alpha_spin.value()
        self.delay_ms = self.delay_spin.value()
        
        # Create random uniform input between 0 and 0.5
        u = np.random.uniform(0, 0.5, self.train_len + self.test_len)
        
        # Initialize target array
        y = np.zeros(u.shape)
        
        # Generate NARMA-10 sequence
        for t in range(self.order, len(u)):
            sum_term = 0
            for i in range(self.order):
                sum_term += y[t-1-i]
            
            y[t] = 0.3 * y[t-1] + 0.05 * y[t-1] * sum_term + \
                   1.5 * u[t-1] * u[t-10] + 0.1
        
        return u, y
    
    def scale_to_voltage(self, value, min_val=0, max_val=0.5):
        """Scale normalized values to voltage range using configurable min/max"""
        return self.input_v_min + (value - min_val) * (self.input_v_max - self.input_v_min) / (max_val - min_val)
    
    def set_input_voltages(self, input_value):
        """Set selected channels to represent NARMA input value"""
        if not hasattr(self, 'dwf') or not self.dwf:
            print("Devices not initialized")
            return False, (0, 0, 0)
            
        try:
            # Scale input to voltage range
            v = self.scale_to_voltage(input_value)
            
            # Get active channels
            ch1_active = self.channel_1_check.isChecked()
            ch2_active = self.channel_2_check.isChecked()
            ch3_active = self.channel_3_check.isChecked()
            
            # Set voltages based on channel selection
            # Active channels get the signal voltage, inactive channels get 0V
            ch1_voltage = v if ch1_active else 0.0
            ch2_voltage = v if ch2_active else 0.0
            ch3_voltage = v if ch3_active else 0.0
            
            # Set channel voltages
            self.dwf.FDwfAnalogOutNodeOffsetSet(self.hdwf1, c_int(0), c_int(0), c_double(ch1_voltage))  # Ch1
            self.dwf.FDwfAnalogOutNodeOffsetSet(self.hdwf1, c_int(1), c_int(0), c_double(ch2_voltage))  # Ch2 
            self.dwf.FDwfAnalogOutNodeOffsetSet(self.hdwf2, c_int(0), c_int(0), c_double(ch3_voltage))  # Ch3
            
            # Calculate expected summed output for monitoring
            num_active = sum([ch1_active, ch2_active, ch3_active])
            expected_sum = v * num_active
            
            # Verify settings on first active channel
            actual_v = c_double()
            if ch1_active:
                self.dwf.FDwfAnalogOutNodeOffsetGet(self.hdwf1, c_int(0), c_int(0), byref(actual_v))
            elif ch2_active:
                self.dwf.FDwfAnalogOutNodeOffsetGet(self.hdwf1, c_int(1), c_int(0), byref(actual_v))
            elif ch3_active:
                self.dwf.FDwfAnalogOutNodeOffsetGet(self.hdwf2, c_int(0), c_int(0), byref(actual_v))
            
            return True, expected_sum
            
        except Exception as e:
            print(f"Error setting voltages: {e}")
            return False, 0
    
    def read_voltage(self):
        """Read voltage with averaging"""
        if not hasattr(self, 'dwf') or not self.dwf:
            return None
        
        try:
            v_avg = 0
            num_samples = 5  # Average 5 readings
            
            for _ in range(num_samples):
                self.dwf.FDwfAnalogInStatus(self.hdwf1, c_int(False), None)
                voltage = c_double()
                self.dwf.FDwfAnalogInStatusSample(self.hdwf1, c_int(0), byref(voltage))
                v_avg += voltage.value
                time.sleep(0.01)  # Small delay between readings
            
            v_avg = v_avg / num_samples
            return v_avg
            
        except Exception as e:
            print(f"Error reading voltage: {e}")
            return None
            
    def update_params_from_ui(self):
        """Update parameters from UI controls"""
        self.train_len = self.train_spin.value()
        self.test_len = self.test_spin.value()
        self.washout = self.washout_spin.value()
        self.alpha = self.alpha_spin.value()
        self.delay_ms = self.delay_spin.value()
        self.n_estimators = self.n_estimators_spin.value()
        self.max_depth = self.max_depth_spin.value()
        # Ridge extra params
        self.ridge_fit_intercept = self.ridge_fit_intercept_check.isChecked()
        self.ridge_normalize = self.ridge_normalize_check.isChecked()
        self.ridge_solver = self.ridge_solver_combo.currentText()
        # Extended readout params
        self.use_state_taps = self.use_state_taps_check.isChecked()
        self.n_state_taps = self.n_state_taps_spin.value()
        self.input_v_min = self.input_v_min_spin.value()
        self.input_v_max = self.input_v_max_spin.value()
        
    def start_training(self):
        """Begin NARMA-10 training"""
        self.update_params_from_ui()
        
        # Generate data
        self.all_inputs, self.all_targets = self.generate_narma10_data()
        self.train_inputs = self.all_inputs[:self.train_len]
        self.train_targets = self.all_targets[:self.train_len]
        
        # Reset data arrays
        self.inputs = []
        self.states = []
        self.targets = []
        self.predictions = []
        
        # Setup progress tracking
        self.current_step = 0
        self.max_steps = self.train_len
        self.progress_bar.setRange(0, self.max_steps)
        self.progress_bar.setValue(0)
        
        # Update UI
        self.status_label.setText("Training started...")
        self.train_btn.setEnabled(False)
        self.test_btn.setEnabled(False)
        self.save_btn.setEnabled(False)
        self.load_btn.setEnabled(False)
        
        # Start the training process
        self.is_training = True
        self.is_testing = False
        
        # Start timer for training steps
        self.monitor_timer.start()
    
    def process_training_step(self):
        """Process a single training step"""
        if self.current_step >= self.train_len:
            self.finish_training()
            return
            
        # Get current input
        input_val = self.train_inputs[self.current_step]
        
        # Set input voltages
        success, v_set = self.set_input_voltages(input_val)
        if not success:
            self.status_label.setText(f"Error setting voltages at step {self.current_step}")
            return
            
        # Allow system to settle
        QApplication.processEvents()
        time.sleep(self.delay_ms / 1000.0)
        
        # Read reservoir state
        state = self.read_voltage()
        if state is None:
            self.status_label.setText(f"Error reading voltage at step {self.current_step}")
            return
            
        # Store data
        self.inputs.append(input_val)
        self.states.append(state)
        self.targets.append(self.train_targets[self.current_step])
        self.predictions.append(0)  # No prediction during training
        
        # Update progress
        self.current_step += 1
        self.progress_bar.setValue(self.current_step)
        self.status_label.setText(f"Training: {self.current_step}/{self.train_len}")
        
        # Update plots
        if self.current_step % 10 == 0:  # Update every 10 steps to avoid slowing down
            self.plot_canvas.update_plots(
                self.inputs, 
                self.states, 
                self.targets, 
                self.predictions
            )
    
    def finish_training(self):
        """Complete the training process"""
        self.is_training = False
        self.monitor_timer.stop()
        
        # Calculate effective start considering both washout and taps
        # ALIGNMENT LOGIC: effective_start ensures consistent slicing across X_train, y_train, X_test, y_test
        # - Baseline mode (use_state_taps=False): effective_start = washout (no extra offset)
        # - Tapped mode (use_state_taps=True): effective_start = max(washout, n_state_taps)
        #   This ensures we drop enough samples for taps AND washout, maintaining alignment
        effective_start = max(self.washout, self.n_state_taps if self.use_state_taps else 0)
        
        # Get states and targets after effective_start
        states_full = np.array(self.states)
        targets_full = np.array(self.targets)
        
        if len(states_full) <= effective_start:
            self.status_label.setText(f"Error: Not enough data (need > {effective_start} samples)")
            self.train_btn.setEnabled(True)
            self.load_btn.setEnabled(True)
            return
        
        # Build features based on mode
        if self.use_state_taps:
            # Extended readout: use tapped delays
            # ALIGNMENT: build_state_tap_matrix returns shape (T-L, L+1) where row i corresponds to time i+L
            # So X_taps[0] uses states[L:L+1], X_taps[1] uses states[L+1:L+2], etc.
            # Therefore targets must start at index n_state_taps to align: y_train = targets[n_state_taps:]
            X_taps = self.build_state_tap_matrix(states_full, self.n_state_taps)
            # Targets align with taps (drop first n_state_taps samples)
            y_train = targets_full[self.n_state_taps:]
            # Apply washout by further slicing if washout > n_state_taps
            if effective_start > self.n_state_taps:
                washout_offset = effective_start - self.n_state_taps
                X_taps = X_taps[washout_offset:]
                y_train = y_train[washout_offset:]
            
            # For now, use linear taps only (can add nonlinear transforms later)
            X_features = X_taps  # Shape: (T-L-washout_offset, L+1)
        else:
            # Baseline readout: original behavior (EXACTLY matches original code when use_state_taps=False)
            # ALIGNMENT: When use_state_taps=False, effective_start = washout, so this is identical to original:
            #   Original: X_train = self.states[self.washout:], y_train = self.targets[self.washout:]
            #   This code: X_train = states_full[washout:], y_train = targets_full[washout:]
            # Features are built EXACTLY as before: [state, state², sin(3*state), cos(2*state)]
            X_train = states_full[effective_start:]
            y_train = targets_full[effective_start:]
            
            # Create features with nonlinear transformations (baseline)
            X_features = np.column_stack([
                X_train.reshape(-1, 1),                  # Original state
                X_train.reshape(-1, 1)**2,               # Squared
                np.sin(X_train.reshape(-1, 1) * 3),      # Sine transform
                np.cos(X_train.reshape(-1, 1) * 2)       # Cosine transform
            ])
        
        # Make sure we have data
        if len(X_features) == 0 or len(y_train) == 0:
            self.status_label.setText("Error: No training data collected")
            self.train_btn.setEnabled(True)
            self.load_btn.setEnabled(True)
            return
        
        # Train Ridge Regression model with optional normalization
        if self.ridge_normalize:
            self.ridge_model = make_pipeline(
                StandardScaler(),
                Ridge(
                    alpha=self.alpha,
                    fit_intercept=self.ridge_fit_intercept,
                    solver=self.ridge_solver
                )
            )
        else:
            self.ridge_model = Ridge(
                alpha=self.alpha,
                fit_intercept=self.ridge_fit_intercept,
                solver=self.ridge_solver
            )
        self.ridge_model.fit(X_features, y_train)
        ridge_train_pred = self.ridge_model.predict(X_features)
        self.ridge_train_rmse = np.sqrt(mean_squared_error(y_train, ridge_train_pred))
        
        # Train Random Forest model
        self.rf_model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=42
        )
        self.rf_model.fit(X_features, y_train)
        rf_train_pred = self.rf_model.predict(X_features)
        self.rf_train_rmse = np.sqrt(mean_squared_error(y_train, rf_train_pred))
        
        # Update UI
        self.status_label.setText("Training completed")
        self.rmse_label.setText(f"Train RMSE - Ridge: {self.ridge_train_rmse:.6f} | RF: {self.rf_train_rmse:.6f}")
        self.train_btn.setEnabled(True)
        self.test_btn.setEnabled(True)
        self.save_btn.setEnabled(True)
        self.load_btn.setEnabled(True)
        
        print(f"Training completed. Ridge RMSE: {self.ridge_train_rmse:.6f}, RF RMSE: {self.rf_train_rmse:.6f}")
        
        # Save training data
        self.save_training_data()
        
        # Generate training report
        self.generate_training_report()
    
    def start_testing(self):
        """Begin NARMA-10 testing"""
        if not hasattr(self, 'rf_model') or self.rf_model is None or not hasattr(self, 'ridge_model') or self.ridge_model is None:
            self.status_label.setText("Error: No model trained. Train first.")
            return
        
        # Get test data
        self.test_inputs = self.all_inputs[self.train_len:self.train_len+self.test_len]
        self.test_targets = self.all_targets[self.train_len:self.train_len+self.test_len]
        
        # Reset data arrays for testing
        self.inputs = []
        self.states = []
        self.targets = []
        self.rf_predictions = []
        self.ridge_predictions = []
        
        # Setup progress tracking
        self.current_step = 0
        self.max_steps = self.test_len
        self.progress_bar.setRange(0, self.max_steps)
        self.progress_bar.setValue(0)
        
        # Update UI
        self.status_label.setText("Testing started...")
        self.train_btn.setEnabled(False)
        self.test_btn.setEnabled(False)
        self.save_btn.setEnabled(False)
        self.load_btn.setEnabled(False)
        
        # Start the testing process
        self.is_training = False
        self.is_testing = True
        
        # Start timer for testing steps
        self.monitor_timer.start()
    
    def process_testing_step(self):
        """Process a single testing step"""
        if self.current_step >= self.test_len:
            self.finish_testing()
            return
        
        # Get current input
        input_val = self.test_inputs[self.current_step]
        
        # Set input voltages
        success, v_set = self.set_input_voltages(input_val)
        if not success:
            self.status_label.setText(f"Error setting voltages at step {self.current_step}")
            return
        
        # Allow system to settle
        QApplication.processEvents()
        time.sleep(self.delay_ms / 1000.0)
        
        # Read reservoir state
        state = self.read_voltage()
        if state is None:
            self.status_label.setText(f"Error reading voltage at step {self.current_step}")
            return
        
        # Store data (we'll make predictions after collecting all states if using taps)
        self.inputs.append(input_val)
        self.states.append(state)
        self.targets.append(self.test_targets[self.current_step])
        
        # For baseline mode, make prediction immediately
        # For tapped mode, we need to wait until we have enough states
        if not self.use_state_taps:
            # Baseline: create features with nonlinear transformations (same as in training)
            features = np.array([
                state,
                state**2,
                np.sin(state * 3),
                np.cos(state * 2)
            ]).reshape(1, -1)
            
            # Make predictions
            rf_pred = self.rf_model.predict(features)[0]
            ridge_pred = self.ridge_model.predict(features)[0]
            
            self.rf_predictions.append(rf_pred)
            self.ridge_predictions.append(ridge_pred)
        else:
            # Tapped mode: need at least n_state_taps+1 states to make first prediction
            if len(self.states) > self.n_state_taps:
                # Build tapped delay feature
                states_array = np.array(self.states)
                tap_features = self.build_state_tap_matrix(states_array, self.n_state_taps)
                # Use the most recent tap vector
                features = tap_features[-1:].reshape(1, -1)
                
                rf_pred = self.rf_model.predict(features)[0]
                ridge_pred = self.ridge_model.predict(features)[0]
                
                self.rf_predictions.append(rf_pred)
                self.ridge_predictions.append(ridge_pred)
            else:
                # Not enough states yet, append placeholder
                self.rf_predictions.append(0.0)
                self.ridge_predictions.append(0.0)
        
        # Update progress
        self.current_step += 1
        self.progress_bar.setValue(self.current_step)
        self.status_label.setText(f"Testing: {self.current_step}/{self.test_len}")
        
        # Update plots
        if self.current_step % 5 == 0:  # Update more frequently during testing
            self.plot_canvas.update_plots(
                self.inputs, 
                self.states, 
                self.targets, 
                self.rf_predictions  # Show RF by default
            )
    
    def finish_testing(self):
        """Complete the testing process"""
        self.is_testing = False
        self.monitor_timer.stop()
        
        # Calculate effective start considering both washout and taps
        # ALIGNMENT LOGIC: Same effective_start as training ensures predictions[i] aligns with targets[i]
        # - Baseline mode: predictions start at index=washout, align with targets[washout:]
        # - Tapped mode: first n_state_taps predictions are placeholders (0.0), real predictions start at n_state_taps
        #   effective_start = max(washout, n_state_taps) ensures we evaluate only valid aligned pairs
        effective_start = max(self.washout, self.n_state_taps if self.use_state_taps else 0)
        
        # Calculate test metrics only on valid segment (after effective_start)
        if len(self.rf_predictions) > 0 and len(self.targets) > 0 and len(self.ridge_predictions) > 0:
            # Extract valid segment - predictions and targets are aligned at indices >= effective_start
            if effective_start > 0 and effective_start < len(self.targets):
                y_true_valid = np.array(self.targets[effective_start:])
                rf_pred_valid = np.array(self.rf_predictions[effective_start:])
                ridge_pred_valid = np.array(self.ridge_predictions[effective_start:])
            else:
                # No washout/taps needed, use all data
                y_true_valid = np.array(self.targets)
                rf_pred_valid = np.array(self.rf_predictions)
                ridge_pred_valid = np.array(self.ridge_predictions)
            
            # Calculate RMSE
            self.rf_test_rmse = np.sqrt(mean_squared_error(y_true_valid, rf_pred_valid))
            self.ridge_test_rmse = np.sqrt(mean_squared_error(y_true_valid, ridge_pred_valid))
            
            # Calculate NRMSE (normalized by target std)
            target_std = np.std(y_true_valid)
            self.rf_test_nrmse = self.rf_test_rmse / target_std if target_std > 0 else np.inf
            self.ridge_test_nrmse = self.ridge_test_rmse / target_std if target_std > 0 else np.inf
            
            # Calculate R2
            self.rf_test_r2 = r2_score(y_true_valid, rf_pred_valid)
            self.ridge_test_r2 = r2_score(y_true_valid, ridge_pred_valid)
            
            # Calculate baseline predictor RMSE (mean of training target)
            # Use training targets after effective_start for baseline
            train_targets_valid = np.array(self.all_targets[:self.train_len])
            if effective_start < len(train_targets_valid):
                baseline_mean = np.mean(train_targets_valid[effective_start:])
            else:
                baseline_mean = np.mean(train_targets_valid)
            baseline_pred = np.full_like(y_true_valid, baseline_mean)
            self.baseline_test_rmse = np.sqrt(mean_squared_error(y_true_valid, baseline_pred))
            
            # Store metrics for CSV output
            self.test_metrics = {
                'dt': self.delay_ms,
                'washout': self.washout,
                'n_state_taps': self.n_state_taps,
                'use_state_taps': self.use_state_taps,
                'input_v_min': self.input_v_min,
                'input_v_max': self.input_v_max,
                'ridge_rmse': self.ridge_test_rmse,
                'ridge_nrmse': self.ridge_test_nrmse,
                'ridge_r2': self.ridge_test_r2,
                'rf_rmse': self.rf_test_rmse,
                'rf_nrmse': self.rf_test_nrmse,
                'rf_r2': self.rf_test_r2,
                'baseline_rmse': self.baseline_test_rmse
            }
            
            # Update UI
            self.status_label.setText("Testing completed")
            self.rmse_label.setText(
                f"Test RMSE - Ridge: {self.ridge_test_rmse:.6f} | RF: {self.rf_test_rmse:.6f} | "
                f"Baseline: {self.baseline_test_rmse:.6f}"
            )
            
            print(f"Testing completed. Ridge RMSE: {self.ridge_test_rmse:.6f}, RF RMSE: {self.rf_test_rmse:.6f}")
            print(f"Ridge NRMSE: {self.ridge_test_nrmse:.6f}, R2: {self.ridge_test_r2:.6f}")
            print(f"RF NRMSE: {self.rf_test_nrmse:.6f}, R2: {self.rf_test_r2:.6f}")
            print(f"Baseline RMSE: {self.baseline_test_rmse:.6f}")
            
            # Final plot update (show both predictions)
            self.plot_canvas.update_plots(
                self.inputs,
                self.states,
                self.targets,
                self.rf_predictions  # Show RF by default
            )
            
            # Save test results
            self.save_results()
            
            # Generate test report
            self.generate_test_report()
            
            # After test report, update analysis tab
            self.update_analysis_tab()
        else:
            self.status_label.setText("Error: No test data collected")
            # Initialize empty metrics to avoid errors
            if not hasattr(self, 'test_metrics'):
                self.test_metrics = {}
        
        # Re-enable buttons
        self.train_btn.setEnabled(True)
        self.test_btn.setEnabled(True)
        self.save_btn.setEnabled(True)
        self.load_btn.setEnabled(True)
    
    def update_monitoring(self):
        """Update real-time monitoring during training or testing"""
        if self.is_training:
            self.process_training_step()
        elif self.is_testing:
            self.process_testing_step()
    
    def toggle_monitoring(self):
        """Start or stop continuous monitoring of real-time values"""
        if self.monitor_timer.isActive():
            # Stop monitoring
            self.monitor_timer.stop()
            self.monitor_btn.setText("Start Monitoring")
            self.status_label.setText("Monitoring stopped")
        else:
            # Start monitoring
            self.monitor_timer.start(100)  # Update every 100ms
            self.monitor_btn.setText("Stop Monitoring")
            self.status_label.setText("Monitoring active")
    
    def clear_plots(self):
        """Clear all plot data"""
        self.inputs = []
        self.states = []
        self.targets = []
        self.predictions = []
        
        self.plot_canvas.update_plots([], [], [], [])
        
    def save_model(self):
        """Save the trained model"""
        if self.model is None:
            self.status_label.setText("Error: No model to save")
            return
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"narma10_model_{timestamp}.pkl"
        
        model_data = {
            'model': self.model,
            'train_len': self.train_len,
            'test_len': self.test_len,
            'washout': self.washout,
            'alpha': self.alpha,
            'order': self.order,
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'channels': {
                'ch1_active': self.channel_1_check.isChecked(),
                'ch2_active': self.channel_2_check.isChecked(),
                'ch3_active': self.channel_3_check.isChecked()
            }
        }
        
        try:
            with open(filename, 'wb') as f:
                pickle.dump(model_data, f)
            self.status_label.setText(f"Model saved as {filename}")
        except Exception as e:
            self.status_label.setText(f"Error saving model: {str(e)}")
    
    def load_model(self):
        """Load a previously trained model"""
        try:
            # Simple file selection dialog approach
            filename, _ = QFileDialog.getOpenFileName(
                self,
                "Load Model",
                "",
                "Pickle Files (*.pkl)"
            )
            
            if not filename:
                return
                
            with open(filename, 'rb') as f:
                model_data = pickle.load(f)
                
            # Extract model and parameters
            self.model = model_data['model']
            
            # Update UI with loaded parameters
            self.train_spin.setValue(model_data.get('train_len', self.train_len))
            self.test_spin.setValue(model_data.get('test_len', self.test_len))
            self.washout_spin.setValue(model_data.get('washout', self.washout))
            self.alpha_spin.setValue(model_data.get('alpha', self.alpha))
            self.n_estimators = model_data.get('n_estimators', self.n_estimators)
            self.max_depth = model_data.get('max_depth', self.max_depth)
            
            # Update channel checkboxes if saved
            if 'channels' in model_data:
                channels = model_data['channels']
                self.channel_1_check.setChecked(channels.get('ch1_active', True))
                self.channel_2_check.setChecked(channels.get('ch2_active', True))
                self.channel_3_check.setChecked(channels.get('ch3_active', True))
            
            # Enable testing
            self.test_btn.setEnabled(True)
            self.save_btn.setEnabled(True)
            
            self.status_label.setText(f"Model loaded from {filename}")
            
        except Exception as e:
            self.status_label.setText(f"Error loading model: {str(e)}")
            print(f"Error loading model: {str(e)}")
    
    def save_results(self):
        """Save testing results to CSV with metadata"""
        if len(self.inputs) == 0 or len(self.rf_predictions) == 0 or len(self.ridge_predictions) == 0:
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Generate filename based on mode
        if self.use_state_taps:
            filename = f"narma10_results_{timestamp}_tapsL{self.n_state_taps}.csv"
        else:
            filename = f"narma10_results_{timestamp}_baseline.csv"
        
        try:
            # Prepare data array
            data = np.column_stack((
                self.inputs,
                self.states,
                self.targets,
                self.rf_predictions,
                self.ridge_predictions
            ))
            
            # Write file with metadata header if available
            with open(filename, 'w') as f:
                # Write metadata as comments if available
                if hasattr(self, 'test_metrics'):
                    metrics = self.test_metrics
                    metadata_lines = [
                        f"# NARMA-10 Test Results",
                        f"# Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                        f"# dt (ms): {metrics['dt']}",
                        f"# washout: {metrics['washout']}",
                        f"# n_state_taps: {metrics['n_state_taps']}",
                        f"# use_state_taps: {metrics['use_state_taps']}",
                        f"# input_v_min: {metrics['input_v_min']}",
                        f"# input_v_max: {metrics['input_v_max']}",
                        f"# Ridge RMSE: {metrics['ridge_rmse']:.6f}",
                        f"# Ridge NRMSE: {metrics['ridge_nrmse']:.6f}",
                        f"# Ridge R2: {metrics['ridge_r2']:.6f}",
                        f"# RF RMSE: {metrics['rf_rmse']:.6f}",
                        f"# RF NRMSE: {metrics['rf_nrmse']:.6f}",
                        f"# RF R2: {metrics['rf_r2']:.6f}",
                        f"# Baseline RMSE: {metrics['baseline_rmse']:.6f}",
                    ]
                    for line in metadata_lines:
                        f.write(line + '\n')
                
                # Write header
                f.write('input,state,target,rf_prediction,ridge_prediction\n')
                
                # Write data
                np.savetxt(f, data, delimiter=',', fmt='%.6f')
            
            print(f"Results saved to {filename}")
        except Exception as e:
            print(f"Error saving results: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def save_training_data(self):
        """Save training data to CSV"""
        if len(self.inputs) == 0:
            return
                
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"narma10_training_{timestamp}.csv"
        
        try:
            data = np.column_stack((
                self.inputs,
                self.states,
                self.targets
            ))
            
            np.savetxt(
                filename, 
                data, 
                delimiter=',', 
                header='input,state,target',
                comments=''
            )
            
            print(f"Training data saved to {filename}")
        except Exception as e:
            print(f"Error saving training data: {str(e)}")
    
    def generate_training_report(self):
        """Generate a PDF report for training results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"narma10_training_report_{timestamp}.pdf"
        
        try:
            with PdfPages(filename) as pdf:
                # Create a figure for the report
                fig = Figure(figsize=(8.5, 11))
                
                # Add a title
                fig.suptitle("NARMA-10 Training Report", fontsize=16)
                
                # Add configuration information
                ax_config = fig.add_subplot(311)
                ax_config.axis('off')
                # Get active channels info
                active_channels = []
                if self.channel_1_check.isChecked():
                    active_channels.append("Ch1")
                if self.channel_2_check.isChecked():
                    active_channels.append("Ch2")
                if self.channel_3_check.isChecked():
                    active_channels.append("Ch3")
                channels_str = ", ".join(active_channels)
                
                # Reservoir state statistics
                states_arr = np.array(self.states)
                state_min = np.min(states_arr) if len(states_arr) > 0 else float('nan')
                state_max = np.max(states_arr) if len(states_arr) > 0 else float('nan')
                state_mean = np.mean(states_arr) if len(states_arr) > 0 else float('nan')
                state_median = np.median(states_arr) if len(states_arr) > 0 else float('nan')
                state_std = np.std(states_arr) if len(states_arr) > 0 else float('nan')
                
                config_text = (
                    f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                    f"NARMA-10 Parameters:\n"
                    f"- Training Samples: {self.train_len}\n"
                    f"- Washout Period: {self.washout}\n"
                    f"- Sample Delay: {self.delay_ms} ms\n"
                    f"- Active Channels: {channels_str} ({len(active_channels)} channels)\n\n"
                    f"Random Forest Parameters:\n"
                    f"- Number of Trees: {self.n_estimators}\n"
                    f"- Max Tree Depth: {self.max_depth}\n\n"
                    f"Reservoir State Voltage Statistics:\n"
                    f"- Min: {state_min:.4f} V\n"
                    f"- Max: {state_max:.4f} V\n"
                    f"- Mean: {state_mean:.4f} V\n"
                    f"- Median: {state_median:.4f} V\n"
                    f"- Std Dev: {state_std:.4f} V\n\n"
                    f"Results:\n"
                    f"- Training RMSE (Ridge): {self.ridge_train_rmse:.6f}\n"
                    f"- Training RMSE (RF): {self.rf_train_rmse:.6f}\n"
                )
                ax_config.text(0.05, 0.95, config_text, transform=ax_config.transAxes, 
                               verticalalignment='top', fontsize=10)
                
                # Add training data plots
                ax_data = fig.add_subplot(312)
                t = np.arange(len(self.inputs))
                ax_data.plot(t, self.inputs, 'b-', label='Input')
                ax_data.plot(t, self.targets, 'r-', label='Target')
                ax_data.set_xlabel('Time Step')
                ax_data.set_ylabel('Value')
                ax_data.set_title('Training Input and Target')
                ax_data.legend()
                ax_data.grid(True)
                
                # Add state plot
                ax_state = fig.add_subplot(313)
                ax_state.plot(t, self.states, 'g-', label='Reservoir State')
                ax_state.set_xlabel('Time Step')
                ax_state.set_ylabel('Voltage (V)')
                ax_state.set_title('Reservoir State (Voltage)')
                ax_state.grid(True)
                
                # Adjust layout and save
                fig.tight_layout(rect=[0, 0, 1, 0.95])
                pdf.savefig(fig)
                
                # Add feature importance page if model exists
                if hasattr(self, 'rf_model') and self.rf_model is not None:
                    fig_feat = Figure(figsize=(8.5, 11))
                    fig_feat.suptitle("Feature Importance Analysis (Random Forest)", fontsize=16)
                    
                    # Feature importance
                    ax_feat = fig_feat.add_subplot(111)
                    feature_names = ['State', 'State²', 'Sin(State)', 'Cos(State)']
                    importances = self.rf_model.feature_importances_
                    indices = np.argsort(importances)[::-1]
                    
                    ax_feat.bar(range(len(importances)), importances[indices])
                    ax_feat.set_xticks(range(len(importances)))
                    ax_feat.set_xticklabels([feature_names[i] for i in indices])
                    ax_feat.set_title('Feature Importances (RF)')
                    ax_feat.set_ylabel('Importance')
                    fig_feat.tight_layout(rect=[0, 0, 1, 0.95])
                    
                    pdf.savefig(fig_feat)
            
            print(f"Training report saved to {filename}")
            self.status_label.setText(f"Training report saved to {filename}")
            
        except Exception as e:
            print(f"Error generating training report: {str(e)}")
    
    def generate_test_report(self):
        """Generate a PDF report for test results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"narma10_test_report_{timestamp}.pdf"
        
        try:
            with PdfPages(filename) as pdf:
                # Create a figure for the report
                fig = Figure(figsize=(8.5, 11))
                
                # Add a title
                fig.suptitle("NARMA-10 Test Report", fontsize=16)
                
                # Add configuration information
                ax_config = fig.add_subplot(411)
                ax_config.axis('off')
                # Get active channels info
                active_channels = []
                if self.channel_1_check.isChecked():
                    active_channels.append("Ch1")
                if self.channel_2_check.isChecked():
                    active_channels.append("Ch2")
                if self.channel_3_check.isChecked():
                    active_channels.append("Ch3")
                channels_str = ", ".join(active_channels)
                
                config_text = (
                    f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                    f"NARMA-10 Parameters:\n"
                    f"- Training Samples: {self.train_len}\n"
                    f"- Testing Samples: {self.test_len}\n"
                    f"- Washout Period: {self.washout}\n"
                    f"- Sample Delay: {self.delay_ms} ms\n"
                    f"- Active Channels: {channels_str} ({len(active_channels)} channels)\n\n"
                    f"Random Forest Parameters:\n"
                    f"- Number of Trees: {self.n_estimators}\n"
                    f"- Max Tree Depth: {self.max_depth}\n\n"
                    f"Results:\n"
                    f"- Test RMSE (Ridge): {self.ridge_test_rmse:.6f}\n"
                    f"- Test RMSE (RF): {self.rf_test_rmse:.6f}\n"
                )
                ax_config.text(0.05, 0.95, config_text, transform=ax_config.transAxes, 
                              verticalalignment='top', fontsize=10)
                
                # Add test data plots
                t = np.arange(len(self.inputs))
                
                # Input plot
                ax_input = fig.add_subplot(412)
                ax_input.plot(t, self.inputs, 'b-', label='Input')
                ax_input.set_xlabel('Time Step')
                ax_input.set_ylabel('Input Value')
                ax_input.set_title('Test Input')
                ax_input.grid(True)
                
                # State plot
                ax_state = fig.add_subplot(413)
                ax_state.plot(t, self.states, 'g-', label='Reservoir State')
                ax_state.set_xlabel('Time Step')
                ax_state.set_ylabel('Voltage (V)')
                ax_state.set_title('Reservoir State (Voltage)')
                ax_state.grid(True)
                
                # Target vs Prediction plot
                ax_pred = fig.add_subplot(414)
                ax_pred.plot(t, self.targets, 'r-', label='Target')
                ax_pred.plot(t, self.rf_predictions, 'c--', label='RF Prediction')
                ax_pred.plot(t, self.ridge_predictions, 'm-.', label='Ridge Prediction')
                ax_pred.set_xlabel('Time Step')
                ax_pred.set_ylabel('Value')
                ax_pred.set_title('Target vs Predictions')
                ax_pred.legend()
                ax_pred.grid(True)
                
                # Adjust layout and save
                fig.tight_layout(rect=[0, 0, 1, 0.95])
                pdf.savefig(fig)
                
                # Add error analysis page
                fig_error = Figure(figsize=(8.5, 11))
                fig_error.suptitle("Error Analysis", fontsize=16)
                
                # Error distribution
                ax_err = fig_error.add_subplot(211)
                errors_rf = np.array(self.targets) - np.array(self.rf_predictions)
                errors_ridge = np.array(self.targets) - np.array(self.ridge_predictions)
                ax_err.hist(errors_rf, bins=20, alpha=0.5, label='RF Error')
                ax_err.hist(errors_ridge, bins=20, alpha=0.5, label='Ridge Error')
                ax_err.set_xlabel('Error')
                ax_err.set_ylabel('Frequency')
                ax_err.set_title('Error Distribution')
                ax_err.legend()
                
                # Error vs time
                ax_err_time = fig_error.add_subplot(212)
                ax_err_time.plot(t, np.abs(errors_rf), 'c-', label='RF Abs Error')
                ax_err_time.plot(t, np.abs(errors_ridge), 'm-', label='Ridge Abs Error')
                ax_err_time.set_xlabel('Time Step')
                ax_err_time.set_ylabel('Absolute Error')
                ax_err_time.set_title('Absolute Error vs Time Step')
                ax_err_time.legend()
                ax_err_time.grid(True)
                
                fig_error.tight_layout(rect=[0, 0, 1, 0.95])
                pdf.savefig(fig_error)
            
            print(f"Test report saved to {filename}")
            self.status_label.setText(f"Test report saved to {filename}")
            
        except Exception as e:
            print(f"Error generating test report: {str(e)}")
    
    def update_analysis_tab(self):
        """Update PCA, t-SNE, learning curve, RMSE bar, and confusion matrix plots after testing"""
        if not hasattr(self, 'rf_predictions') or not hasattr(self, 'ridge_predictions') or len(self.rf_predictions) == 0:
            return
        import matplotlib.pyplot as plt
        self.analysis_fig.clf()
        axs = self.analysis_fig.subplots(2, 3)
        # Prepare features
        X = np.column_stack([
            np.array(self.states).reshape(-1, 1),
            np.array(self.states).reshape(-1, 1) ** 2,
            np.sin(np.array(self.states).reshape(-1, 1) * 3),
            np.cos(np.array(self.states).reshape(-1, 1) * 2)
        ])
        y_true = np.array(self.targets)
        y_rf = np.array(self.rf_predictions)
        y_ridge = np.array(self.ridge_predictions)
        # PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        axs[0, 0].scatter(X_pca[:, 0], X_pca[:, 1], c=y_true, cmap='viridis', alpha=0.7)
        axs[0, 0].set_title('PCA (colored by target)')
        # t-SNE
        try:
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X)//4))
            X_tsne = tsne.fit_transform(X)
            axs[0, 1].scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_true, cmap='viridis', alpha=0.7)
            axs[0, 1].set_title('t-SNE (colored by target)')
        except Exception as e:
            axs[0, 1].text(0.5, 0.5, f't-SNE failed: {e}', ha='center')
        # Learning curve (Ridge)
        from sklearn.linear_model import Ridge
        from sklearn.model_selection import learning_curve
        train_sizes, train_scores, test_scores = learning_curve(
            Ridge(alpha=self.alpha, fit_intercept=self.ridge_fit_intercept, solver=self.ridge_solver),
            X, y_true, cv=3, scoring='neg_root_mean_squared_error', train_sizes=np.linspace(0.1, 1.0, 5)
        )
        axs[0, 2].plot(train_sizes, -np.mean(train_scores, axis=1), 'o-', label='Train RMSE')
        axs[0, 2].plot(train_sizes, -np.mean(test_scores, axis=1), 's-', label='Test RMSE')
        axs[0, 2].set_title('Learning Curve (Ridge)')
        axs[0, 2].set_xlabel('Training Size')
        axs[0, 2].set_ylabel('RMSE')
        axs[0, 2].legend()
        # RMSE bar plot
        axs[1, 0].bar(['Ridge', 'RF'], [self.ridge_test_rmse, self.rf_test_rmse], color=['m', 'c'])
        axs[1, 0].set_title('Test RMSE Comparison')
        axs[1, 0].set_ylabel('RMSE')
        # Regression confusion matrix (2D hist)
        h = axs[1, 1].hist2d(y_true, y_rf, bins=30, cmap='Blues')
        axs[1, 1].set_xlabel('True')
        axs[1, 1].set_ylabel('RF Predicted')
        axs[1, 1].set_title('RF: True vs Predicted')
        self.analysis_fig.colorbar(h[3], ax=axs[1, 1])
        h2 = axs[1, 2].hist2d(y_true, y_ridge, bins=30, cmap='Purples')
        axs[1, 2].set_xlabel('True')
        axs[1, 2].set_ylabel('Ridge Predicted')
        axs[1, 2].set_title('Ridge: True vs Predicted')
        self.analysis_fig.colorbar(h2[3], ax=axs[1, 2])
        self.analysis_fig.tight_layout()
        self.analysis_canvas.draw()

    def save_analysis_pdf(self):
        """Save all analysis plots to a PDF file"""
        from matplotlib.backends.backend_pdf import PdfPages
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"narma10_analysis_{timestamp}.pdf"
        with PdfPages(filename) as pdf:
            self.update_analysis_tab()  # Ensure latest plots
            pdf.savefig(self.analysis_fig)
        self.status_label.setText(f"Analysis PDF saved: {filename}")
        print(f"Analysis PDF saved: {filename}")

    def closeEvent(self, event):
        """Clean up when closing the application"""
        print("\nClosing application and cleaning up devices...")
        if hasattr(self, 'dwf') and self.dwf:
            if hasattr(self, 'hdwf1'):
                print("Closing device 1")
                self.dwf.FDwfDeviceClose(self.hdwf1)
            if hasattr(self, 'hdwf2'):
                print("Closing device 2")
                self.dwf.FDwfDeviceClose(self.hdwf2)
        event.accept()

if __name__ == "__main__":
    try:
        app = QApplication(sys.argv)
        window = MyceliumNarma10App()
        window.show()
        sys.exit(app.exec())
    except Exception as e:
        print(f"Error starting application: {str(e)}")
        import traceback
        traceback.print_exc() 