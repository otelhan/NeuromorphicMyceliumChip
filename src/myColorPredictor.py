import sys
from PyQt6.QtWidgets import (QApplication, QMainWindow, QPushButton, QVBoxLayout, 
                            QLabel, QWidget, QGroupBox, QSlider)
from PyQt6.QtGui import QColor, QPainter
from PyQt6.QtCore import Qt, QRect, QTimer
import numpy as np
import os
from color_classifier import load_saved_model
import random
from datetime import datetime
import ctypes
from ctypes import c_int, c_uint, c_double, byref, create_string_buffer
import time
from ctypes.util import find_library
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from collections import deque
import matplotlib.pyplot as plt
import yaml
import joblib

# Add these constants
RTLD_GLOBAL = 0x00100
RTLD_NOW = 0x00002

class ColorDisplay(QWidget):
    def __init__(self):
        super().__init__()
        self.color = QColor(0, 0, 0)
        self.setMinimumSize(200, 200)
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(QRect(0, 0, self.width(), self.height()), self.color)
        
    def setColor(self, r, g, b):
        self.color = QColor(r, g, b)
        self.update()

class ColorPredictorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Color Predictor")
        self.setMinimumSize(300, 500)
        
        # Load the trained model
        print("Loading classifier model...")
        latest_config = self.get_latest_config('models', prefix='voltage_config_')  # Look for voltage model
        if latest_config:
            self.classifier, self.scaler = self.load_voltage_model(latest_config)
            print("Model loaded successfully!")
        else:
            print("No voltage model found! Please train the model first.")
            return

        # Initialize Digilent
        self.setup_digilent()
        
        # Current RGB values
        self.current_rgb = [0, 0, 0]
        
        # Add data storage for plotting
        self.time_data = deque(maxlen=100)  # Store last 100 points
        self.red_data = deque(maxlen=100)
        self.green_data = deque(maxlen=100)
        self.blue_data = deque(maxlen=100)
        self.output_data = deque(maxlen=100)
        self.start_time = time.time()
        
        # Add timer for continuous monitoring
        self.monitor_timer = QTimer()
        self.monitor_timer.timeout.connect(self.update_monitoring)
        self.monitor_timer.setInterval(100)  # Update every 100ms
        self.monitoring_active = False
        
        self.setup_ui()

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
                
                # Rest of the setup code...
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
                            self.dwf = ctypes.cdll.LoadLibrary(framework_path)
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
                            print("\nOpening first device for Red control...")
                            result1 = self.dwf.FDwfDeviceOpen(c_int(0), byref(self.hdwf1))
                            if self.hdwf1.value == 0:
                                error_msg = self.get_error_message()
                                raise Exception(f"Failed to open first device: {error_msg}")
                            
                            # Open and configure second device
                            print("\nOpening second device for Green and Blue control...")
                            result2 = self.dwf.FDwfDeviceOpen(c_int(1), byref(self.hdwf2))
                            if self.hdwf2.value == 0:
                                error_msg = self.get_error_message()
                                self.dwf.FDwfDeviceClose(self.hdwf1)  # Clean up first device
                                raise Exception(f"Failed to open second device: {error_msg}")
                            
                            # Configure first device (Red)
                            self.dwf.FDwfDeviceAutoConfigureSet(self.hdwf1, c_int(1))
                            self.dwf.FDwfAnalogOutNodeEnableSet(self.hdwf1, c_int(0), c_int(0), c_int(1))
                            self.dwf.FDwfAnalogOutNodeFunctionSet(self.hdwf1, c_int(0), c_int(0), c_int(1))
                            
                            # Configure second device (Green and Blue)
                            self.dwf.FDwfDeviceAutoConfigureSet(self.hdwf2, c_int(1))
                            self.dwf.FDwfAnalogOutNodeEnableSet(self.hdwf2, c_int(0), c_int(0), c_int(1))  # Green
                            self.dwf.FDwfAnalogOutNodeEnableSet(self.hdwf2, c_int(1), c_int(0), c_int(1))  # Blue
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

    def get_error_message(self):
        """Get the last error message from the device"""
        error_msg = create_string_buffer(512)
        self.dwf.FDwfGetLastErrorMsg(error_msg)
        return error_msg.value.decode()

    def setup_ui(self):
        """Create the user interface"""
        # Central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Color display
        self.color_display = ColorDisplay()
        layout.addWidget(self.color_display)
        
        # Deviation slider group
        deviation_group = QGroupBox("Color Variation")
        deviation_layout = QVBoxLayout()
        
        self.deviation_slider = QSlider(Qt.Orientation.Horizontal)
        self.deviation_slider.setMinimum(1)
        self.deviation_slider.setMaximum(50)
        self.deviation_slider.setValue(20)  # Default 20%
        self.deviation_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.deviation_slider.setTickInterval(5)
        
        self.deviation_label = QLabel("Deviation: 20%")
        self.deviation_slider.valueChanged.connect(self.update_deviation_label)
        
        deviation_layout.addWidget(self.deviation_label)
        deviation_layout.addWidget(self.deviation_slider)
        deviation_group.setLayout(deviation_layout)
        layout.addWidget(deviation_group)
        
        # Generate Color button
        self.generate_btn = QPushButton("Generate Random Color")
        self.generate_btn.clicked.connect(self.generate_random_color)
        layout.addWidget(self.generate_btn)
        
        # RGB value label
        self.rgb_label = QLabel("RGB: [0, 0, 0]")
        self.rgb_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.rgb_label)
        
        # Prediction result label
        self.prediction_label = QLabel("Prediction: None")
        self.prediction_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.prediction_label)
        
        # Voltage reading label
        self.voltage_label = QLabel("Voltage: N/A")
        self.voltage_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.voltage_label)
        
        # Add matplotlib figure
        self.figure = Figure(figsize=(6, 4))
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.setup_plot()
        layout.addWidget(self.canvas)
        
        # Add Start/Stop Monitoring button
        self.monitor_btn = QPushButton("Start Monitoring")
        self.monitor_btn.clicked.connect(self.toggle_monitoring)
        layout.addWidget(self.monitor_btn)
        
        # Make window larger to accommodate the graph
        self.setMinimumSize(500, 800)

    def setup_plot(self):
        """Initialize the plot"""
        self.ax.set_title('Voltage Signals')
        self.ax.set_xlabel('Time (s)')
        self.ax.set_ylabel('Voltage (V)')
        self.ax.grid(True)
        
        # Initialize empty lines
        self.red_line, = self.ax.plot([], [], 'r-', label='Red')
        self.green_line, = self.ax.plot([], [], 'g-', label='Green')
        self.blue_line, = self.ax.plot([], [], 'b-', label='Blue')
        self.output_line, = self.ax.plot([], [], 'k-', label='Output')
        
        # Set y-axis limits to show up to 6V
        self.ax.set_ylim(0, 6)
        
        # Add horizontal lines for voltage ranges
        self.ax.axhline(y=2.5, color='gray', linestyle='--', alpha=0.5, label='Voltage Threshold')
        
        # Add range labels
        self.ax.text(0.02, 5.5, 'High Range', color='gray', alpha=0.7)
        self.ax.text(0.02, 0.5, 'Low Range', color='gray', alpha=0.7)
        
        self.ax.legend(loc='upper right')
        self.figure.tight_layout()
        
        # Store text annotations in a list
        self.text_annotations = []

    def update_plot(self, v_red, v_green, v_blue, output_voltage, prediction=None, probability=None):
        """Update the plot with new voltage values"""
        current_time = time.time() - self.start_time
        self.time_data.append(current_time)
        
        # Store voltage data
        self.red_data.append(v_red)
        self.green_data.append(v_green)
        self.blue_data.append(v_blue)
        self.output_data.append(output_voltage)
        
        # Update line data
        self.red_line.set_data(list(self.time_data), list(self.red_data))
        self.green_line.set_data(list(self.time_data), list(self.green_data))
        self.blue_line.set_data(list(self.time_data), list(self.blue_data))
        self.output_line.set_data(list(self.time_data), list(self.output_data))
        
        # Remove old text annotations
        for text in self.text_annotations:
            text.remove()
        self.text_annotations.clear()
        
        # Add prediction text if available
        if prediction and probability:
            text = self.ax.text(current_time, output_voltage + 0.2, 
                               f"{prediction}\n({probability:.1%})", 
                               color='black', 
                               fontsize=8,
                               horizontalalignment='right',
                               bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
            self.text_annotations.append(text)
        
        # Adjust x-axis limits to show last 10 seconds
        if current_time > 10:
            self.ax.set_xlim(current_time - 10, current_time)
        else:
            self.ax.set_xlim(0, 10)
        
        # Keep y-axis fixed at 0-6V
        self.ax.set_ylim(0, 6)
        
        # Redraw canvas
        self.canvas.draw()

    def load_voltage_model(self, config_path):
        """Load a saved voltage-based model and its configuration"""
        print(f"Loading configuration from: {config_path}")
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        print(f"Loading model from: {config['model_path']}")
        clf = joblib.load(config['model_path'])
        
        print(f"Loading scaler from: {config['scaler_path']}")
        scaler = joblib.load(config['scaler_path'])
        
        return clf, scaler

    def get_latest_config(self, model_dir, prefix='voltage_config_'):
        """Get the most recent model configuration file"""
        if not os.path.exists(model_dir):
            return None
        config_files = [f for f in os.listdir(model_dir) if f.startswith(prefix) and f.endswith('.yaml')]
        if not config_files:
            return None
        latest_config = max(config_files, key=lambda x: os.path.getctime(os.path.join(model_dir, x)))
        return os.path.join(model_dir, latest_config)

    def update_deviation_label(self):
        """Update the deviation label when slider changes"""
        value = self.deviation_slider.value()
        self.deviation_label.setText(f"Deviation: {value}%")

    def generate_random_color(self):
        """Generate random RGB values close to ROYGBIV colors with Gaussian deviation"""
        # ROYGBIV base colors in RGB
        roygbiv = {
            'red': [255, 0, 0],
            'orange': [255, 127, 0],
            'yellow': [255, 255, 0],
            'green': [0, 255, 0],
            'blue': [0, 0, 255],
            'indigo': [75, 0, 130],
            'violet': [148, 0, 211]
        }
        
        # Randomly select one of the ROYGBIV colors
        color_name = random.choice(list(roygbiv.keys()))
        base_color = roygbiv[color_name]
        
        # Get current deviation percentage from slider
        deviation_percent = self.deviation_slider.value() / 100.0
        
        # Calculate standard deviation for Gaussian distribution
        sigma = [val * deviation_percent for val in base_color]
        
        # Generate random values with Gaussian distribution
        self.current_rgb = [
            int(min(255, max(0, np.random.normal(base, sig))))
            for base, sig in zip(base_color, sigma)
        ]
        
        # Update display
        self.color_display.setColor(*self.current_rgb)
        self.rgb_label.setText(f"RGB: {self.current_rgb} ({color_name})")
        
        print(f"\nGenerated {color_name} color:")
        print(f"Base RGB: {base_color}")
        print(f"Deviation: {deviation_percent*100:.1f}%")
        print(f"Sigma: {[f'{s:.1f}' for s in sigma]}")
        print(f"Final RGB: {self.current_rgb}")
        
        # Automatically predict after generating
        self.predict_current_color()

    def scale_output(self, value, min_input=0, max_input=255, min_output=1.0, max_output=5.0):
        return min_output + (((value - min_input) * (max_output - min_output)) / (max_input - min_input))

    def set_voltage_outputs(self):
        """Set voltage outputs using both devices"""
        if not (self.hdwf1 and self.hdwf2):
            print("Devices not initialized")
            return False, (0, 0, 0)
        
        try:
            # Scale RGB to voltage (1.0-5.0V range, matching training data)
            v_red = self.scale_output(self.current_rgb[0])    # Device 1, W1
            v_green = self.scale_output(self.current_rgb[1])  # Device 1, W2
            v_blue = self.scale_output(self.current_rgb[2])   # Device 2, W1
            
            print(f"\n=== Setting Voltages ===")
            print(f"Input RGB values: {self.current_rgb}")
            print(f"Calculated voltages:")
            print(f"Red   (Device 1, W1): {v_red:.3f}V")
            print(f"Green (Device 1, W2): {v_green:.3f}V")
            print(f"Blue  (Device 2, W1): {v_blue:.3f}V")
            
            # Set voltages matching training setup
            self.dwf.FDwfAnalogOutNodeOffsetSet(self.hdwf1, c_int(0), c_int(0), c_double(v_red))   # Red on D1-W1
            self.dwf.FDwfAnalogOutNodeOffsetSet(self.hdwf1, c_int(1), c_int(0), c_double(v_green)) # Green on D1-W2
            self.dwf.FDwfAnalogOutNodeOffsetSet(self.hdwf2, c_int(0), c_int(0), c_double(v_blue))  # Blue on D2-W1
            
            # Verify settings
            red_actual = c_double()
            green_actual = c_double()
            blue_actual = c_double()
            
            self.dwf.FDwfAnalogOutNodeOffsetGet(self.hdwf1, c_int(0), c_int(0), byref(red_actual))
            self.dwf.FDwfAnalogOutNodeOffsetGet(self.hdwf1, c_int(1), c_int(0), byref(green_actual))
            self.dwf.FDwfAnalogOutNodeOffsetGet(self.hdwf2, c_int(0), c_int(0), byref(blue_actual))
            
            return True, (red_actual.value, green_actual.value, blue_actual.value)
            
        except Exception as e:
            print(f"Error setting voltages: {e}")
            return False, (0, 0, 0)

    def read_voltage(self):
        """Read voltage with averaging like in training"""
        if not self.dwf:
            return None
        
        try:
            v_avg = 0
            for _ in range(10):  # Average 10 readings like in training
                self.dwf.FDwfAnalogInStatus(self.hdwf1, c_int(False), None)
                voltage = c_double()
                self.dwf.FDwfAnalogInStatusSample(self.hdwf1, c_int(0), byref(voltage))
                v_avg += voltage.value
                time.sleep(0.1)  # Same delay as training
            
            v_avg = v_avg / 10
            print(f"Average voltage reading: {v_avg:.3f}V")
            return v_avg
            
        except Exception as e:
            print(f"Error reading voltage: {e}")
            return None

    def toggle_monitoring(self):
        """Start or stop continuous monitoring"""
        if self.monitoring_active:
            self.monitor_timer.stop()
            self.monitor_btn.setText("Start Monitoring")
            self.monitoring_active = False
        else:
            self.monitor_timer.start()
            self.monitor_btn.setText("Stop Monitoring")
            self.monitoring_active = True

    def update_monitoring(self):
        """Update plot with current voltage readings"""
        if not self.dwf:
            return

        try:
            # Read output voltage from first device
            self.dwf.FDwfAnalogInStatus(self.hdwf1, c_int(False), None)
            voltage = c_double()
            self.dwf.FDwfAnalogInStatusSample(self.hdwf1, c_int(0), byref(voltage))
            
            # Get current input voltages and verify they were set correctly
            success, (v_red, v_green, v_blue) = self.set_voltage_outputs()
            
            if success:
                # Create feature vector with voltage and voltage_range
                voltage_range = 1 if voltage.value > 2.5 else 0
                features = self.current_rgb + [voltage.value, voltage_range]
                
                # Make prediction
                features_scaled = self.scaler.transform([features])
                prediction = self.classifier.predict(features_scaled)[0]
                probabilities = self.classifier.predict_proba(features_scaled)[0]
                max_prob = max(probabilities)
                
                # Update plot with actual measured voltages
                self.update_plot(v_red, v_green, v_blue, voltage.value, prediction, max_prob)
                
                # Update voltage label
                self.voltage_label.setText(f"Voltage: {voltage.value:.3f}V")
                
                print("\n=== Current Readings ===")
                print(f"Set Voltages (R,G,B): {v_red:.3f}V, {v_green:.3f}V, {v_blue:.3f}V")
                print(f"Output Voltage: {voltage.value:.3f}V")
                print(f"Voltage Range: {'High' if voltage_range else 'Low'}")
                print(f"Prediction: {prediction} ({max_prob:.1%})")
                
        except Exception as e:
            print(f"Error in monitoring: {e}")
            self.monitor_timer.stop()
            self.monitor_btn.setText("Start Monitoring")
            self.monitoring_active = False

    def predict_current_color(self):
        """Send RGB values to Digilent and predict color using voltage"""
        print(f"\n=== Starting color prediction for RGB: {self.current_rgb} ===")
        
        if self.set_voltage_outputs():
            print("Waiting for voltage to stabilize...")
            time.sleep(0.5)
            
            # Read voltage with averaging
            voltage = self.read_voltage()
            if voltage is not None:
                self.voltage_label.setText(f"Voltage: {voltage:.3f}V")
                
                print("\nMaking prediction...")
                # Create feature vector with voltage and voltage_range
                voltage_range = 1 if voltage > 2.5 else 0
                features = self.current_rgb + [voltage, voltage_range]
                
                # Scale and predict
                features_scaled = self.scaler.transform([features])
                prediction = self.classifier.predict(features_scaled)[0]
                probabilities = self.classifier.predict_proba(features_scaled)[0]
                
                # Find the highest probability
                max_prob = max(probabilities)
                prediction_text = f"Prediction: {prediction} ({max_prob:.2%})"
                print(prediction_text)
                self.prediction_label.setText(prediction_text)
                
                # Update color display
                self.color_display.setColor(*self.current_rgb)
            else:
                self.voltage_label.setText("Voltage: Error reading")
        else:
            self.prediction_label.setText("Prediction: Error setting voltage")

    def run_voltage_test(self):
        """Test output voltages for different RGB combinations"""
        print("\n=== Starting Voltage Test ===")
        
        # Test colors (R,G,B)
        test_colors = [
            (255, 0, 0, "Red"),
            (0, 255, 0, "Green"),
            (0, 0, 255, "Blue"),
            (255, 255, 0, "Yellow"),
            (255, 165, 0, "Orange"),
            (75, 0, 130, "Indigo"),
            (138, 43, 226, "Violet")
        ]
        
        results = []
        for r, g, b, color_name in test_colors:
            print(f"\nTesting {color_name} (R:{r}, G:{g}, B:{b})")
            self.current_rgb = [r, g, b]
            
            # Set voltages and wait for stabilization
            success, (v_red, v_green, v_blue) = self.set_voltage_outputs()
            if success:
                time.sleep(0.5)  # Wait for voltage to stabilize
                
                # Take multiple readings
                voltages = []
                for _ in range(10):
                    voltage = self.read_voltage()
                    if voltage is not None:
                        voltages.append(voltage)
                    time.sleep(0.1)
                
                if voltages:
                    avg_voltage = sum(voltages) / len(voltages)
                    std_voltage = np.std(voltages)
                    print(f"Output voltage: {avg_voltage:.3f}V ± {std_voltage:.3f}V")
                    results.append({
                        'color': color_name,
                        'rgb': (r,g,b),
                        'voltage_mean': avg_voltage,
                        'voltage_std': std_voltage
                    })
                else:
                    print("Failed to read voltage")
            else:
                print("Failed to set voltages")
        
        # Print summary
        print("\n=== Test Results ===")
        print("Color\t\tRGB\t\t\tOutput Voltage")
        print("-" * 50)
        for r in results:
            print(f"{r['color']:<8}\t{r['rgb']}\t{r['voltage_mean']:.3f}V ± {r['voltage_std']:.3f}V")

        # Plot results
        plt.figure(figsize=(10, 6))
        colors = [r['color'] for r in results]
        voltages = [r['voltage_mean'] for r in results]
        errors = [r['voltage_std'] for r in results]
        
        plt.errorbar(colors, voltages, yerr=errors, fmt='o')
        plt.title('Output Voltage by Color')
        plt.xlabel('Color')
        plt.ylabel('Output Voltage (V)')
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.tight_layout()
        plt.show()

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
    app = QApplication(sys.argv)
    window = ColorPredictorApp()
    window.show()
    sys.exit(app.exec()) 