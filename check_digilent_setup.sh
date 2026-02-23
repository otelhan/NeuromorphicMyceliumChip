#!/bin/bash

echo "=== Digilent Device Setup Diagnostic ==="
echo ""

# Check 1: Framework installation
echo "1. Checking Digilent WaveForms SDK installation..."
FRAMEWORK_PATH="/Library/Frameworks/dwf.framework/dwf"
if [ -f "$FRAMEWORK_PATH" ]; then
    echo "   ✓ Framework found at: $FRAMEWORK_PATH"
    ls -lh "$FRAMEWORK_PATH"
else
    echo "   ✗ Framework NOT found at: $FRAMEWORK_PATH"
    echo "   → You need to install Digilent WaveForms SDK"
    echo "   → Download from: https://digilent.com/reference/software/waveforms/waveforms-3/start"
    echo ""
fi

# Check 2: Architecture
echo ""
echo "2. Checking system architecture..."
ARCH=$(arch)
echo "   Current architecture: $ARCH"
if [ "$ARCH" = "arm64" ]; then
    echo "   ⚠ Running on ARM64 - need x86_64 for Digilent framework"
    echo "   → The script should use 'arch -x86_64' to force x86_64 mode"
fi

# Check 3: Python architecture
echo ""
echo "3. Checking Python architecture..."
if command -v python3 &> /dev/null; then
    PYTHON_ARCH=$(python3 -c "import platform; print(platform.machine())")
    echo "   Python architecture: $PYTHON_ARCH"
else
    echo "   ✗ Python3 not found"
fi

# Check 4: USB devices
echo ""
echo "4. Checking for connected USB devices..."
if command -v system_profiler &> /dev/null; then
    USB_DEVICES=$(system_profiler SPUSBDataType 2>/dev/null | grep -i "digilent\|analog discovery" | wc -l | tr -d ' ')
    echo "   Found $USB_DEVICES Digilent-related USB device(s)"
    if [ "$USB_DEVICES" -eq "0" ]; then
        echo "   ⚠ No Digilent devices detected"
        echo "   → Make sure your Analog Discovery 2 devices are connected via USB"
    fi
else
    echo "   Could not check USB devices"
fi

# Check 5: Try to load framework (if exists)
echo ""
echo "5. Testing framework loading..."
if [ -f "$FRAMEWORK_PATH" ]; then
    echo "   Attempting to load framework..."
    arch -x86_64 /bin/bash -c "
        export DYLD_LIBRARY_PATH=/Library/Frameworks/dwf.framework:\$DYLD_LIBRARY_PATH
        export DYLD_FRAMEWORK_PATH=/Library/Frameworks:\$DYLD_FRAMEWORK_PATH
        python3 << 'PYTHON_EOF'
import sys
import os
from ctypes import cdll, c_int, byref

try:
    framework_path = '/Library/Frameworks/dwf.framework/dwf'
    print(f'   Loading: {framework_path}')
    dwf = cdll.LoadLibrary(framework_path)
    print('   ✓ Framework loaded successfully')
    
    # Try to enumerate devices
    deviceCount = c_int()
    result = dwf.FDwfEnum(c_int(0), byref(deviceCount))
    print(f'   Devices found: {deviceCount.value}')
    
    if deviceCount.value == 0:
        print('   ⚠ No devices detected')
        print('   → Check USB connections')
        print('   → Try unplugging and replugging devices')
    elif deviceCount.value == 1:
        print('   ⚠ Only 1 device found (need 2)')
        print('   → Connect second Analog Discovery 2 device')
    else:
        print(f'   ✓ Found {deviceCount.value} device(s)')
        
except FileNotFoundError as e:
    print(f'   ✗ Framework not found: {e}')
except OSError as e:
    print(f'   ✗ Failed to load framework: {e}')
    print('   → May need to install Rosetta 2: softwareupdate --install-rosetta')
except Exception as e:
    print(f'   ✗ Error: {e}')
PYTHON_EOF
    " 2>&1
else
    echo "   Skipping (framework not installed)"
fi

echo ""
echo "=== Diagnostic Complete ==="
echo ""
echo "Common issues and solutions:"
echo "1. Framework not installed → Install Digilent WaveForms SDK"
echo "2. No devices found → Connect 2 Analog Discovery 2 devices via USB"
echo "3. Architecture mismatch → Script should use 'arch -x86_64'"
echo "4. Permission denied → May need to grant USB access in System Settings"
