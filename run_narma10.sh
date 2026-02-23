#!/bin/bash

# Script to run the Mycelium NARMA-10 application
# This will launch the GUI application for running NARMA-10 experiments

# Print diagnostic information
echo "=== Environment Setup ==="
echo "Current architecture: $(arch)"
echo "Python version: $(python3 --version)"
echo "Framework path exists: $(ls -l /Library/Frameworks/dwf.framework/dwf 2>/dev/null || echo "Not found")"

# Force x86_64 architecture for the entire session
export ARCHFLAGS="-arch x86_64"
export DYLD_LIBRARY_PATH=/Library/Frameworks/dwf.framework:$DYLD_LIBRARY_PATH
export DYLD_FRAMEWORK_PATH=/Library/Frameworks:$DYLD_FRAMEWORK_PATH
export DYLD_ARCH_OVERRIDE=x86_64
export QT_MAC_WANTS_LAYER=1

# Print environment variables
echo "=== Environment Variables ==="
echo "ARCHFLAGS: $ARCHFLAGS"
echo "DYLD_LIBRARY_PATH: $DYLD_LIBRARY_PATH"
echo "DYLD_FRAMEWORK_PATH: $DYLD_FRAMEWORK_PATH"
echo "DYLD_ARCH_OVERRIDE: $DYLD_ARCH_OVERRIDE"

# Run cleanup first
echo "=== Cleaning up devices ==="
arch -x86_64 /bin/bash -c "source venv/bin/activate && python src/cleanup_devices.py"

# Wait a moment for devices to reset
sleep 2

# Run the NARMA-10 application
echo "=== Starting NARMA-10 Application ==="
echo "Note: This will open a GUI window. Use it to:"
echo "  1. Connect devices"
echo "  2. Configure parameters (NARMA order, training length, etc.)"
echo "  3. Select which channels to use"
echo "  4. Start training and testing"
echo ""

arch -x86_64 /bin/bash -c "source venv/bin/activate && python src/mycelium_narma10.py"
