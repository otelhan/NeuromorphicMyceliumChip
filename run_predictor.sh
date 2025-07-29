#!/bin/bash

# Print diagnostic information
echo "=== Environment Setup ==="
echo "Current architecture: $(arch)"
echo "Python version: $(python3 --version)"
echo "Framework path exists: $(ls -l /Library/Frameworks/dwf.framework/dwf)"

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
arch -x86_64 /bin/bash -c "source venv/bin/activate && python cleanup_devices.py"

# Wait a moment for devices to reset
sleep 2

# Run with x86_64 architecture
echo "=== Starting Application ==="
arch -x86_64 /bin/bash -c "source venv/bin/activate && python $1" 