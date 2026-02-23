#!/usr/bin/env python3
"""Test script to check device access with proper architecture"""

import sys
import os
from ctypes import cdll, c_int, byref, create_string_buffer

# Set up environment for x86_64
os.environ['DYLD_LIBRARY_PATH'] = '/Library/Frameworks/dwf.framework:' + os.environ.get('DYLD_LIBRARY_PATH', '')
os.environ['DYLD_FRAMEWORK_PATH'] = '/Library/Frameworks:' + os.environ.get('DYLD_FRAMEWORK_PATH', '')

try:
    framework_path = '/Library/Frameworks/dwf.framework/dwf'
    print(f"Loading framework: {framework_path}")
    dwf = cdll.LoadLibrary(framework_path)
    print("✓ Framework loaded")
    
    # Enumerate devices
    deviceCount = c_int()
    result = dwf.FDwfEnum(c_int(0), byref(deviceCount))
    print(f"\nEnumeration result: {result}")
    print(f"Devices found: {deviceCount.value}")
    
    if deviceCount.value == 0:
        print("\n⚠ No devices detected by framework")
        print("\nTroubleshooting:")
        print("1. Make sure devices are powered on (LEDs should be lit)")
        print("2. Try unplugging and replugging USB cables")
        print("3. Check System Settings → Privacy & Security → USB device access")
        print("4. Try opening Digilent WaveForms app first to initialize devices")
        print("5. Make sure no other application is using the devices")
    else:
        print(f"\n✓ Found {deviceCount.value} device(s)")
        for i in range(deviceCount.value):
            # Get device info
            deviceName = create_string_buffer(32)
            deviceSN = create_string_buffer(32)
            dwf.FDwfEnumDeviceName(c_int(i), deviceName)
            dwf.FDwfEnumSN(c_int(i), deviceSN)
            print(f"  Device {i}: {deviceName.value.decode()} - SN: {deviceSN.value.decode()}")
            
            # Try to open device
            hdwf = c_int()
            result = dwf.FDwfDeviceOpen(c_int(i), byref(hdwf))
            if hdwf.value != 0:
                print(f"    ✓ Successfully opened device {i}")
                dwf.FDwfDeviceClose(hdwf)
            else:
                error_msg = create_string_buffer(512)
                dwf.FDwfGetLastErrorMsg(error_msg)
                print(f"    ✗ Failed to open: {error_msg.value.decode()}")
    
except FileNotFoundError:
    print("✗ Framework not found")
except OSError as e:
    print(f"✗ Failed to load framework: {e}")
    print("May need Rosetta 2: softwareupdate --install-rosetta")
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
