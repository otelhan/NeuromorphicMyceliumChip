import os
import subprocess
import time
import glob
import ctypes
from ctypes import c_int, byref

def reset_devices():
    """Attempt to physically reset the Digilent devices"""
    try:
        # Load the framework
        dwf = ctypes.cdll.LoadLibrary("/Library/Frameworks/dwf.framework/dwf")
        
        # Enumerate devices
        deviceCount = c_int()
        dwf.FDwfEnum(c_int(0), byref(deviceCount))
        
        # Try to reset each device
        for i in range(deviceCount.value):
            hdwf = c_int()
            try:
                # Try to open and close each device
                dwf.FDwfDeviceOpen(c_int(i), byref(hdwf))
                if hdwf.value != 0:
                    # Reset device
                    dwf.FDwfDeviceReset(hdwf)
                    # Close device
                    dwf.FDwfDeviceClose(hdwf)
                    print(f"Reset device {i}")
            except:
                pass
    except:
        print("Could not load framework for reset")

def cleanup_digilent_processes():
    """Clean up any processes using Digilent devices"""
    print("\n=== Starting Digilent Cleanup ===")
    
    # List of all possible Digilent related processes
    digilent_processes = [
        "waveforms",
        "WaveForms",
        "python",
        "Python",
        "AD2",
        "Digilent"
    ]
    
    # Kill any process containing Digilent-related names
    try:
        ps_output = subprocess.check_output(['ps', 'aux']).decode()
        for line in ps_output.split('\n'):
            if any(proc in line for proc in digilent_processes):
                try:
                    pid = int(line.split()[1])
                    print(f"Killing process: {line.strip()}")
                    os.kill(pid, 9)
                except:
                    pass
    except:
        pass
    
    print("Waiting for processes to terminate...")
    time.sleep(3)
    
    # Check and kill processes using framework files
    framework_components = [
        '/Library/Frameworks/dwf.framework/dwf',
        '/Library/Frameworks/dwf.framework/Versions/A/Frameworks/libdjtg.dylib',
        '/Library/Frameworks/dwf.framework/Versions/A/Frameworks/libdmgr.dylib',
        '/Library/Frameworks/dwf.framework/Versions/A/Frameworks/libdabs.dylib',
        '/Library/Frameworks/dwf.framework/Versions/A/Frameworks/libdpcutil.dylib'
    ]
    
    for component in framework_components:
        try:
            output = subprocess.check_output(['lsof', component], stderr=subprocess.STDOUT).decode()
            for line in output.split('\n')[1:]:
                if line:
                    try:
                        pid = int(line.split()[1])
                        print(f"Killing process using {component}: {pid}")
                        os.kill(pid, 9)
                    except:
                        pass
        except:
            pass
    
    print("Waiting for framework processes to terminate...")
    time.sleep(3)
    
    # Remove lock files
    lock_patterns = [
        '/tmp/digilent_*',
        '/tmp/waveforms_*',
        '/var/tmp/digilent_*',
        '/var/tmp/waveforms_*',
        os.path.expanduser('~/Library/Application Support/WaveForms/*/lockfile'),
        os.path.expanduser('~/Library/Application Support/Digilent/*/lockfile')
    ]
    
    for pattern in lock_patterns:
        for lock_file in glob.glob(pattern):
            try:
                os.remove(lock_file)
                print(f"Removed lock file: {lock_file}")
            except:
                pass
    
    print("Attempting physical device reset...")
    reset_devices()
    
    print("Waiting for devices to stabilize...")
    time.sleep(5)
    
    print("=== Cleanup Complete ===\n")

if __name__ == "__main__":
    cleanup_digilent_processes() 