import os
os.environ['MAVLINK20'] = '1'

import time
import collections
import collections.abc
collections.MutableMapping = collections.abc.MutableMapping

from dronekit import connect, VehicleMode

# 1. Connect to the Pixhawk
connection_string = '/dev/ttyTHS1'
print(f"Connecting to companion computer link on: {connection_string}")
vehicle = connect(connection_string, baud=921600, wait_ready=False, heartbeat_timeout=60)

try:
    print("Bypassing GPS/EKF checks for indoor bench test...")

    print("Switching flight mode to STABILIZE (No GPS required)...")
    vehicle.mode = VehicleMode("STABILIZE")

    while vehicle.mode.name != 'STABILIZE':
        print(" Waiting for mode change to STABILIZE...")
        time.sleep(0.5)

    print("Forcing Arm command...")
    vehicle.armed = True

    # Confirm the vehicle has armed
    while not vehicle.armed:
        print(" Waiting for arming confirmation...")
        time.sleep(0.5)

    print("SUCCESS! Motors armed. Holding for 5 seconds...")
    time.sleep(5)
    
    print("Mission complete. Hijacking RC throttle to 0...")
    # ---> THE CRITICAL FIX <---
    # Force the throttle channel (Channel 3) to absolute minimum (1000 PWM)
    vehicle.channels.overrides['3'] = 1000
    time.sleep(0.5)

    # Aggressively send the disarm command until the Pixhawk complies
    while vehicle.armed:
        print(" Sending disarm command...")
        vehicle.armed = False
        time.sleep(0.5)

    print("Drone disarmed safely. Clearing RC overrides...")
    vehicle.channels.overrides = {}

except KeyboardInterrupt:
    print("\nEmergency override triggered by user! Forcing Disarm...")
    
    # ---> THE CRITICAL FIX (EMERGENCY BLOCK) <---
    vehicle.channels.overrides['3'] = 1000
    time.sleep(0.5)
    
    while vehicle.armed:
        print(" Sending emergency disarm command...")
        vehicle.armed = False
        time.sleep(0.5)
        
    print("Emergency disarm successful. Clearing RC overrides...")
    vehicle.channels.overrides = {}

finally:
    time.sleep(2)
    print("Closing vehicle telemetry communication link.")
    vehicle.close()
