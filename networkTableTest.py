import time
import ntcore # uv add pyntcore https://robotpy.readthedocs.io/en/2023.5/install/pyntcore.html

if __name__ == "__main__":
    networktable_instance = ntcore.NetworkTableInstance.getDefault()
    table = networktable_instance.getTable("fuelCV")
    table.putString("A String from pi 5", "PI IS CONNECTED")
    table.putNumber("yawDegrees", 100)
    table.putNumber("pitchDegrees", 0)
    table.putNumber("distance", 5)