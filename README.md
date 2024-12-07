# Image Conversion Node for ROS 2

Convert camera images to grayscale or color with an easy-to-toggle ROS 2 service.
## Features

    🌀 Dynamic Mode Toggle: Switch between grayscale and color processing with a service call.
    📸 Camera Integration: Subscribes to /camera/image_raw and publishes processed images to /camera/image_converted.
    🛠 Customizable: Easily extend for additional image processing modes.

## Prerequisites

Ensure the following are installed:

    ✅ ROS 2 Jazzy (or compatible version).
    ✅ usb_cam package for streaming images.
    ✅ rqt_image_view for visualizing images.

### Installation

Step 1: Clone the Repository

Clone this package into the src directory of your ROS 2 workspace:

    cd ~/mowito_ws/src
    git clone https://github.com/your-username/image_conversion.git

Step 2: Install Dependencies

Use rosdep to install any missing dependencies:

    cd ~/mowito_ws
    rosdep install --from-paths src --ignore-src -r -y

Install the usb_cam package:

    sudo apt install ros-jazzy-usb-cam

Step 3: Build the Workspace

Build the workspace using colcon:

    cd ~/mowito_ws
    colcon build

Step 4: Source the Workspace

Source your workspace to make the package available:

    source ~/mowito_ws/install/setup.bash

### Running the Package

Step 1: Launch the Nodes

Run the launch file to start the usb_cam and image_conversion nodes:

    ros2 launch image_conversion image_conversion.launch.xml

Step 2: Verify Active Topics

Check that the nodes are running and publishing topics:

    ros2 topic list

You should see:

    /camera/image_raw - Published by usb_cam.
    /camera/image_converted - Published by image_conversion.

### Testing the Functionality

Step 1: Visualize Images

Use rqt_image_view to view the input and output images:

    ros2 run rqt_image_view rqt_image_view

In the GUI:

    Select /camera/image_raw to view the original image.
    Select /camera/image_converted to view the processed image.

Step 2: Toggle Processing Modes

Use the /change_mode service to toggle between grayscale and color modes.

Switch to Grayscale:

    ros2 service call /change_mode std_srvs/srv/SetBool "{data: true}"

Switch to Color:

    ros2 service call /change_mode std_srvs/srv/SetBool "{data: false}"

## File Structure:

    image_conversion/
    ├── include/                   # Header files
    ├── launch/
    │   └── image_conversion.launch.xml  # Launch file
    ├── src/
    │   └── image_conversion.cpp   # Main node implementation
    ├── CMakeLists.txt             # Build configuration
    └── package.xml                # ROS 2 package metadata


