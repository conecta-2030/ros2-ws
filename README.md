# Lidar and camera object detection

This README provides instructions on how to build and run the ROS 2 Humble project located in this repository.

## Prerequisites

* **ROS 2 Humble Hawksbill:** Ensure that ROS 2 Humble is installed on your system. Follow the official ROS 2 installation instructions for your operating system: [ROS 2 Humble Installation](https://docs.ros.org/en/humble/Installation.html).
* **Python 3.10:** This project uses Python 3.10. Verify your Python version using:

    ```bash
    python3 --version
    ```

    If Python 3.10 is not installed, install it using your system's package manager.
* **Colcon:** Colcon is the build tool used in ROS 2. If it's not already installed, install it using:

    ```bash
    sudo apt update
    sudo apt install python3-colcon-common-extensions
    ```

## Packages

* **cv\_basics:**
    * This package utilizes OpenCV and YOLO for pedestrian and vehicle detection from camera input.
    * **Configuration:** Before running, the PoE camera IP address must be configured.
    * **Dependencies:** OpenCV, YOLO, and associated Python libraries.
* **mmdet3d\_ros2:**
    * This package employs mmdet3d inference to detect vehicles and pedestrians from point cloud data.
    * **Configuration:** The point cloud topic must be updated to match the incoming point cloud data topic.
    * **Dependencies:** mmdet3d, and associated python libraries.
* **ros2-lidar-object-detection:**
    * This package creates MarkerArray messages from bounding box detection results and publishes them for visualization in RViz.
    * **Dependencies:** ROS 2 MarkerArray message definitions.

## Installation and Build Instructions

1.  **Clone the repository:**

    ```bash
    git clone [your_repository_url] ros2_ws
    cd ros2_ws
    ```

2.  **Install Python dependencies:**

    Install the Python dependencies listed in the root `requirements.txt`:

    ```bash
    python3 -m pip install -r requirements.txt
    ```

    This will install all the required python packages for all the ros2 packages within the src directory.

3.  **Build the ROS 2 workspace:**

    Build the workspace using Colcon:

    ```bash
    colcon build
    ```

    This command will build all the ROS 2 packages in the `src` directory.

4.  **Source the setup files:**

    Source the setup files to add the built packages to your ROS 2 environment:

    ```bash
    source install/setup.bash
    ```

    Add this line to your `~/.bashrc` file to automatically source the setup files every time you open a new terminal:

    ```bash
    echo "source ~/ros2_ws/install/setup.bash" >> ~/.bashrc
    source ~/.bashrc
    ```

## Running the Project

The `run_all.sh` script is provided to launch all the necessary nodes for the project. Make the script executable and run it:

```bash
chmod +x run_all.sh
./run_all.sh
```

This script automatically executes all the necessary launch files or `ros2 run` commands.