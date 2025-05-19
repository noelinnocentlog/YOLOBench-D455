ARG ROS_DISTRO=humble
FROM ros:${ROS_DISTRO} AS deps

# Create ros2_ws and copy files
WORKDIR /root/ros2_ws
SHELL ["/bin/bash", "-c"]
COPY . /root/ros2_ws/src

# Install system dependencies
RUN apt-get update && \
    apt-get -y --quiet --no-install-recommends install python3-pip

# Install ROS 2 dependencies
RUN rosdep update && \
    rosdep install --from-paths src --ignore-src -r -y

# More robust pip install with retries and increased timeout
RUN INSTALL_CMD="pip3 install --no-cache-dir --default-timeout=1000 -r src/requirements.txt -i https://pypi.org/simple"; \
    for i in {1..5}; do \
        if [ "$ROS_DISTRO" = "jazzy" ] || [ "$ROS_DISTRO" = "rolling" ]; then \
            $INSTALL_CMD --break-system-packages --ignore-installed && break; \
        else \
            $INSTALL_CMD && break; \
        fi; \
        echo 'Retrying pip install in 10 seconds...'; \
        sleep 10; \
    done

# Build the workspace with colcon
FROM deps AS builder
ARG CMAKE_BUILD_TYPE=Release
RUN source /opt/ros/${ROS_DISTRO}/setup.bash && colcon build

# Source the ROS 2 setup file
RUN echo "source /root/ros2_ws/install/setup.bash" >> ~/.bashrc

# Default to bash shell
CMD ["bash"]

