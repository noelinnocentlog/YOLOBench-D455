from setuptools import setup
import os
from glob import glob

package_name = 'yolo_bench_d455'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch',
            glob(os.path.join('launch', '*.py'))),
        ('share/' + package_name + '/config',
            glob(os.path.join('config', '*.yaml'))),
        ('share/' + package_name + '/rviz',
            glob(os.path.join('rviz', '*.rviz'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='your.email@example.com',
    description='High-performance object tracking with depth information',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'tracking_node = yolo_bench_d455.tracking_node:main',
            'visualization = yolo_bench_d455.visualization:main',
            'performance_monitor = yolo_bench_d455.performance_monitor:main',
        ],
    },
)
