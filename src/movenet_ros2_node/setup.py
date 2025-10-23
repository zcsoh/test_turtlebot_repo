from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'movenet_ros2_node'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'models'), glob('models/*.*')),
    ],
    install_requires=['setuptools', 'numpy', 'opencv-python', 'openvino-dev', 'cv_bridge', 'rclpy'],
    zip_safe=True,
    maintainer='zicong',
    maintainer_email='2201174@sit.singaporetech.edu.sg',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'movenet_detector = movenet_ros2_node.movenet_node:main'
        ],
    },
)
