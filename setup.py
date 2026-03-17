from distutils.core import setup

setup(
    name='head_embedded',
    python_requires='>=3.8',
    author='Felix jung',
    version='0.0.1',
    packages=['head_embedded','head_embedded.single_fish'],
    license='BSD 3-Clause License',
    description='head_embedded VR for ZebVR platform',
    long_description=open('README.md').read(),
    install_requires=[
        "numpy", 
        "scipy",
        "opencv-python-headless",
        "PyQt5",
        "numba",
        "opencv-python"
        "kalman @ git+https://github.com/ElTinmar/Kalman.git@main"
    ]
)