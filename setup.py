from setuptools import find_packages, setup

setup(
    name="action_labeler",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "ultralytics",
        "supervision",
        "pillow",
        "tqdm",
        "pyyaml",
        "numpy",
        "pandas",
        "matplotlib",
    ],
    description="A tool for labeling actions in images",
    author="Justin",
    author_email="",
    url="https://github.com/Jchang4/action_labeler",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
)
