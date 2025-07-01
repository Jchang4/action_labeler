import os

from setuptools import find_packages, setup

# Read version from __version__.py
version_file = os.path.join(
    os.path.dirname(__file__), "action_labeler", "__version__.py"
)
with open(version_file) as f:
    exec(f.read())

# Read README for long description
readme_file = os.path.join(os.path.dirname(__file__), "README.md")
with open(readme_file, "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="action_labeler",
    version=__version__,
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
    extras_require={
        "dev": [
            "jupyter",
            "notebook",
            "ipykernel",
        ],
        "ai": [
            "openai",
            "transformers",
            "torch",
            "torchvision",
        ],
    },
    description="A Python package for labeling actions in images using AI models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Justin",
    author_email="",
    url="https://github.com/Jchang4/action_labeler",
    project_urls={
        "Bug Reports": "https://github.com/Jchang4/action_labeler/issues",
        "Source": "https://github.com/Jchang4/action_labeler",
        "Documentation": "https://github.com/Jchang4/action_labeler#readme",
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],
    keywords="computer-vision, machine-learning, ai, image-labeling, yolo, object-detection, action-recognition",
    python_requires=">=3.9",
    include_package_data=True,
    zip_safe=False,
)
