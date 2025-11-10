from setuptools import setup, find_packages

setup(
    name="stLVG_upload",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "scipy",

        "torch==2.2.1",
        "torch-geometric==2.5.2",

        "scanpy==1.9.8",
        "anndata",

        "matplotlib",
        "plotly==5.20.0",
        "opencv-python==4.9.0.80",

        "faiss-cpu==1.8.0",
        "pynvml==11.5.0",
        "tqdm",
        "scikit-learn",
    ],
    extras_require={
        "dev": [ 
            "jupyter",
            "ipykernel",
            "pytest",
        ],
        "gpu": [  
            "faiss-gpu==1.8.0", 
        ],
    },
    python_requires=">=3.8,<3.12", 
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)