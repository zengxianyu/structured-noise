import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="structured_noise",
    version="0.1.0",
    author="Your Name",
    author_email="zengxianyu18@gmail.com",
    description="Structured Noise Generation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/zengxianyu/structured_noise",
    package_dir={"": "."},
    packages=["structured_noise"],
    install_requires=[
        "torch",
        "torchvision",
        "numpy",
        "Pillow",
        "scikit-image",
        "tqdm",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: CC-BY-NC-4.0",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
