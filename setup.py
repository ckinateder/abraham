import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="abraham3k",
    version="1.1.3",
    author="Calvin Kinateder",
    author_email="calvinkinateder@gmail.com",
    description="Algorithmically predict public sentiment on a topic using VADER sentiment analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ckinateder/abraham",
    project_urls={
        "Bug Tracker": "https://github.com/ckinateder/abraham/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
)