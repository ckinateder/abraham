import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="abraham3k",
    version=open("version").read().strip(),
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
    packages=setuptools.find_packages(include=["abraham3k", "abraham3k.*"]),
    python_requires=">=3.6",
    install_requires=[
        "nltk>=3.6.1",
        "newspaper3k>=0.2.8",
        "GoogleNews>=1.5.7",
        "pandas>=1.2.3",
        "tqdm>=4.58.0",
        "flair>=0.8.0.post1",
        "twint>=2.1.20",
    ],
)