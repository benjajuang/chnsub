from setuptools import setup, find_packages
import io

# read your README for the long description
with io.open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="chnsub",
    version="1.0.2",                    # bump this
    description="Generate smart Chinese subtitles (SRT files)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="benjajuang",
    url="https://github.com/benjajuang/chnsub",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "yt-dlp",
        "openai",
        "pysubs2",
        "python-dotenv",
        "whisperx",
        "transformers",
        "torch",
    ],
    entry_points={
        "console_scripts": [
            "chnsub=chnsub.main:main",  # <— point into your package
        ],
    },
    python_requires=">=3.7",
)
