import setuptools
from pathlib import Path

README = (Path(__file__).parent/"README.md").read_text()

setuptools.setup(
    name="llm-templates",
    version="0.1.8",
    author="Juan Pablo Manson",
    author_email="jpmanson@gmail.com",
    description="Instruction/chat prompts creation library for text generation LLMs. It supports local and Hugging Face models.",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/jpmanson/llm-templates",
    packages=setuptools.find_packages(),
    include_package_data=True,
    classifiers=[],
    python_requires=">=3.9",
    license_files=("LICENSE",),
    install_requires=[
        "requests>=2.28.0",
        "Jinja2>=2.11.3",
    ]
)