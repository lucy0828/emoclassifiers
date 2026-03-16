from setuptools import setup, find_packages

setup(
    name="emoclassifiers",
    version="1.0.0",
    description="LLM-based automatic classifiers for affective cues in user-chatbot conversations",
    packages=find_packages(),
    install_requires=[
        "openai>=1.51.0",
    ],
    python_requires=">=3.8",
)
