from setuptools import setup, find_packages

setup(
    name="graphcontainer",
    version="0.1.0",
    description="GraphContainer project",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    include_package_data=True,
    python_requires=">=3.11",
    install_requires=[
        "openai>=1.108.1",
        "python-dotenv>=1.2.1",
    ],
)
