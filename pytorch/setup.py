from setuptools import setup, find_packages

setup(name="SFcalculator_torch",
    version='0.1',
    author="Minhaun Li",
    description="A Differentiable pipeline connecting molecule models and crystallpgraphy data", 
    url=" ",
    author_email='minhuanli@g.harvard.edu',
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=[
        "gemmi>=0.5.6",
        "reciprocalspaceship>=0.9.18",
    ],
)