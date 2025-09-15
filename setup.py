import setuptools


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="kilojoule",
    version="0.4.1",
    author="Jack Maddox",
    author_email="jackmaddox@gmail.com",
    description="A convenience package for engineering calculations with a focus on fluids mechanics, thermodynamics, and heat transfer",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/johnfmaddox/kilojoule",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "numpy",
        "scipy",
        "sympy",
        "pandas",
        "matplotlib",
        "pint",
        "pint-pandas",
        "uncertainties",
        "coolprop",
        "pyromat",
        "regex",
        "rich",
        "schemdraw",  # required to use the drawing library
        "sigfig",  # required to check solutions
        "emoji",  # optional for "prettier" solution checking
        "icecream",
        "seaborn",
    ],
    python_requires=">=3.8",
)
