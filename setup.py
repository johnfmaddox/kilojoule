import setuptools

with open("README.rst", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="kilojoule",
    version="0.2.10-dev.1",
    author="Jack Maddox",
    author_email="jackmaddox@gmail.com",
    description="A convenience package for thermodynamic and heat transfer calculations",
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
        "coolprop",
        "pyromat",
        "regex",
        "astor",
        "rich",
        "schemdraw",  # required to use the drawing library
        "sigfig",  # required to check solutions
        "emoji",  # optional for "prettier" solution checking
    ],
    python_requires=">=3.7",
)
