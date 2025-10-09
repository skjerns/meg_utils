import setuptools

# Read the contents of your README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read the requirements from your requirements.txt file
with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = f.read().splitlines()

# This setup is configured to create a package named 'meg_utils'
# directly from the .py files in the project root.
# The 'package_dir' argument maps the package name to the root directory ('.')
# and 'packages' specifies the name of the package.
setuptools.setup(
    name="meg_utils",
    version="0.1.0",
    author="Simon Kern",
    author_email="simon.kern@zi-mannheim.de",
    description="A collection of utility functions for MEG analysis.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/simon-kern/meg_utils", # Replace with your actual repo URL
    packages=['meg_utils'],
    package_dir={'meg_utils': '.'},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=requirements,
)