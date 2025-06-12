from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines() # to make libraries in a seperate line

    # setup
    setup(
        name= "mlops", # my project name
        version= "0.1", # my project version
        author= "farhan",
        author_email= "mohammadfarhanalam09@gamil.com",
        description="MLOPS Project",
        packages=find_packages(),
        install_requires = requirements,
        python_requires = ">=3.7"
    )