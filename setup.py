from setuptools import find_packages, setup
from typing import List

# find_packages will search for folder and convert it into package. In our case our folder is 'classifier'.
# Any folder containing __init__ file will be considered as a package for find_packages.

REQUIREMENT_FILE_NAME = "requirements.txt"
HYPHEN_E_DOT = "-e ."


def get_requirements() -> List[str]:
    """
    This function provides list of library names that is required for this project
    """
    with open(REQUIREMENT_FILE_NAME) as requirement_file:
        requirement_list = requirement_file.readlines()
    requirement_list = [requirement_name.replace("\n", "") for requirement_name in requirement_list]
    if HYPHEN_E_DOT in requirement_list:
        requirement_list.remove(HYPHEN_E_DOT)
    return requirement_list


setup(
    name="spam_classification",
    version="0.0.1",
    author="Himanshu Goswami",
    author_email="himgos@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements(),
)
