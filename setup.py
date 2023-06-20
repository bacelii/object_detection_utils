from setuptools import setup, find_packages
from pathlib import Path

def get_install_requires(filepath=None):
    if filepath is None:
        filepath = "./"
    """Returns requirements.txt parsed to a list"""
    fname = Path(filepath).parent / 'requirements.txt'
    targets = []
    if fname.exists():
        with open(fname, 'r') as f:
            targets = f.read().splitlines()
    return targets

def get_links():
    return [
    ]

# new comment  for setup.py
setup(
   name='object_detection_utils', # the name of the package, which can be different than the folder when using pip instal
   version='1.0',
   description='Modules to help faciliatate object detection using deep learning in pytorch',
   author='Brendan Celii',
   author_email='brendanacelii@gmail.com',
   packages=find_packages(),  #teslls what packages to be included for the install
   
    #install_requires=get_install_requires(), #external packages as dependencies
    #dependency_links = get_links(),
   
   install_requires=[], #external packages as dependencies
    
    # if wanted to install with the extra requirements use pip install -e ".[interactive]"
    extras_require={
        #'interactive': ['matplotlib>=2.2.0', 'jupyter'],
    },
    
    # if have a python script that wants to be run from the command line
    entry_points={
        #'console_scripts': ['pipeline_download=Applications.Eleox_Data_Fetch.Eleox_Data_Fetcher_vp1:main']
    },
    scripts=[], 
    
)