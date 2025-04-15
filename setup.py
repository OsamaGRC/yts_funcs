from setuptools import setup, find_packages

setup(
    name='my_library',  # Name of your package
    version='0.1',      # Initial version
    packages=find_packages(),  # Automatically find packages
    description='A simple library with basic functions',
    author='Your Name',
    author_email='your_email@example.com',
    url='https://github.com/yourusername/my_library',  # GitHub URL (optional)
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    install_requires=[],  # Add dependencies if needed
)