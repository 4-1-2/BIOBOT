import setuptools

setuptools.setup(
    name="biobot-backend",
    version="0.0.1",
    author="4 y medio",
    author_email="example@gmail.com",
    description="biobot backend package",
    url="https://github.com/4-1-2/BIOBOT",
    scripts=[],
    packages=setuptools.find_packages(include=['biobot', 'biobot.*']),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        'flask >= 1.0.0', 
        'cloudant >= 2.13.0',
        'torch >=1.8',
        'torchvision >= 0.9.0',
        'openai>=0.8.0',
        'numpy>=1.19.4'
    ]
)