import setuptools

setuptools.setup(
    name="seishmc",
    version="0.0.1",
    author="Liang Ding",
    author_email="myliang.ding@mail.utoronto.ca",
    description="Full moment tensor inversion using Hamiltonian Monte Carlo (HMC) sampling",
    long_description="Full moment tensor inversion using Hamiltonian Monte Carlo (HMC) sampling",
    long_description_content_type="text/markdown",
    url="https://github.com/Liang-Ding/seishmc",
    project_urls={
        "Bug Tracker": "https://github.com/Liang-Ding/seishmc/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    keywords=[
        "seismology"
    ],
    # package_dir={"": "seishmc"},
    python_requires='>=3.7.0',
    install_requires=[
        "numpy",
        "pandas",
        "seaborn>=0.11.2",
    ],
    packages=setuptools.find_packages(),
)