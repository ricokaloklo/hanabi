import setuptools

setuptools.setup(
    name="hanabi",
    version="0.0.6",
    author="Rico Ka Lok Lo, Ignacio Magana",
    author_email="kllo@caltech.edu",
    description="Hierarchical bayesian ANAlysis on lensed GW signals using BIlby",
    long_description="LONG DESCRIPTION HERE",
    url="https://git.ligo.org/ka-lok.lo/hanabi",
    packages=[
        "hanabi",
        "hanabi.lensing",
        "hanabi.bilby_pipe",
        "hanabi.parallel_bilby",
        "hanabi.hierarchical"
    ],
    entry_points={
        "console_scripts": [
            "hanabi_joint_pipe=hanabi.bilby_pipe.joint_main:main",
            "hanabi_joint_analysis=hanabi.bilby_pipe.joint_analysis:main"
        ]
    },
    install_requires=[
        "bilby==1.0.2",
        "bilby_pipe==1.0.1"
    ],
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: MIT License",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: POSIX",
    ],
    python_requires='>=3.6',
)
