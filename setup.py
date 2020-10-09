import setuptools

setuptools.setup(
    name="hanabi",
    version="0.1.0",
    author="Rico Ka Lok Lo, Ignacio Magana",
    author_email="kllo@caltech.edu",
    description="Hierarchical bayesian ANAlysis on lensed GW signals using BIlby",
    long_description="LONG DESCRIPTION HERE",
    url="https://git.ligo.org/ka-lok.lo/hanabi",
    packages=[
        "hanabi",
        "hanabi.lensing",
        "hanabi.inference",
        "hanabi.hierarchical"
    ],
    entry_points={
        "console_scripts": [
            "hanabi_joint_pipe=hanabi.inference.joint_main:main",
            "hanabi_joint_analysis=hanabi.inference.joint_analysis:main"
        ]
    },
    install_requires=[
        "bilby==1.0.2",
        "bilby_pipe==1.0.1"
    ],
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: POSIX",
    ],
    python_requires='>=3.6',
)
