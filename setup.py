import setuptools

# Read hanabi/_version.py
# Code from StackOverflow: https://stackoverflow.com/questions/458550/standard-way-to-embed-version-into-python-package
import re
VERSIONFILE="hanabi/_version.py"
verstrline = open(VERSIONFILE, "rt").read()
VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(VSRE, verstrline, re.M)
if mo:
    verstr = mo.group(1)
else:
    raise RuntimeError("Unable to find version string in %s." % (VERSIONFILE,))

setuptools.setup(
    name="hanabi",
    version=verstr,
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
    package_data={
        "hanabi.hierarchical": ["data/o3a_bbhpop_inj_info.hdf"]
    },
    entry_points={
        "console_scripts": [
            "hanabi_joint_pipe=hanabi.inference.joint_main:main",
            "hanabi_joint_analysis=hanabi.inference.joint_analysis:main",
            "hanabi_joint_generation_pbilby=hanabi.inference.joint_generation_pbilby:main",
            "hanabi_postprocess_result=hanabi.inference.postprocessing:main",
        ]
    },
    install_requires=[
        "bilby==1.0.2",
        "bilby_pipe==1.0.2",
        "gwpopulation",
        "parallel_bilby==0.1.5",
        "mpi4py",
    ],
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: POSIX",
    ],
    python_requires='>=3.6',
)
