import setuptools
import subprocess
from pathlib import Path

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

# Write a more detailed version file
# Code modified from bilby_pipe
def write_version_file(version):
    version_file = Path("hanabi") / ".version"

    try:
        git_log = subprocess.check_output(
            ["git", "log", "-1", "--pretty=%h %ai"]
        ).decode("utf-8")
        git_diff = (
            subprocess.check_output(["git", "diff", "."])
            + subprocess.check_output(["git", "diff", "--cached", "."])
        ).decode("utf-8")
    except subprocess.CalledProcessError:  # git calls failed
        # we already have a version file, let's use it
        if version_file.is_file():
            return version_file.name
        # otherwise just return the version information
        else:
            git_version = version
    else:
        git_version = "{}: ({}) {}".format(
            version, "UNCLEAN" if git_diff else "CLEAN", git_log.rstrip()
        )
        print(f"parsed git version info as: {git_version!r}")

    with open(version_file, "w") as f:
        print(git_version, file=f)
        print(f"created {version_file}")

    return version_file.name

version_file = write_version_file(verstr)

setuptools.setup(
    name="hanabi",
    version=verstr,
    author="Rico Ka Lok Lo, Ignacio Magana",
    author_email="kllo@caltech.edu",
    description="Hierarchical bayesian ANAlysis on lensed GW signals using BIlby",
    long_description="Identify and characterize strongly-lensed gravitational waves",
    url="https://git.ligo.org/ka-lok.lo/hanabi",
    packages=[
        "hanabi",
        "hanabi.lensing",
        "hanabi.inference",
        "hanabi.hierarchical"
    ],
    package_data={
        "hanabi": [version_file],
        "hanabi.hierarchical": ["data/o3a_bbhpop_inj_info.hdf",
                                "pdetclassifier/pdetclassifier.py",
                                "pdetclassifier/compute_selection_functions.py"
                                "pdetclassifier/trained_2e7_O3_precessing_higherordermodes_3detectors.h5",
                                "pdetclassifier/trained_2e7_O1O2_precessing_higherordermodes_3detectors.h5",
                                "pdetclassifier/trained_2e7_design_precessing_higherordermodes_3detectors.h5",
        ],
    },
    entry_points={
        "console_scripts": [
            "hanabi_joint_pipe=hanabi.inference.joint_main:main",
            "hanabi_joint_analysis=hanabi.inference.joint_analysis:main",
            "hanabi_joint_pipe_pbilby=hanabi.inference.joint_generation_pbilby:main",
            "hanabi_joint_analysis_pbilby=hanabi.inference.joint_analysis_pbilby:main",
            "hanabi_postprocess_result=hanabi.inference.postprocessing:main",
        ]
    },
    install_requires=[
        "bilby>=2.0.1",
        "bilby_pipe>=1.0.8",
        "gwpopulation",
        "parallel_bilby>=2.0.2",
        "mpi4py",
        "configargparse==1.4",
    ],
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: POSIX",
    ],
    python_requires='>=3.8',
)
