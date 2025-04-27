from setuptools import setup, Extension
import numpy as np

extensions = [
    Extension(
        "pyinsurance.portfolio.tipp_",
        ["pyinsurance/portfolio/tipp_.pyx"],
        include_dirs=[np.get_include()],
    ),
]

setup(
    ext_modules=extensions,
)
