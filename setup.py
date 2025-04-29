from setuptools import setup, Extension  # type: ignore
import numpy as np

extensions = [
    Extension(
        "pyinsurance.portfolio._tipp",
        ["pyinsurance/portfolio/_tipp.pyx"],
        include_dirs=[np.get_include()],
    ),
]

setup(
    ext_modules=extensions,
)
