from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy, os, sys

if sys.platform == 'darwin':
    os.environ['CFLAGS']   = '-DGGML_USE_ACCELERATE -O3 -std=gnu11'
    os.environ['CXXFLAGS'] = '-DGGML_USE_ACCELERATE -O3 -std=c++11'
    os.environ['LDFLAGS']  = '-framework Accelerate'
else:
    os.environ['CFLAGS']   = '-mavx -mavx2 -mfma -mf16c -O3 -std=gnu11'
    os.environ['CXXFLAGS'] = '-mavx -mavx2 -mfma -mf16c -O3 -std=c++11'

ext_modules = [
    Extension(
        name="whispercpp",
        sources=["whispercpp.pyx", "whisper.cpp/whisper.cpp"],
        language="c++",
        extra_compile_args=["-std=c++11"],
   )
]
ext_modules = cythonize(ext_modules)

whisper_clib = ('whisper_clib', {'sources': ['whisper.cpp/ggml.c']})

setup(
    name='whispercpp',
    version='1.2',
    description='Python custom bindings for whisper.cpp',
    author='Yannis Tevissen',
    author_email='yannis.tevissen@newsbridge.io',
    libraries=[whisper_clib],
    ext_modules = cythonize("whispercpp.pyx"),
    include_dirs = ['./whisper.cpp/', numpy.get_include()],
    install_requires=[
      'numpy',
      'ffmpeg-python',
      'requests',
    ],
)