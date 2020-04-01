from setuptools import setup, find_packages

setup(name='tinnsleep',
      version='0.0.0',
      description='Detecting events in sleeping tinnitus patients',
      url='',
      author='Louis Korczowski',
      author_email='louis.korczowski@gmail.com',
      license='(c) SIOPI',
      packages=find_packages(),
      install_requires=['numpy', 'scipy', 'scikit-learn', 'pandas', 'pytest', 'matplotlib'],
      zip_safe=False)