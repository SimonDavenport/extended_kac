from setuptools import setup

setup(name='extended_kac',
      version='0.1',
      description='An extension of program kac to non-unitary conformal field theories',
      url='https://github.com/SimonDavenport/extended_kac.git',
      author='Simon C Davenport',
      author_email='simon.davenport2@gmail.com',
      license='GPL',
      packages=['lie_algebra', 'cft', 'bcft'],
      zip_safe=False)
