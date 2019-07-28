from setuptools import setup
setup(
    name='nbsvm-sklearn',
    version='0.0.2',
    description='sklearn wrapper for NB-SVM algorithm',
    url='https://github.com/Pritishsingla/nbsvm-sklearn.git',
    author='Pritish Singla',
    author_email='pritishs901@gmail.com',
    license='MIT',
    packages=['nbsvm-sklearn'],
    install_requires=[
        'numpy>=1.16.4',
        'scikit-learn>=0.21.0',
    ],
    zip_safe=False
  )
