from setuptools import setup, find_packages


setup(name='MaskCL',
    #   description='Cluster-guided Asymmetric Contrastive Learning for Unsupervised Person Re-Identification',
    #   author='MingkunLi',
    #   url='https://github.com/MingkunLishigure/MASKCL',
      install_requires=[
          'numpy', 'torch', 'torchvision',
          'six', 'h5py', 'Pillow', 'scipy',
          'scikit-learn', 'metric-learn', 'faiss'],
      packages=find_packages(),
      keywords=[
          'Unsupervised Learning',
          'Unsupervised Domain Adaptation',
          'Contrastive Learning',
          'Object Re-identification'
      ])
