# from distutils.core import setup
from setuptools import setup,find_packages

setup(
    name='DeepMSProfiler',
    version='1.0.1',
    description='A tool for mining global features from raw metabolomics data',
    author='Yongjie Deng',
    author_email='dengyj9@mail2.sysu.edu.cn',
    packages=find_packages(),
    install_requires = ["pyteomics","pyteomics","lxml",
                        "tensorflow==2.2","keras==2.3","pandas","numpy","scikit-learn","scikit-image",
                        "Pillow","matplotlib","efficientnet","tqdm","umap-learn"],
    python_requires='>=3.6',
    py_modules=['DeepMSProfiler.mainRun','DeepMSProfiler.modelTrain', 'DeepMSProfiler.modelPred',
                'DeepMSProfiler.modelFeature'],
    url="https://github.com/yjdeng9/DeepMSProfiler",
)