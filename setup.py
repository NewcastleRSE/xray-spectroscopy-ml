from setuptools import setup

setup(
    name='xanesnet',
    version='0.1.0',    
    description='Theoretical simulation of X-ray spectroscopy (XS)',
    url='https://github.com/NewcastleRSE/xray-spectroscopy-ml',
    author='Professor Thomas Penfold',
    author_email=' tom.penfold@ncl.ac.uk',
    license='This project is licensed under the GPL-3.0 License - see the LICENSE.md file for details.',
    packages=['xanesnet', 'xanesnet.descriptor', 'xanesnet.scheme', 'xanesnet.spectrum', 'xanesnet.model'],
    package_dir={'xanesnet': './xanesnet'},
    install_requires = [
        'ase==3.22.1',
        'matplotlib==3.4.3',
        'numpy==1.22',
        'pyemd==0.5.1',
        'scikit_learn==1.2.0',
        'seaborn==0.11.2',
        'tensorflow-macos==2.9; sys_platform=="darwin"',
        'tensorflow==2.9.2; sys_platform!="darwin"',
        'torch==1.11.0',
        'torchinfo==1.7.1',
        'tqdm==4.62.3',
        'shap==0.41.0',
        'pyyaml==6.0',
        'mlflow==2.1.1',
        'dscribe==1.2.2',
        'optuna==3.1.1',
        'sphinx',
        'sphinx_rtd_theme',
        'pyscf'
    ]
)
