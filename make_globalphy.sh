#wget http://repo.continuum.io/miniconda/Miniconda-3.7.0-Linux-x86_64.sh 
#bash -b Miniconda-3.7.0-Linux-x86_64.sh 
conda create -n globalphy python=3.4 --yes 
source activate globalphy
conda install -n globalphy pip numpy matplotlib scipy h5py pyqt ipython-notebook requests setuptools cython matplotlib scikit-learn nose six --yes  
conda install -n globalphy numba
mkdir globalphy
cd globalphy
git clone https://github.com/kwikteam/global_superclustering.git
git clone https://github.com/kwikteam/klustakwik2.git
cd klustakwik2
python setup.py install


