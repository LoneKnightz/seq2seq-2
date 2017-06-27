

# Speech processing

## Install Yaafe

~~~
sudo apt-get install cmake cmake-curses-gui libargtable2-0 libargtable2-dev
libsndfile1 libsndfile1-dev libmpg123-0 libmpg123-dev libfftw3-3 libfftw3-dev
liblapack-dev libhdf5-serial-dev

wget https://sourceforge.net/projects/yaafe/files/yaafe-v0.64.tgz/download -O yaafe-v0.64.tgz

tar xzf yaafe-v0.64.tgz
cd yaafe-v0.64

# fix bug in the official release
cat src_cpp/yaafe-core/Ports.h | sed "s/\tpush_back/\tthis->push_back/g" > src_cpp/yaafe-core/Ports.h.fixed
mv src_cpp/yaafe-core/Ports.h.fixed src_cpp/yaafe-core/Ports.h

mkdir build
cd build
cmake ..
make
sudo make install

echo "export PYTHONPATH=/usr/local/python_packages/:\$PYTHONPATH" >> ~/.bashrc
echo "export LD_LIBRARY_PATH=/usr/local/lib/:\$LD_LIBRARY_PATH" >> ~/.bashrc
echo "export YAAFE_PATH=/usr/local/yaafe_extensions" >> ~/.bashrc
~~~