# compile paq8px on Linux with cmake
# sudo apt update
# sudo apt install gcc
# sudo apt install build-essential cmake

rm -rf CMakeFiles CMakeCache.txt cmake_install.cmake Makefile compile_commands.json paq8px
cmake -G "Unix Makefiles" -DNDEBUG=ON ..
make -j$(nproc)
rm -rf CMakeFiles CMakeCache.txt cmake_install.cmake Makefile compile_commands.json


