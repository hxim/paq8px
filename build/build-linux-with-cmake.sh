# compile paq8px on Ubuntu with cmake
# sudo apt update
# sudo apt install gcc build-essential cmake
#------------------------
# if you're on arch, use these instead:
# sudo pacman -Syu
# sudo pacman -S gcc base-devel cmake

rm -rf CMakeFiles CMakeCache.txt cmake_install.cmake Makefile compile_commands.json paq8px
cmake -G "Unix Makefiles" -DNDEBUG=ON ..
make -j$(nproc)
rm -rf CMakeFiles CMakeCache.txt cmake_install.cmake Makefile compile_commands.json


