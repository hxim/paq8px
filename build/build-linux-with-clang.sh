# compile paq8px on Lubuntu (x64) with clang
# prepare with:
# sudo apt update
# sudo apt install clang

clang -c -DNDEBUG -I../zlib/  -O3 -flto -funroll-loops -ftree-vectorize -fexceptions -include unistd.h ../zlib/adler32.c ../zlib/crc32.c ../zlib/deflate.c ../zlib/gzlib.c ../zlib/inffast.c ../zlib/inflate.c ../zlib/inftrees.c ../zlib/trees.c ../zlib/zutil.c 

clang++ -DNDEBUG -I../zlib/  -O3 -flto -funroll-loops -ftree-vectorize -s -static -fno-rtti -std=gnu++1z adler32.o crc32.o deflate.o gzlib.o inffast.o inflate.o inftrees.o trees.o zutil.o ../file/*.cpp ../filter/*.cpp ../model/*.cpp ../text/*.cpp ../*.cpp -opaq8px.exe

rm -rf *.o
