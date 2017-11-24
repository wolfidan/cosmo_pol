 #!/bin/bash

fname=$1
swig -python ${fname}.i
gcc -fPIC -c ${fname}.c ${fname}_wrap.c -I/usr/include/python2.7
ld -shared ${fname}.o ${fname}_wrap.o -o _${fname}.so
