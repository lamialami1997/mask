#!/bin/bash
./genseq s1.dna 20000000
./genseq s2.dna 20000000
echo -e "GCC :\n"
./mask.g s1.dna s2.dna
echo -e "ICX :\n"
./mask.i s1.dna s2.dna

