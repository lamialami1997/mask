#!/bin/bash
./genseq s1.dna 15000
./genseq s2.dna 15000
taskset -c 2 ./mask s1.dna s2.dna
