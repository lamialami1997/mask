set term png size 1900,700 enhanced font "Terminal,12"
set output "plot.png"
set grid

set datafile separator ";"

set auto x

set style data histogram
set style fill solid border -1
set boxwidth 0.9

set xtic rotate by -44 scale 0

set multiplot layout 1, 1 rowsfirst

set key left
set yrange [0:70]
set ylabel "GB/s"
set title "Comparison between GCC and ICC on mask benchmark with sequence lenth 20000000 and with different number of threads"
plot "maskgcc_8.dat" using 4:xticlabels(stringcolumn(1)) t "8 Threads + GCC", \
	 "maskicx_8.dat" using 4:xticlabels(stringcolumn(1)) t "8 Threads +ICX", \
	 "maskgcc_32.dat" using 4:xticlabels(stringcolumn(1)) t "32 Threads + GCC", \
	 "maskicx_32.dat" using 4:xticlabels(stringcolumn(1)) t "32 Threads +ICX", \
	 "maskgcc_64.dat" using 4:xticlabels(stringcolumn(1)) t "64 Threads +GCC", \
	 "maskicx_64.dat" using 4:xticlabels(stringcolumn(1)) t "64 Threads +ICX", \
	 "maskgcc_128.dat" using 4:xticlabels(stringcolumn(1)) t "128 Threads +GCC", \
	 "maskicx_128.dat" using 4:xticlabels(stringcolumn(1)) t "128 Threads +ICX", \
