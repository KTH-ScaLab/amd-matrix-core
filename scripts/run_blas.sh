#!/bin/sh

BINARY=$1

PROF_OUTFILE=tmp.csv
PARAMS_FILE=prof_params.txt

# create profiling parameters file
rm $PARAMS_FILE
for PCS in F16 F32 F64; do
echo "pmc: \
    SQ_INSTS_VALU_MFMA_$PCS \
    SQ_INSTS_VALU_FMA_$PCS \
    SQ_INSTS_VALU_MUL_$PCS \
    SQ_INSTS_VALU_ADD_$PCS \
    SQ_INSTS_VALU_MFMA_MOPS_$PCS" >> $PARAMS_FILE
done

header_printed=

# avoid cold start
$BINARY 1 > /dev/null 2>&1

for n in 16 32 64 128 256 512 768 1024 1280 1536 1792 2048 2304 2560 2816 3072 3328 3584 3840 4096 8192 16384 32768 65536; do
    # get profiling information
    rocprof --timestamp on \
        -i $PARAMS_FILE -o $PROF_OUTFILE \
        $BINARY $n > /dev/null

    # print headers, if not already printed
    if [ -z "$header_printed" ]; then
        header_printed=1
        head -n1 $PROF_OUTFILE
    fi

    echo N=$n

    # remove two first lines (header + warmup)
    cat $PROF_OUTFILE
done