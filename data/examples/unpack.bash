#!/bin/bash

#
# Unpacks data for examples
#

cd $(dirname ${BASH_SOURCE[0]})
wd=$PWD

for filename in \
    SPECFEM3D.tgz;
do
    cd $wd
    cd $(dirname $filename)
    echo "Unpacking $filename"
    tar -xzf $filename
done
echo "Done"
echo ""
