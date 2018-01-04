#!/bin/bash
echo "Reference,Distorted,Map,Q" > data.csv
paste -d',' \
    <(ls -1 stim/ | grep .mat | cut -d'_' -f-2 | sed -e 's|$|.mat|' | sort) \
    <(ls -1 stim/ | grep .mat | sort) \
    <(ls -1 vdp/ | sort) \
    <(ls -1 vdp/ | sort | cut -d_ -f8 | sed -e 's|\.png||') >> data.csv
