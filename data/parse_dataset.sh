#!/bin/bash
echo "Reference,Distorted,Map,Q" > data.csv
paste -d',' <(ls -1 ref/ | sort) <(ls -1 stim/ | sort) <(ls -1 vdp/ | sort) <(ls -1 vdp/ | sort | cut -d_ -f4 | sed -e 's|\.png||') >> data.csv
