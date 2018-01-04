#!/bin/bash

echo "Reference,Distorted,A_Map,A_P75,A_P95,L_Map,L_P75,L_P95,R_Map,R_P75,R_P95" > data.csv
paste -d',' \
    <(ls -1 stim/ | grep .mat | cut -d'_' -f-2 | sed -e 's|$|.mat|' | sort) \
    <(ls -1 stim/ | grep .mat | sort) \
    <(ls -1 driim/ | grep _a_P | sort) \
    <(ls -1 driim/ | grep _a_P | sort | cut -d_ -f9 ) \
    <(ls -1 driim/ | grep _a_P | sort | cut -d_ -f11 | sed -e 's|\.png||') \
    <(ls -1 driim/ | grep _l_P | sort) \
    <(ls -1 driim/ | grep _l_P | sort | cut -d_ -f9 ) \
    <(ls -1 driim/ | grep _l_P | sort | cut -d_ -f11 | sed -e 's|\.png||') \
    <(ls -1 driim/ | grep _r_P | sort) \
    <(ls -1 driim/ | grep _r_P | sort | cut -d_ -f9 ) \
    <(ls -1 driim/ | grep _r_P | sort | cut -d_ -f11 | sed -e 's|\.png||')  >> data.csv
