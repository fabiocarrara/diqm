#!/bin/bash

# DRIIM metric has failed for the following images:
# img_100524, img_101386, img_107774
# We filter them out.

echo "Reference,Distorted,A_Map,A_P75,A_P95,L_Map,L_P75,L_P95,R_Map,R_P75,R_P95" > data.csv
paste -d',' \
    <(ls -1 ref/ | grep .mat | egrep -v 'img_(100524|101386|107774)' | sort) \
    <(ls -1 stim/ | grep .mat | egrep -v 'img_(100524|101386|107774)' | sort) \
    <(ls -1 driim/ | grep _a_ | sort) \
    <(ls -1 driim/ | grep _a_ | sort | cut -d_ -f5 ) \
    <(ls -1 driim/ | grep _a_ | sort | cut -d_ -f7 | sed -e 's|\.png||') \
    <(ls -1 driim/ | grep _l_ | sort) \
    <(ls -1 driim/ | grep _l_ | sort | cut -d_ -f5 ) \
    <(ls -1 driim/ | grep _l_ | sort | cut -d_ -f7 | sed -e 's|\.png||') \
    <(ls -1 driim/ | grep _r_ | sort) \
    <(ls -1 driim/ | grep _r_ | sort | cut -d_ -f5 ) \
    <(ls -1 driim/ | grep _r_ | sort | cut -d_ -f7 | sed -e 's|\.png||')  >> data.csv
