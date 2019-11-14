#!/bin/bash

# VDP Q
python train.py data/vdp/ q -b32 -l q_normal_b32_e30
python eval.py data/vdp/ q -r runs/vdp/q_normal_b32_e30/

# VDP Map
python train.py data/vdp/ vdp -l vdp_normal_b4_e30
python eval.py data/vdp/ vdp -r runs/vdp/vdp_normal_b4_e30/

# VDP-COMP Q
python train.py data/vdp-comp/ q -b32 -l q_normal_b32_e30
python eval.py data/vdp-comp/ q -r runs/vdp-comp/q_normal_b32_e30/

# VDP-COMP Map
python train.py data/vdp-comp/ vdp -e 60 -l vdp_normal_b4_e60
python eval.py data/vdp-comp/ vdp -r runs/vdp-comp/vdp_normal_b4_e60/

# DRIIM-WEB
python train.py data/driim-web/ driim
python eval.py data/driim-web/ driim -r runs/driim-web/normal_b4_e30/

# DRIIM-ITMO
python train.py data/driim-itmo/ driim
python eval.py data/driim-itmo/ driim -r runs/driim-itmo/normal_b4_e30/

# DRIIM-TMO
python train.py data/driim-tmo/ driim
python eval.py data/driim-tmo/ driim -r runs/driim-tmo/normal_b4_e30/

# DRIIM-COMP
python train.py data/driim-comp/ driim
python eval.py data/driim-comp/ driim -r runs/driim-comp/normal_b4_e30/


# TIMINGS
DIMENSIONS=(
    '128 128 1'
    '128 128 3'
    '256 128 1'
    '256 128 3'
    '256 256 1'
    '256 256 3'
    '512 256 1'
    '512 256 3'
    '512 512 1'
    '512 512 3'
    '1024 512 1'
    '1024 512 3'
    '1024 1024 1'
    '1024 1024 3'
    '2048 1024 1'
    '2048 1024 3'
)

echo 'DRIIM/VDP:'
for DIM in "${DIMENSIONS[@]}"; do
    TF_CPP_MIN_LOG_LEVEL=6 python timing.py driim -b1 -i $DIM 2> /dev/null | tail -n1
done

echo 'Q:'
for DIM in "${DIMENSIONS[@]}"; do
    TF_CPP_MIN_LOG_LEVEL=6 python timing.py q -b1 -i $DIM 2> /dev/null | tail -n1
done