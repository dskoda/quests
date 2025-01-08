#!/bin/bash
# This script enables the download of some of the datasets studied in the work
# It also serves as a demonstration of the method
#

if [[ ! -f gap20-full.xyz ]]
then
    wget https://www.repository.cam.ac.uk/bitstreams/d511936c-aac2-4d26-ba4b-5f405ec9bba0/download -O gap20.tgz
    tar -zxvf gap20.tgz
    mv Carbon_GAP_20/Carbon_Data_Set_Total.xyz ./gap20-full.xyz
    rm -rf Carbon_GAP_20
fi

if [[ ! -d gap20 ]]
then
    python3 process_gap20.py
fi

for subset in Graphene Diamond Graphite Nanotubes Fullerenes
do
    echo "DEMO: computing entropy for $subset"
    quests entropy gap20/${subset}.xyz
done
