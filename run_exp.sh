#!/bin/bash 

echo "Run baseline"
cd proj_invivo/baseline
python most_freq.py
cd ../nlp_approach
echo "Running word2vec"
python word2vec.py
echo "Running GRU, may take longer"
python gru.py

