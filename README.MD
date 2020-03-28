- python3 -m venv venv
- source venv/bin/activate
- rm -rf tgen
- rm -rf e2e_metrics
- git clone https://github.com/UFAL-DSG/tgen.git 
- git clone https://github.com/tuetschek/e2e-metrics.git
- mv e2e-metrics e2e_metrics
- pip3 install -r requirements.txt
- pip3 install -r e2e_metrics/requirements.txt
- edit measure_scores.py to add e2e_metrics.x for the relevant inputs
- comment out the two uncomment matplotlib imports  e2e_metrics/pycocotools/coco.py
- pip install git+https://github.com/ufal/pytreex
- comment out import kenlm in tgen/tgen/lexicalize.py
- cd tgen/e2e-challenge/input
./convert.py -a name,near -n ../../../e2e-dataset/trainset.csv train
./convert.py -a name,near -n -m ../../../e2e-dataset/devset.csv devel
 cd ../../.. 
 python3 build_word2vec.py 
 python3 run_tests_v2.py (to create the model)
 python3 reinforcement_learning.py
