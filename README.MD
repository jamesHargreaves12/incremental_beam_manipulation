Comands to run to get experiments working:
- python3 -m venv venv
- source venv/bin/activate
- git clone https://github.com/UFAL-DSG/tgen.git 
- git clone https://github.com/tuetschek/e2e-metrics.git
- git clone https://github.com/thushv89/attention_keras.git
- mv e2e-metrics e2e_metrics
- pip3 install -r requirements.txt
- pip3 install -r e2e_metrics/requirements.txt
- pip3 install git+https://github.com/ufal/pytreex
These next commands are needed to use the e2e_metrics without change working directory to the e2e_metrics directory
- sed -i '' -e 's/from pycocotools/from e2e_metrics.pycocotools/g' e2e_metrics/measure_scores.py 
- sed -i '' -e 's/from pycocoevalcap.eval/from e2e_metrics.pycocoevalcap.eval/g' e2e_metrics/measure_scores.py 
- sed -i '' -e 's/from metrics.pymteval/from e2e_metrics.metrics.pymteval/g' e2e_metrics/measure_scores.py 

The comands setup the data from the e2e-dataset for use in this thesis 
- cd tgen/e2e-challenge/input
./convert.py -a name,near -n ../../../e2e-dataset/trainset.csv train
./convert.py -a name,near -n -m ../../../e2e-dataset/devset.csv devel
 cd ../../.. 
 
 To train the Seq2seq model:
 python3 TrainSeq2Seq.py -c new_configs/model_configs/seq2seq_model.yaml
 
 To test the vanilla beam search:
 python3 get_results.py -c new_configs/vanilla_beam_search.yaml
 
 To train beam manipulator model:
 python3 get_results.py -c new_configs/setup_bm_traindata.yaml
 python3 train_beam_manipulator.py -c new_configs/model_configs/bm_model.yaml
 
 To test beam manipulator:
 python3 get_results.py -c new_configs/bm_test.yaml
 
 To view the bleu scores of the experiments:
 python3 get_results_bleu_scores.py
 For information on running further experiments contact jameshargreaves12@gmail.com