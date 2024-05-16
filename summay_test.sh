echo "3-shot"
echo "Full"
bash scripts/summarization/eval.sh 3 full

echo "H2O"
bash scripts/summarization/eval.sh 3 h2o 0.05 0.05 1.0
bash scripts/summarization/eval.sh 3 h2o 0.1 0.1 1.0
bash scripts/summarization/eval.sh 3 h2o 0.15 0.15 1.0
bash scripts/summarization/eval.sh 3 h2o 0.2 0.2 1.0
bash scripts/summarization/eval.sh 3 h2o 0.25 0.25 1.0
bash scripts/summarization/eval.sh 3 h2o 0.3 0.3 1.0
bash scripts/summarization/eval.sh 3 h2o 0.4 0.4 1.0

echo "Decay"
bash scripts/summarization/eval.sh 3 h2o 0.1 0.0 0.1
bash scripts/summarization/eval.sh 3 h2o 0.2 0.0 0.1
bash scripts/summarization/eval.sh 3 h2o 0.3 0.0 0.1
bash scripts/summarization/eval.sh 3 h2o 0.4 0.0 0.1
bash scripts/summarization/eval.sh 3 h2o 0.5 0.0 0.1
bash scripts/summarization/eval.sh 3 h2o 0.6 0.0 0.1
bash scripts/summarization/eval.sh 3 h2o 0.8 0.0 0.1