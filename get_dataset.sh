if [ ! -d "dataset" ]; then
  mkdir dataset
fi

aws s3 cp --no-sign-request "s3://cse-cic-ids2018/Processed Traffic Data for ML Algorithms/Wednesday-28-02-2018_TrafficForML_CICFlowMeter.csv" "dataset"
python dataset.py