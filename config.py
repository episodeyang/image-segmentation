import os
from waterbear import DefaultBear

ENV = DefaultBear(lambda: None, **os.environ)
data_dir = ENV.DATA_DIR or "../../datasets/miniscapes"
output_dir = ENV.OUTPUT_DIR or "../../datasets/miniscapes-processed"
run_dir = ENV.RUN_DIR or "./training_logs"
demo_dir = ENV.DEMO_DIR or "../../runs/image-segmentation/demo"
