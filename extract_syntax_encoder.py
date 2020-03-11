from allennlp.models.archival import load_archive
from allennlp.predictors.predictor import Predictor

from udify import util

import argparse
import tarfile
from pathlib import Path

parser = argparse.ArgumentParser(description='Extract BERT encoder from UDify model')
parser.add_argument('--ckpt', type=str, required=True,
	help='location of AllenNLP model archive')

def main(args):
	#UDify setup (from predict.py file)
	import_submodules("udify")

	archive_dir = Path(args.ckpt).resolve().parent

	if not os.path.isfile(archive_dir / "weights.th"):
	    with tarfile.open(args.archive) as tar:
	        tar.extractall(archive_dir)

	config_file = archive_dir / "config.json"

	overrides = {}
	if args.device is not None:
	    overrides["trainer"] = {"cuda_device": args.device}
	if args.lazy:
	    overrides["dataset_reader"] = {"lazy": args.lazy}
	configs = [Params(overrides), Params.from_file(config_file)]
	params = util.merge_configs(configs)

	predictor = "udify_predictor" if not args.raw_text else "udify_text_predictor"


	#load model
	archive = load_archive(args.ckpt) #cuda_device=cuda_device)
	predictor = Predictor.from_archive(archive)

	print(predictor)


if __name__ == "__main__":
	args = parser.parse_args()
	print(args) 
	main(args)

#EOF