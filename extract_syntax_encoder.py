from allennlp.models.archival import load_archive
from allennlp.predictors.predictor import Predictor
from allennlp.common.util import import_submodules

import argparse
#import sys
#sys.path.append('/private/home/tblevins/udify/')

parser = argparse.ArgumentParser(description='Extract BERT encoder from UDify model')
parser.add_argument('--ckpt', type=str, required=True,
	help='location of AllenNLP model archive')

def main(args):
	#UDify setup (from predict.py file)
	import_submodules("udify")

	predictor = "udify_predictor"

	#load model
	archive = load_archive(args.ckpt) #cuda_device=cuda_device)
	predictor = Predictor.from_archive(archive, predictor)

	print(predictor._model)


if __name__ == "__main__":
	args = parser.parse_args()
	print(args) 
	main(args)

#EOF