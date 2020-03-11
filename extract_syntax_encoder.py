from allennlp.models.archival import load_archive
from allennlp.predictors.predictor import Predictor

import argparse

parser = argparse.ArgumentParser(description='Extract BERT encoder from UDify model')
parser.add_argument('--ckpt', type=str, required=True,
	help='location of AllenNLP model archive')

def main(args):
	#load model
	archive = load_archive(args.ckpt) #cuda_device=cuda_device)
	predictor = Predictor.from_archive(archive)

	print(predictor)


if __name__ == "__main__":
	args = parser.parse_args()
	print(args) 
	main(args)

#EOF