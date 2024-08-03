from .normal import NormalLearner
from .hgg import DiffusionLearner

learner_collection = {
	'normal': NormalLearner,
	'diffusion': DiffusionLearner,
}

def create_learner(args):
	return learner_collection[args.learn](args)