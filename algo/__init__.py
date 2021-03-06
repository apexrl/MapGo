from .normal import NormalLearner
from .hgg_learner import HGGLearner

learner_collection = {
	'normal': NormalLearner,
	'hgg': HGGLearner,
}

def create_learner(args):
	return learner_collection[args.learn](args)