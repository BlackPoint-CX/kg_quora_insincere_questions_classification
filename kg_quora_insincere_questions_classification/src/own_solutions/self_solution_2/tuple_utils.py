from collections import namedtuple


class GridSearchResultTuple(namedtuple('GridSearchResultTuple', ['best_score', 'best_estimator', 'best_params'])):
    pass


class TrainTuple(namedtuple('TrainTuple', ['index_name','build_fn', 'param_grid'])):
    pass
