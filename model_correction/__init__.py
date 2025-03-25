from model_correction.base_correction_method import Vanilla
from model_correction.clarc import AClarc, Clarc, PClarc
from model_correction.rrc import RRClarc, RRClarcJacobian
from model_correction.rrc_multi import RRClarcMulti


def get_correction_method(method_name):
    CORRECTION_METHODS = {
        'Vanilla': Vanilla,
        'Clarc': Clarc,
        'AClarc': AClarc,
        'PClarc': PClarc,
        'RRClarc': RRClarc,
        'RRClarcMulti': RRClarcMulti,
        'RRClarcJac': RRClarcJacobian,
    }

    assert method_name in CORRECTION_METHODS.keys(), f"Correction method '{method_name}' unknown," \
                                                     f" choose one of {list(CORRECTION_METHODS.keys())}"
    return CORRECTION_METHODS[method_name]
