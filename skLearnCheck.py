from sklearn.utils.estimator_checks import check_estimator
from elm import ExtremeLearningMachine
import atexit

atexit.register(check_estimator(ExtremeLearningMachine()))
# check_estimator(ExtremeLearningMachine())
