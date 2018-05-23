from lvq.lvq_1 import Lvq1
from lvq.lvq_facade import Lvq_facade

data_filename = './data/diabetes.csv'
result_filename = './data/result_1.csv'

lvq_facade = Lvq_facade(data_filename, result_filename)
lvq_facade.start_process()