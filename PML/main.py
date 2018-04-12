#shebang

from numpy import exp, array, random, dot

class Neurona():
    def __sigmoide(self, x):
    	"""
    	Funcion de activacion
    	"""
        return 1 / (1 + exp(-x))