import torch
import flowtorch
import flowtorch.bijectors as bijectors

import inspect

#bees = inspect.getmembers(bijectors, inspect.isclass)
#print(bees)

print(bijectors.__all__)

b = bijectors.Sigmoid()
b2 = bijectors.AffineAutoregressive()
