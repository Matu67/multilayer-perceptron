from framework import Network
import numpy as np

test = Network(np.random.rand(8), [9, 16], 5)
test.print()
test.forward_prop()
print("-----------------------------------------")
test.print()
