from framework import Network
import numpy as np

test = Network(np.random.rand(8), [9, 16], 5)
test.print()
test.fill_nodes()
print("-----------------------------------------")
test.print()
