from framework import Network
import numpy as np

test = Network(np.random.rand(8), [9, 16], 5)
test.forward_prop()
print("--------------------NODES--------------------")
test.print_nodes()
print("-------------------WEIGHTS-------------------")
test.print_weights()
print("--------------------BIASES-------------------")
test.print_biases()
test.calc_der_C_wrt_a([0, 0, 0, 1, 0])
print(test.back_prop([0, 0, 0, 1, 0]))
