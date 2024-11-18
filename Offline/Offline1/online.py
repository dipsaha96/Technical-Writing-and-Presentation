import numpy as np
import matplotlib.pyplot as plt
import DiscreteSignal as ds
class LTI_Discrete:
    def __init__(self, impulse_response: ds.DiscreteSignal):
        self.impulse_response = impulse_response

    def linear_combination_of_impulses(self, input_signal: ds.DiscreteSignal):
        INF = input_signal.INF
        unit_impulses = []
        coefficients = []
        for i in range(0, INF + 1):
            coefficient = input_signal.values[INF + i]
            unit_impulse = ds.DiscreteSignal(INF)
            unit_impulse.set_value_at_time(INF, 1)
            unit_impulses.append(unit_impulse.shift_signal(i))
            coefficients.append(coefficient)

        return unit_impulses, coefficients

    def output(self, input_signal: ds.DiscreteSignal):
        INF = input_signal.INF
        output_signal = ds.DiscreteSignal(len(poly1)+len(poly2)-1)
        coefficients=[]
        constituent_impulses = []
        for i in range(0, INF + 1):
            coefficients.append(input_signal.values[INF + i])
            response = self.impulse_response.shift_signal(i)
            constituent_impulses.append(response)
            output_signal = output_signal.add(
                response.multiply_const_factor(input_signal.values[INF + i])
            )
        return output_signal, constituent_impulses, coefficients
    
if __name__ == "__main__":
    INF = 10
    d1 = int(input("Degree of the first Polynomial: "))
    poly1 = list(map(int, input("Coefficients: ").split()))
    
    d2 = int(input("Degree of the second Polynomial: "))
    poly2 = list(map(int, input("Coefficients: ").split()))
    poly1s = ds.DiscreteSignal(len(poly1))
    for i in range(0, len(poly1)-1):
        poly1s.set_value_at_time(i,poly1[i])
    poly2s = ds.DiscreteSignal(len(poly2))
    for i in range(0, len(poly2)-1):
        poly2s.set_value_at_time(i,poly2[i])

    lti = LTI_Discrete(poly1s)
    
    output_signal, constituent_impulses, coefficients = lti.output(poly2s)

    for constituent_impulse, coefficient in zip(constituent_impulses, coefficients):
        constituent_impulse.multiply_const_factor(coefficient)

    print("Degree of the Polynomial:", len(constituent_impulses) - 1)



