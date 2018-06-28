# -*- coding: utf-8 -*-
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

testdir = os.path.dirname(__file__)

import smite
import numpy as np

def main():

    X = np.random.randint(10, size=3000)
    Y = np.random.randint(10, size=3000)
    
    # Uncomment this for an example of a time series (Y) clearly anticipating values of X
    #Y = np.roll(X,-1)
    
    symX = smite.symbolize(X,3)
    symY = smite.symbolize(Y,3)
    
    MI = smite.symbolic_mutual_information(symX, symY)
    
    TXY = smite.symbolic_transfer_entropy(symX, symY)
    TYX = smite.symbolic_transfer_entropy(symY, symX)
    TE = TYX - TXY
    
    print("---------------------- Random Case ----------------------")
    print("Mutual Information = " + str(MI))
    print("T(Y->X) = " + str(TXY) + "    T(X->Y) = " + str(TYX))
    print("Transfer of Entropy = " + str(TE))

# Shifted Values
    X = np.random.randint(10, size=3000)
    Y = np.roll(X,-1)
    
    symX = smite.symbolize(X,3)
    symY = smite.symbolize(Y,3)
    
    MI = smite.symbolic_mutual_information(symX, symY)
    
    TXY = smite.symbolic_transfer_entropy(symX, symY)
    TYX = smite.symbolic_transfer_entropy(symY, symX)
    TE = TYX - TXY
    
    print("------------------ Y anticipates X Case -----------------")
    print("Mutual Information = " + str(MI))
    print("T(Y->X) = " + str(TXY) + "    T(X->Y) = " + str(TYX))
    print("Transfer of Entropy = " + str(TE))

if __name__ == "__main__":
    main()