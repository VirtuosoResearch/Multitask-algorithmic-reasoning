import numpy as np

def addition(digits1, digits2):
    digits1 = digits1[::-1]
    digits2 = digits2[::-1]
    digits = digits1 + digits2
    carry = False
    for i in range(len(digits)):
        if digits[i] >= 10:
            if i == len(digits) - 1:
                digits = np.append(digits, [1])
            else:
                digits[i+1] += 1
            digits[i] -= 10
            carry = True
    digits1 = digits1[::-1]
    digits2 = digits2[::-1]
    return digits[::-1], carry

def substraction(digits1, digits2):
    digits1 = digits1[::-1]
    digits2 = digits2[::-1]
    digits = digits1 - digits2
    borrow = False
    for i in range(len(digits)):
        if digits[i] < 0:
            assert i != len(digits) - 1
            digits[i+1] -= 1
            digits[i] += 10
            borrow = True
    digits1 = digits1[::-1]
    digits2 = digits2[::-1]
    return digits[::-1], borrow

def multiplication(digits1, digits2):
    digits1 = digits1[::-1]
    digits2 = digits2[::-1]
    digits = np.zeros(len(digits1) + len(digits2) - 1)
    for i in range(len(digits1)):
        for j in range(len(digits2)):
            digits[i+j] += digits1[i] * digits2[j]
    carry = False
    for i in range(len(digits)):
        if digits[i] >= 10:
            if i == len(digits) - 1:
                digits = np.append(digits, [1])
            else:
                digits[i+1] += 1
            digits[i] -= 10
            carry = True
    digits1 = digits1[::-1]
    digits2 = digits2[::-1]
    return digits[::-1], carry
