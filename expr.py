from torch.utils.data import Dataset
import numpy as np
from numpy.random import MT19937, RandomState, SeedSequence

def generate_addition(a, b):
    text = "%d+%d;" % (a, b)
    carry = 0
    s = a+b
    while True:
        text += "%d%d%d" % (a % 10, b % 10, carry)
        if a == 0 and b == 0 and carry == 0:
            break
        text += "%d" % ((a+b+carry)%10)
        carry = (a%10+b%10+carry)//10
        a = a//10
        b = b//10
    text += "=%d" % s
    return text

def generate_multiplication_single(a, b):
    if a < 10:
        return "%d*%d;=%d" % (a, b, a*b)
    product = a * b
    text = "%d*%d;" % (a, b)
    prods = []
    while a > 0:
        prods.append((a % 10) * b)
        text += generate_multiplication_single(a % 10, b)
        text += ";"
        a = a // 10
    # Do additions
    s = prods[0]
    for i in range(1, len(prods)):
        s = s // 10
        text += generate_addition(s, prods[i])
        text += ";"
        s += prods[i]
    text += "=%d" % product
    return text

def generate_multiplication(a, b):
    if b < 10:
        return generate_multiplication_single(a, b)
    product = a * b
    text = "%d*%d;" % (a, b)
    # Single digit multiplications
    prods = []
    while b > 0:
        prods.append(a * (b % 10))
        text += generate_multiplication_single(a, b % 10)
        text += ";"
        b = b // 10
    # Do additions
    s = prods[0]
    for i in range(1, len(prods)):
        s = s // 10
        text += generate_addition(s, prods[i])
        text += ";"
        s += prods[i]
    text += "=%d" % product
    return text

def get_multiplication_answer(s):
    try:
        eq = s.rindex('=')
        ed = s.rindex('$')
        return int(s[eq+1:ed])
    except:
        return None

class ExpressionDataset(Dataset):
    def __init__(self, count, begin=0):
        super(ExpressionDataset, self).__init__()
        self.count = count
        self.begin = begin
        
    def __getitem__(self, i):
        assert i < self.count and i >= 0, "Out of bounds"
        rng = np.random.RandomState(i+self.begin)
        x = rng.randint(10000, 99999, 2)
        return generate_multiplication(x[0], x[1]) + '$'
        
    def __len__(self):
        return self.count
    
class TestDataset(Dataset):
    def __init__(self, count, begin=0):
        super(ExpressionDataset, self).__init__()
        self.count = count
        self.begin = begin
        
    def __getitem__(self, i):
        assert i < self.count and i >= 0, "Out of bounds"
        rng = np.random.RandomState(i+self.begin)
        x = rng.randint(10000, 99999, 2)
        return ((x[0], x[1]), x[0] * x[1])
        
    def __len__(self):
        return self.count