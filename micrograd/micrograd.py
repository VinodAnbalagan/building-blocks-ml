class Value:
    def __init__(self, data, _children=(), _op=''):
        self.data = data 
        self.grad = 0.0 
        self._backward = lambda:None 
        self._prev = set(_children) 
        self._op = _op 

    def __repr__(self): 
        return f"Value(data={self.data}, grad={self.grad})" 

    def __add__(self, other): 
        out = Value(self.data + other.data, (self, other), '+')        

        def _backward(): 
            self.grad += out.grad 
            other.grad += out.grad 
        out. _backward = _backward 
        return out     

    def __mul__(self, other): 
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad 
            other.grad += self.data * out.grad 
        out. _backward = _backward 
        return out  

    def __pow__(self, other): 
        assert isinstance(other, (int, float)) # only support int/float powers for now 
        out = Value(self.data ** other, (self,), f'**{other}')

        def _backward():
            self.grad += (other * self.data **(other - 1)) * out.grad 
        out._backward = _backward 
        return out 

    def tanh(self): 
        import math 
        x = self.data 
        t = (math.exp(2*x)-1)/(math.exp(2*x)+1) 
        out = Value(t,(self,),'tanh') 

        def _backward(): 
            self.grad += (1 - t**2) * out.grad 
        out._backward = _backward 
        return out     










# --- Test --- 
# a = Value(2.0) 
# b = Value(3.0) 
# c = a + b 

# c.grad = 1.0 
# c._backward() 
# print(a.grad) # expect 1.0
# print(b.grad)  # expect 1.0

# test multiplication — fresh values
# a = Value(2.0)
# b = Value(3.0)
# c = a * b
# c.grad = 1.0
# c._backward()
# print(a.grad)  # expect 3.0
# print(b.grad)  # expect 2.0

# Pow 
#a = Value(2.0)
#b = a ** 3
#print(b)          # expect 8.0

#b.grad = 1.0
#b._backward()
#print(a.grad)     # expect 3 * 2^2 * 1.0 = 12.0

a = Value(0.0)
b = a.tanh()
print(b)        # expect tanh(0) = 0.0

b.grad = 1.0
b._backward()
print(a.grad)   # expect 1 - 0² = 1.0

a = Value(1.0)
b = a.tanh()
print(b)        # expect tanh(1) ≈ 0.7616

b.grad = 1.0
b._backward()
print(a.grad)   # expect 1 - 0.7616² ≈ 0.42