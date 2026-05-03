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
        other = other if isinstance(other, Value) else Value(other) 
        out = Value(self.data + other.data, (self, other), '+')        

        def _backward(): 
            self.grad += out.grad 
            other.grad += out.grad 
        out. _backward = _backward 
        return out 

    def __radd__(self, other): 
        return self + other         

    def __mul__(self, other): 
        other = other if isinstance(other, Value) else Value(other)
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

    def __neg__(self): 
        return self * Value(-1.0)
    
    def __sub__(self, other): 
        return self + (-other) 



    def tanh(self): 
        import math 
        x = self.data 
        t = (math.exp(2*x)-1)/(math.exp(2*x)+1) 
        out = Value(t,(self,),'tanh') 

        def _backward(): 
            self.grad += (1 - t**2) * out.grad 
        out._backward = _backward 
        return out     

    def backward(self): 
        topo = [] 
        visited = set() 

        def build_topo(v): 
            if v not in visited: 
                visited.add(v) 
                for child in v._prev: 
                    build_topo(child)
                topo.append(v) 

        build_topo(self) 

        self.grad = 1.0 # dL/dL = 1, the starting gradient
        for node in reversed(topo): # walk backwards through the graph
            node._backward()  # each node sends gradients to its parents






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

# a = Value(0.0)
# b = a.tanh()
# print(b)        # expect tanh(0) = 0.0

# b.grad = 1.0
# b._backward()
# print(a.grad)   # expect 1 - 0² = 1.0

# a = Value(1.0)
# b = a.tanh()
# print(b)        # expect tanh(1) ≈ 0.7616

# b.grad = 1.0
# b._backward()
# print(a.grad)   # expect 1 - 0.7616² ≈ 0.42

# a = Value(0.5)
# b = Value(0.3)
# c = a * b       # c = 6.0
# d = c + Value(1.0)  # d = 7.0
# e = d.tanh()    # e = tanh(7.0) ≈ 1.0
# e.backward() 
# print('a.grad:', a.grad)
# print('b.grad:', b.grad)
# print('c.grad:', c.grad)
# print('d.grad:', d.grad)
# print('e.grad:', e.grad)

 


  