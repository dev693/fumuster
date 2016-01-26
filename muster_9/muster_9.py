import numpy as np

class Network:
    def __init__(self, inputs, alpha = 10.0, layers = 2):
        self.len = inputs
        self.layers = layers
        self.alpha = alpha
        self.o = []
        self.d = []

        self.w = []
        for i in range(layers):
            layer = layers - i
            self.w.append(np.random.rand(inputs + 1, layer))

    def clear_output(self):
        self.o = []
        for i in range(self.layers):
            self.o.append([])

    def sig(self, arr):
        e = np.exp(arr)
        return e / (1 - e)

    def forward(self, input):
        self.o = [input]

        for i in range(self.layers):
            w = self.w[i]
            input = np.append(np.array(input), [1])
            m = input * np.matrix(w)
            o = self.sig(m)
            self.o.append(o)
            input = o

        return input[0,0]

    def backprop(self, value):
        y = self.o[self.layers][0,0]
        d = y - value
        W = 1
        dW = []

        for i in range(self.layers-1, -1, -1):
            o = self.o[i+1]
            do = np.multiply(o, (1 - o)).T
            D = np.diagflat(do)
            d = D * W * d
            dW.append(-self.alpha * d * self.o[i])
            W = self.w[i][0:self.len,:]

        return list(reversed(dW))

    def apply(self, dW):
        for i in range(self.layers):
            w = self.w[i]
            dw = dW[i].T
            patch = np.matrix(np.zeros((1, dw.shape[1])))
            patched = np.vstack([dw, patch])
            self.w[i] = w + patched

    def learn(self, input, value):

        y = self.forward(input)

        dW = self.backprop(value)

        self.apply(dW)

        return y

    def test(self, input):
        return self.forward(input)

    def print(self):
        i = 0
        for w in self.w:
            print("\\w_%s = \\begin{pmatrix}" % i)
            print("%s" % w)
            print("\\end{pmatrix}")
            i += 1



n = Network(2)
for i in range(100):
    n.learn(np.array([0,0]), 0)
    n.learn(np.array([0,1]), 1)
    n.learn(np.array([1,0]), 1)
    n.learn(np.array([1,1]), 0)
n.print()

print("n(%s,%s) = %s" % (0, 0, n.test(np.array([0,0]))))
print("n(%s,%s) = %s" % (0, 1, n.test(np.array([0,1]))))
print("n(%s,%s) = %s" % (1, 0, n.test(np.array([1,0]))))
print("n(%s,%s) = %s" % (1, 1, n.test(np.array([1,1]))))

