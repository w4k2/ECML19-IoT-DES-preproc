from RBO import RBO
import helper as h

X, y = h.datImport("ecoli4.dat")

print(X.shape)
print(y.shape)

rbo = RBO()
X, y = rbo.fit_sample(X, y)

print(X.shape)
print(y.shape)