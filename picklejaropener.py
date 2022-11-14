import pickle

f = open("resample_compare.pkl", "rb")
data = pickle.load(f)
print(data)
f.close()
