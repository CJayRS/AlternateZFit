import pickle

f = open("z_outdata.pkl", "rb")
data = pickle.load(f)
print(data)
f.close()
