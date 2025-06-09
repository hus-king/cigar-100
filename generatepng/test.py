import pickle

# 替换为你自己的路径
meta = pickle.load(open('./cifar-100-python/meta', 'rb'))

print("Meta keys:", meta.keys())