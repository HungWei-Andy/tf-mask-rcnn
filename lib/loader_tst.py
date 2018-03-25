from loader.COCOLoader import COCOLoader
data = COCOLoader(is_train=False, shuffle=False)
a,b,c,d,e = data.batch(0)
print(a)
print(b)
print(c)
print(d)
print(e)
