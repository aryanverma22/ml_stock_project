a=[1,8,9,0,7,6]
b=[2,77,99,66,44,23]
c=[3,90,75,42,86,14]
arr=[]
all=[a, b, c]
print(all)
arr=[]

for j in range(len(a)):
    temp=[]
    for x in all:
        temp.append(x[j])
    arr.append(temp)

print(arr)
for i in range(8,0,-1):
    print(i)