with open('report3.csv' ,'r') as f:
    k=f.readlines()

with open('report.csv','a') as f:
    for o in range(len(k)):
        i=k[o]
        i=i.split(',')
        i[0]=str(int(i[0])+18100)
        i=(','.join(i))
        
        i=i.split('original,')
        print(i)
        oo=i[1].split(',')
        oo[0]=str(int(oo[0])+50)
        i[1]=','.join(oo)
        p='original,'.join(i)
        f.write(p)