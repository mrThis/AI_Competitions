import subprocess
import time
start_time=time.time()
pool=[]
for i in range(1,6):

    s1 = subprocess.Popen(['python', 'pre_cut.py', str(i)])
    pool.append(s1)

for s in pool:
    s.wait()

m=(time.time()-start_time)//60
print('用时{}小时{}分钟。'.format(m//60,m%60))