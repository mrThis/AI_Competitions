from tqdm import tqdm,tqdm_gui
import time
for k in tqdm(list(range(10000)),leave=False,ascii=True):
    time.sleep(0.001)
