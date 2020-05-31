import pandas as pd
import matplotlib.pyplot as plt

RGB = (0.4157, 0.2784, 0.3333)
TRAIN_RESULT_PATH = '../data/train_result/train_step_result.csv'

train_result = pd.read_csv(TRAIN_RESULT_PATH, delimiter=',', encoding='cp949')
losses = train_result['loss']
cers = train_result['cer']

plt.title('Visualization of training (cer)')
plt.plot(cers, color=RGB, label='cers')
plt.xlabel('step (unit : 1000)', fontsize='x-large')
plt.ylabel('cer', fontsize='x-large')
plt.show()
