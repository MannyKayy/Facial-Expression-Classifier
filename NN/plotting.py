from util import LoadData, Load, Save, DisplayPlot
import numpy as np
import matplotlib.pyplot as plt

#stats_fname = 'nn_stats.npz'
#stats_fname = 'cnn_stats.npz'

#stats_fname = 'cnn_stats_32_eps99.npz'

#stats_fname = 'cnn_stats_32_mom09.npz'

#stats_fname = 'cnn_stats_32_batch150.npz'

#stats_fname = 'nn_stats_33_50_100.npz'

stats_fname = 'cnn_stats_33_5_10.npz'

stats = Load(stats_fname)

train = np.array(stats['train_ce'])
valid = np.array(stats['valid_ce'])

plt.figure(1)
plt.plot(train[:, 0], train[:, 1], 'b', label='Train', color='r')
plt.plot(valid[:, 0], valid[:, 1], 'g', label='Validation', color='c')
plt.xlabel('Epoch')

plt.ylabel('Cross Entropy')

plt.title('Training and Validation Entropies of CNN with Filters=5;10')

plt.legend()
plt.show()

plt.savefig('3_3_5_10_cnn_ce')

print('Done')

plt.figure(2)
train = np.array(stats['train_acc'])
valid = np.array(stats['valid_acc'])

plt.plot(train[:, 0], train[:, 1], 'b', label='Train', color='r')
plt.plot(valid[:, 0], valid[:, 1], 'g', label='Validation', color='c')
plt.xlabel('Epoch')

plt.ylabel('Accuracy')

plt.title('Training and Validation Accuracies of CNN with Filters=5;10')

plt.legend()
plt.show()

plt.savefig('3_3_5_10_cnn_accu')

print('Done')
