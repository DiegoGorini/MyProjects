# This is the code for the plot. Each point (x, y) represents the
# accuracy and the time it took to train the model for the following
# values of the hidden units: 16, 32, 64, 128, 256 & 512.

# I manually transcribed the values from the written report.
# I know this sucks, but I hope to automatize this task somehow for the
# next assigments.

import matplotlib.pyplot as plt
# 1 Layer
plt.plot([55.28,56.85,57.50,58.04,58.11,58.34], [33,37,50,77,152,286])
# 2 Layers
plt.plot([54.21,56.58,57.52,58.52,59.24,59.23], [41,49,68,110,189,379])
# 3 Layers
plt.plot([52.80,55.04,57.22,58.51,58.98,58.70], [43,50,70,120,220,502])
# 4 Layers
plt.plot([49.49,54.30,57.18,57.71,58.51,58.38], [40,43,50,88,237,624])
# 5 Layers
plt.plot([49.37,52.12,55.78,56.70,56.27,56.45], [43,50,62,148,310,833])
# Legends & Labels
plt.title('Performance/Time for different No. of Layers')
plt.xlabel('Accuracy %')
plt.ylabel('Time (in secs.)')
plt.legend(['1 Layer', '2 Layers', '3 Layers', '4 Layers', '5 Layers'])
plt.plot()
plt.savefig('Performance_over_Time')
