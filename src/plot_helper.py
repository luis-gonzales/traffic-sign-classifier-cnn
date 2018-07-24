import matplotlib.pyplot as plt

# Plotting helper
def plot_helper(ax, epochs, train_accs, val_accs, alpha_list, batch_size):
  font_sz = 8.5
  
  l1 = ax.plot(np.arange(1,epochs+1), train_accs, 'b', label='train')
  l2 = ax.plot(np.arange(1,epochs+1), val_accs, 'r', label='val')
  ax.set_xlabel('Epochs', fontsize=font_sz)
  ax.set_ylabel('Accuracy', fontsize=font_sz)
  ax.axis([0, epochs, 0.9, 1.02])
  ax.tick_params(labelsize=font_sz)
  ax.yaxis.grid()

  ax2 = ax.twinx()
  l3 = ax2.plot(np.arange(1,epochs+1), alpha_list, label='rate')
  ax2.axis([0, epochs, 0, 0.0012])
  ax2.set_ylabel('Learning Rate', fontsize=font_sz)
  ax2.ticklabel_format(style='sci', scilimits=(0,0))
  ax2.tick_params(labelsize=font_sz)
  ax2.xaxis.set_ticklabels([0, 50, 100, 150])
  ax2.yaxis.offsetText.set_fontsize(font_sz)
  
  # Legend
  lns = l1 + l2 + l3
  labs = [x.get_label() for x in lns]
  ax.legend(lns, labs, loc=(0.63,0.3), fontsize=font_sz-1)

  plt.xticks([0, 50, 100, 150])
  file_str = 'learning_curves/' + 'val-acc-' + str(val_accs[-1])[0:6] + \
             '-alpha-' + str(alpha_list[0])[0:7] + '-batch-' + str(batch_size) + '.png'
  plt.savefig(file_str, bbox_inches='tight', dpi = 300, transparent=True)
