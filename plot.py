import numpy as np
import matplotlib.pyplot as plt
import csv

the_date   = '20171115'
experiment = 'tf_task_training'
logfile    = 'experiment/' + experiment + '.csv'

plotname1  = 'tf_task_training-accuracy'
plotname2  = 'tf_task_training-loss'


def load_logs(logfile, discard_header=False):
    iteration, train_accuracy, valid_accuracy, loss = [], [], [], []
    with open(logfile) as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            if discard_header:
                discard_header = False
                continue
            r_it, r_train_acc, r_valid_acc, r_loss = row
            iteration.append(int(r_it))
            train_accuracy.append(float(r_train_acc))
            valid_accuracy.append(float(r_valid_acc))
            loss.append(float(r_loss))
    return iteration, train_accuracy, valid_accuracy, loss

logfile = 'experiment/' + experiment + '.csv'
it_fc100, tr_acc_fc100, va_acc_fc100, loss_fc100 = load_logs(logfile, discard_header=True)
# iteration, training accuracy, validation accuracy, loss for FC100-ReLU-Dropout-FC3 configuration


# Colours
col_pink   = '#E13375'
col_blue   = '#095998'
col_orange = '#F17B18'
col_green  = '#29C78D'

col_red    = '#d91c22'
col_cyan   = '#1cd9cc'
col_green  = '#81d91c'

#--------------------------------------------------------------------------------
# Accuracy
#--------------------------------------------------------------------------------

# Set figure size
fig, ax = plt.subplots(figsize=(12,6.75))

# Plotting multiple lines
ax.plot(it_fc100, va_acc_fc100, linewidth=2, color=col_blue, label='Validation Accuracy')
ax.plot(it_fc100, tr_acc_fc100, '--', linewidth=2, color=col_red, label='Training Accuracy')

# Plot title
ax.set_title('Task Training Accuracy', y=0.93, x=0.15, fontsize=16)

# Axes labels, fontsize (offset by adding y=0.0 etc to arguments)
ax.set_ylabel('Accuracy', fontsize=16)
ax.set_xlabel('Iteration', fontsize=16)

# Ticks
ax.minorticks_on()
ax.tick_params(axis='both', which='major', right='on', top='on', direction='in', labelsize=12, length=6)
ax.tick_params(axis='both', which='minor', right='on', top='on', direction='in', labelsize=12, length=4)

# Set range limit on axes
ax.set_ylim([0,1.2])
ax.set_xlim([0,3000])

# To set a grid
ax.grid(True)

# Legend, generated from plot details and labels
ax.legend(loc=1, prop={'size': 12})

# plt.show()
plt.savefig('./experiment/' + the_date + '-' + plotname1 + '.png', bbox_inches='tight')
print('Plotted Accuracy.')

#--------------------------------------------------------------------------------
# Loss
#--------------------------------------------------------------------------------

# Set figure size
fig, ax = plt.subplots(figsize=(12,6.75))

ax.plot(it_fc100, loss_fc100, linewidth=2, color=col_blue, label='Loss')

# Plot title
ax.set_title('Task Training Loss', y=0.93, x=0.15, fontsize=16)

# Axes labels, fontsize (offset by adding y=0.0 etc to arguments)
ax.set_ylabel('Loss', fontsize=16)
ax.set_xlabel('Iteration', fontsize=16)

# Ticks
ax.minorticks_on()
ax.tick_params(axis='both', which='major', right='on', top='on', direction='in', labelsize=12, length=6)
ax.tick_params(axis='both', which='minor', right='on', top='on', direction='in', labelsize=12, length=4)

# Set range limit on axes
# ax.set_ylim([0,1.2])
ax.set_xlim([0,3000])

# To set a grid
ax.grid(True)

# Legend, generated from plot details and labels
ax.legend(loc=1, prop={'size': 12})

# plt.show()
plt.savefig('./experiment/' + the_date + '-' + plotname2 + '.png', bbox_inches='tight')
print('Plotted Loss.')
