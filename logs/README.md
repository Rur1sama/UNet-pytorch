# logs folder
```PYTHON
def draw_result_visualization(folder, epoch_result):
    # the change of loss
    np.savetxt(os.path.join(folder, "epoch.txt"), epoch_result, fmt="%.4f", delimiter=',', newline='\n')
    with plt.ioff():
      plt.figure()
      plt.plot(epoch_result[:][0], epoch_result[:][1])
      plt.title("the change of the loss")
      plt.xlabel("epoch")
      plt.ylabel("loss")
      plt.savefig(os.path.join(folder, "loss_change.png"))
      plt.figure()
      plt.plot(epoch_result[:][0], epoch_result[:][2])
      plt.title("the change of the accuracy")
      plt.xlabel("epoch")
      plt.ylabel("accuracy")
      plt.savefig(os.path.join(folder, "accuracy_change.png"))
      plt.figure()
      plt.plot(epoch_result[:][0], epoch_result[:][3])
      plt.title("the change of the MIoU")
      plt.xlabel("epoch")
      plt.ylabel("MIoU")
      plt.savefig(os.path.join(folder, "MIoU_change.png"))
```