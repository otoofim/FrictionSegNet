import torch
import os


# Old function to save the model -- not used in new codebase but kept for reference
def saveModel(
    modelPath, modelName, epoch, model, optimizer, tr_loss, val_loss, kwargs, newpath
):
    if not os.path.exists(modelPath):
        os.makedirs(modelPath)

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "tr_loss": tr_loss,
            "val_loss": val_loss,
            "hyper_params": kwargs,
        },
        os.path.join(newpath, "{}.pth".format(modelName)),
    )
