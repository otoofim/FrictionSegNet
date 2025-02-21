import sys
sys.path.insert(1, './architecture')
sys.path.insert(2, './dataLoaders')
import torchvision.models as models
from torchvision import transforms
from CityscapesLoader import *
from MapillaryDataLoader import *
from MapillaryIntendedObjs import *
from RSCDDataLoader import *
from VolvoDataLoader import *
from volvoDataLoader_onFly import *
import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset
import torch.nn.functional as F
from matplotlib import pyplot as plt
import torch.optim as optim
from tqdm import tqdm
import torchvision
import wandb
import os
import warnings
from statistics import mean
from collections import Counter
from pathlib import Path
from ProUNet import *
from torch.distributions import Normal, Independent, kl, MultivariateNormal
from torchmetrics.classification import MulticlassCalibrationError
from sklearn import metrics
from torch.utils.data.sampler import SubsetRandomSampler

# import neptune



warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore")

#mapillary+ice_11cat_geco


def l2_regularisation(m):
    l2_reg = None

    for W in m.parameters():
        if l2_reg is None:
            l2_reg = W.norm(2)
        else:
            l2_reg = l2_reg + W.norm(2)
    return l2_reg






def saveModel(modelPath, modelName):

    if not os.path.exists(modelPath):
        os.makedirs(modelPath)

    torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'tr_loss': tr_loss,
    'val_loss': val_loss,
    'hyper_params': kwargs,
    }, os.path.join(newpath, "{}.pth".format(modelName)))

def train(**kwargs):

    kwargs["model_add"] = os.path.join("checkpoints", "{}_{}".format(kwargs["project_name"], kwargs["run_name"]))
    
    
    if kwargs["continue_tra"]["enable"]:
        
        wandb.init(config = kwargs, project = kwargs["project_name"], entity = kwargs["entity"],
                    name = kwargs['run_name'], resume = "must", id = kwargs["continue_tra"]["wandb_id"])
        
        print("wandb resumed...")
    else:
        wandb.init(config = kwargs, project = kwargs["project_name"], entity = kwargs["entity"],
                    name = kwargs['run_name'], resume = "allow")

              

    traDatasets = []
    valDatasets = []

    train_sampler = None
    valid_sampler = None
        
    if kwargs["datasetConfig"]["mapillaryRootPath"] != "":
        print("Mapillary dataset added!")
        traDatasets.append(mapillaryDataLoader(**{**kwargs["datasetConfig"], **{"mode":"train"}}))
        valDatasets.append(mapillaryDataLoader(**{**kwargs["datasetConfig"], **{"mode":"val"}}))
    if kwargs["datasetConfig"]["RSCDRootPath"] != "":
        print("RSCD dataset added!")
        traDatasets.append(RSCDDataLoader(**{**kwargs["datasetConfig"], **{"mode":"train"}}))
        valDatasets.append(RSCDDataLoader(**{**kwargs["datasetConfig"], **{"mode":"val"}}))
    if kwargs["datasetConfig"]["volvoRootPath"] != "":
        print("Volvo dataset added!")

        tmpdataset = volvo_onFly(**{**kwargs["datasetConfig"], **{"mode":"train"}})

        dataset_size = tmpdataset.__len__()
        indices = list(range(dataset_size))
        split = int(np.floor(0.2 * dataset_size))
        np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]

        # Creating PT data samplers and loaders:
        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(val_indices)

        traDatasets.append(tmpdataset)
        valDatasets.append(tmpdataset)

    if kwargs["datasetConfig"]["cityscapesRootPath"] != "":
        print("CityScapes dataset added!")
        traDatasets.append(CityscapesLoader(**{**kwargs["datasetConfig"], **{"mode":"train"}}))
        valDatasets.append(CityscapesLoader(**{**kwargs["datasetConfig"], **{"mode":"val"}}))
    if kwargs["datasetConfig"]["ACDCRootPath"] != "":
        print("ACDC dataset added!")
        traDatasets.append(ACDC_onFly(**{**kwargs["datasetConfig"], **{"mode":"train"}}))
        valDatasets.append(ACDC_onFly(**{**kwargs["datasetConfig"], **{"mode":"val"}}))
    
    train_dev_sets = ConcatDataset(traDatasets)
    val_dev_sets = ConcatDataset(valDatasets)
    
    train_loader = DataLoader(dataset = train_dev_sets, batch_size = kwargs["batch_size"], shuffle = True if train_sampler is None else False, drop_last = True, sampler = train_sampler)
    val_loader = DataLoader(dataset = val_dev_sets, batch_size = kwargs["batch_size"], shuffle = True if valid_sampler is None else False, drop_last = True, sampler = valid_sampler)
    

    if kwargs['device'] == "cpu":
        device = torch.device("cpu")
        print("Running on the CPU")
    elif kwargs['device'] == "gpu":
        device = torch.device(kwargs['device_name'])
        print("Running on the GPU")


    model = ProUNet(gecoConfig = kwargs["GECO"], num_classes = traDatasets[0].get_num_classes(), LatentVarSize = kwargs["latent_dim"], beta = kwargs["beta"], training = True, num_samples = kwargs["num_samples"], device = device)
        
    if kwargs["continue_tra"]["enable"]:
        model.load_state_dict(torch.load(os.path.join(kwargs["model_add"], "{}.pth".format(kwargs["continue_tra"]["which_model"])), map_location=device)['model_state_dict'])
        print("model state dict loaded...")

        
    elif kwargs["pretrained"]["enable"]:
        model.load_state_dict(torch.load(os.path.join(kwargs["pretrained"]["model_add"], "{}.pth".format(kwargs["pretrained"]["which_model"])), map_location=device)['model_state_dict'])
        print("model state dict loaded...")

        
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr =  kwargs["learning_rate"], weight_decay = kwargs["momentum"])

    if kwargs["continue_tra"]["enable"]:
        optimizer.load_state_dict(torch.load(os.path.join(kwargs["model_add"], "{}.pth".format(kwargs["continue_tra"]["which_model"])), map_location = device)['optimizer_state_dict'])
        print("optimizer state dict loaded...")
    

    tr_loss = {"Total training loss":[], "KL training loss":[], "Reconstruction training loss":[], "mIoU training":[], "ECE training":[], "MCE training":[], "RMSCE training":[], "lambda fri coeficient":[], "lambda seg coeficient":[], "NLL training":[], "Brie training":[], "Regression training loss":[], "Confusion Matrix training":[]}
    val_loss = {"Total validation loss":[], "KL validation loss":[], "Reconstruction validation loss":[], "mIoU validation":[], "ECE validation":[], "MCE validation":[], "RMSCE validation":[], "NLL validation":[], "Brie validation":[], "Regression validation loss":[], "Confusion Matrix validation":[]}
    
    best_val = -1
    val_every = 2
    wandb.watch(model)

    start_epoch = 0
    resume_position = 0
    end_epoch = kwargs["epochs"]
    total = kwargs["epochs"]
    
#     torch.autograd.set_detect_anomaly(True)

    if kwargs["continue_tra"]["enable"]:
        start_epoch = torch.load(os.path.join(kwargs["model_add"], "{}.pth".format(kwargs["continue_tra"]["which_model"])))['epoch'] + 1
        resume_position = start_epoch

    with tqdm(range(start_epoch, end_epoch), initial = resume_position, total = total, unit="epoch", leave = True, position = 0) as epobar:
        for epoch in epobar:
                
                epobar.set_description("Epoch {}".format(epoch + 1))
                epobar.set_postfix(ordered_dict = {'tr_loss':np.mean(tr_loss["Total training loss"]), 'val_loss':np.mean(val_loss["Total validation loss"])})
                tr_loss = {"Total training loss":[], "KL training loss":[], "Reconstruction training loss":[], "mIoU training":[], "ECE training":[], "MCE training":[], "RMSCE training":[], "lambda fri coeficient":[], "lambda seg coeficient":[], "NLL training":[], "Brie training":[], "Regression training loss":[], "Confusion Matrix training":[]}

                
                out = None
                images = None
                labels = None

                with tqdm(train_loader, unit="batch", leave = False) as batbar:
                    for i, batch in enumerate(batbar):
                        
                        
                        batbar.set_description("Batch {}".format(i + 1))
                        optimizer.zero_grad()
                        model.train()
                        
                        #forward pass
                        batchImg = batch['image'].to(device)
                        batchLabel = batch['label'].to(device)
                        FriLabel = batch['FriLabel'].to(device)
                        seg, priorDists, posteriorDists, friPred = model(batchImg, batchLabel, FriLabel)
                        loss, kl_mean, kl_losses, rec_loss, miou, ious, l1Loss, l2Loss, l3Loss, regLoss, CM = model.loss(batchLabel, seg, priorDists, posteriorDists, FriLabel, friPred)

                        
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., foreach = True)
                        optimizer.step()


                        tr_loss["Total training loss"].append(loss.detach().cpu().item())
                        tr_loss["KL training loss"].append(kl_mean.detach().cpu().item())
                        tr_loss["Reconstruction training loss"].append(rec_loss.detach().cpu().item())
                        tr_loss["mIoU training"].append(torch.mean(miou).detach().cpu().item())
                        tr_loss["ECE training"].append(l1Loss.detach().cpu().item())
                        tr_loss["MCE training"].append(l2Loss.detach().cpu().item())
                        tr_loss["RMSCE training"].append(l3Loss.detach().cpu().item())
                        if kwargs["GECO"]["enable"]:
                            tr_loss["lambda fri coeficient"].append(model.geco.lambda_f.detach().cpu().item())
                            tr_loss["lambda seg coeficient"].append(model.geco.lambda_s.detach().cpu().item())
                        tr_loss["NLL training"].append(nn.NLLLoss()(F.log_softmax(seg), torch.argmax(batchLabel, 1)).detach().cpu().item())
                        tr_loss["Brie training"].append(torch.mean(torch.square(seg - batchLabel)).detach().cpu().item())
                        tr_loss["Regression training loss"].append(torch.mean(regLoss.detach().cpu()).item())
                        tr_loss["Confusion Matrix training"].append(CM)
                        
                        
                        for layer in kl_losses.items():
                            if "KL training loss layer{}".format(layer[0]) in tr_loss.keys():
                                tr_loss["KL training loss layer{}".format(layer[0])].append(torch.mean(layer[1]).detach().cpu().item())
                            else:
                                tr_loss["KL training loss layer{}".format(layer[0])] = []
                                tr_loss["KL training loss layer{}".format(layer[0])].append(torch.mean(layer[1]).detach().cpu().item())

                                
                                
                        for layer in ious.items():
                            if "iou of {} training".format(layer[0]) in tr_loss.keys():
                                tr_loss["iou of {} training".format(layer[0])].append(torch.mean(layer[1]).detach().cpu().item())
                            else:
                                tr_loss["iou of {} training".format(layer[0])] = []
                                tr_loss["iou of {} training".format(layer[0])].append(torch.mean(layer[1]).detach().cpu().item())
                                
                        
                        if i%1000 == 0:
                            
                            if not os.path.exists(kwargs["model_add"]):
                                Path(kwargs["model_add"]).mkdir(parents=True)

                            torch.save({
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'tr_loss': tr_loss,
                            'val_loss': val_loss,
                            'hyper_params': kwargs,
                            }, os.path.join(kwargs["model_add"], "iterative.pth"))
                        
                        

                org_img = {'input': wandb.Image(batch['image'][-5:].detach().cpu()),
                "ground truth": wandb.Image(batch['seg'][-5:].detach().cpu()),
                "prediction": wandb.Image(traDatasets[0].prMask_to_color(seg[-5:].detach().cpu()))
                 }
                wandb.log(org_img)
                
                
                plt.rcParams.update({'font.size': 22})
                plt.rcParams["font.weight"] = "bold"
                plt.rcParams["axes.labelweight"] = "bold"
                fig, ax = plt.subplots(figsize=(22,20), dpi=100)
                ax.set_xlabel('Predicted labels')
                ax.set_ylabel('True labels')
#                 CMEpoch = torch.mean(torch.stack(tr_loss["Confusion Matrix training"]), axis = 0)
                CMEpoch = torch.nansum(torch.stack(tr_loss["Confusion Matrix training"]), dim = 0)
                CMEpoch = torch.round(CMEpoch/CMEpoch.sum(dim=0, keepdim=True), decimals=2)
                CMEpoch = torch.nan_to_num(CMEpoch).cpu().detach().numpy()
                cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = CMEpoch, display_labels = list(classIds.keys())).plot(xticks_rotation = 45, ax = ax)
                wandb.log({"Confusion matrix training": plt})
                

                for key in tr_loss.keys():
                    if "iou" in key or "mIoU" in key:
                        wandb.log({key: torch.tensor([tr_loss[key]]).nanmean(), "epoch": epoch + 1})
                    elif not "Confusion Matrix" in key:
                        wandb.log({key: torch.mean(torch.tensor(tr_loss[key])), "epoch": epoch + 1})
                
                if ((epoch+1) % val_every == 0):
                    with tqdm(val_loader, unit="batch", leave = False) as valbar:
                        with torch.no_grad():
                            val_loss = {"Total validation loss":[], "KL validation loss":[], "Reconstruction validation loss":[], "mIoU validation":[], "ECE validation":[], "MCE validation":[], "RMSCE validation":[], "NLL validation":[], "Brie validation":[], "Regression validation loss":[], "Confusion Matrix validation":[]}
                            
                            for i, batch in enumerate(valbar):
                                                                             
                                valbar.set_description("Val_batch {}".format(i + 1))
                                model.eval()
                                optimizer.zero_grad()
                                loss_sum = []
                                kl_sum = []
                                rec_sum = []
                                iou_sum = []
                                l1Loss_sum = []
                                l2Loss_sum = []
                                l3Loss_sum = []
                                kl_losses_sum = {}
                                ious_sum = {}
                                NLL_sum = []
                                Brie_sum = []
                                reg_sum = []
                                
                                batchImg = batch['image'].to(device)
                                batchLabel = batch['label'].to(device)
                                FriLabel = batch['FriLabel'].to(device)
                             
                                samples, priors, posteriorDists, friPreds = model.evaluation(batchImg, batchLabel, FriLabel)

                                for i, sample in enumerate(samples):
                                                                             
                                    loss, kl_mean, kl_losses, rec_loss, miou, ious, l1Loss, l2Loss, l3Loss, regLoss, CM = model.loss(batchLabel.to(device), sample, priors, posteriorDists, FriLabel, friPreds[i])
                                    

                                    iou_sum.append(miou)
                                    l1Loss_sum.append(l1Loss)
                                    l2Loss_sum.append(l2Loss)
                                    l3Loss_sum.append(l3Loss)
                                    loss_sum.append(loss)
                                    kl_sum.append(kl_mean)
                                    rec_sum.append(rec_loss)
                                    NLL_sum.append(nn.NLLLoss()(F.log_softmax(sample), torch.argmax(batchLabel, 1)))
                                    Brie_sum.append(torch.mean(torch.square(sample - batchLabel)))
                                    reg_sum.append(regLoss)
                                    
                                    for layer in kl_losses.items():
                                        if layer[0] in kl_losses_sum.keys():
                                            kl_losses_sum[layer[0]].append(torch.mean(layer[1].detach().cpu()).item())
                                        else:
                                            kl_losses_sum[layer[0]] = []
                                            kl_losses_sum[layer[0]].append(torch.mean(layer[1].detach().cpu()).item())
                                    

                                    
                                    for layer in ious.items():
                                        if layer[0] in ious_sum.keys():
                                            ious_sum[layer[0]].append(torch.mean(layer[1].detach().cpu()).item())
                                        else:
                                            ious_sum[layer[0]] = []
                                            ious_sum[layer[0]].append(torch.mean(layer[1].detach().cpu()).item())
                                    
                                    
                                    

                                val_loss["Total validation loss"].append(torch.mean(torch.stack(loss_sum).detach().cpu()).item())
                                val_loss["KL validation loss"].append(torch.mean(torch.stack(kl_sum).detach().cpu()).item())
                                val_loss["Reconstruction validation loss"].append(torch.mean(torch.stack(rec_sum).detach().cpu()).item())
                                val_loss["mIoU validation"].append(torch.mean(torch.stack(iou_sum).detach().cpu()).item())
                                val_loss["ECE validation"].append(torch.mean(torch.stack(l1Loss_sum).detach().cpu()).item())
                                val_loss["MCE validation"].append(torch.mean(torch.stack(l2Loss_sum).detach().cpu()).item())
                                val_loss["RMSCE validation"].append(torch.mean(torch.stack(l3Loss_sum).detach().cpu()).item())
                                val_loss["NLL validation"].append(torch.mean(torch.stack(NLL_sum).detach().cpu()).item())
                                val_loss["Brie validation"].append(torch.mean(torch.stack(Brie_sum).detach().cpu()).item())
                                val_loss["Regression validation loss"].append(torch.mean(torch.stack(reg_sum).detach().cpu()).item())
                                val_loss["Confusion Matrix validation"].append(CM)
                                
                                
                                for layer in kl_losses_sum.items():
                                    if "KL validation loss layer{}".format(layer[0]) in val_loss.keys():
                                        val_loss["KL validation loss layer{}".format(layer[0])].append(torch.mean(torch.tensor(layer[1]).detach().cpu()).item())
                                    else:
                                        val_loss["KL validation loss layer{}".format(layer[0])] = []
                                        val_loss["KL validation loss layer{}".format(layer[0])].append(torch.mean(torch.tensor(layer[1])).detach().cpu().item())

                                  
                                
                                for layer in ious_sum.items():
                                    if "iou of {} validation".format(layer[0]) in val_loss.keys():
                                        val_loss["iou of {} validation".format(layer[0])].append(torch.tensor(layer[1]).detach().cpu().nanmean().item())
                                    else:
                                        val_loss["iou of {} validation".format(layer[0])] = []
                                        val_loss["iou of {} validation".format(layer[0])].append(torch.tensor(layer[1]).detach().cpu().nanmean().item())

                                        
                                                                                         
                        wandb.log({"epoch": epoch + 1,
                                   "input_val": wandb.Image(batch['image'][-5:].detach().cpu()),
                                   "ground truth val": wandb.Image(batch['seg'][-5:].detach().cpu()),
                                   "prediction val": wandb.Image(traDatasets[0].prMask_to_color(torch.mean(samples.detach().cpu(),0))[-5:])
                                  })

                        
                        plt.rcParams.update({'font.size': 22})
                        plt.rcParams["font.weight"] = "bold"
                        plt.rcParams["axes.labelweight"] = "bold"
                        fig, ax = plt.subplots(figsize=(22,20), dpi=100)
                        ax.set_xlabel('Predicted labels')
                        ax.set_ylabel('True labels')
#                         CMEpoch = torch.mean(torch.stack(val_loss["Confusion Matrix validation"]), axis = 0)
                        CMEpoch = torch.nansum(torch.stack(val_loss["Confusion Matrix validation"]), dim = 0)
                        CMEpoch = torch.round(CMEpoch/CMEpoch.sum(dim=0, keepdim=True), decimals=2)
                        CMEpoch = torch.nan_to_num(CMEpoch).cpu().detach().numpy()
                        cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = CMEpoch, display_labels = list(classIds.keys())).plot(xticks_rotation = 45, ax = ax)                
                        wandb.log({"Confusion matrix validation": plt})
                        
                        
                        for key in val_loss.keys():
                            if "iou" in key or "mIoU" in key:
                                wandb.log({key: torch.tensor(val_loss[key]).nanmean(), "epoch": epoch + 1})
                            elif not "Confusion Matrix" in key:
                                wandb.log({key: torch.mean(torch.tensor(val_loss[key])), "epoch": epoch + 1})

                        if torch.mean(torch.tensor(val_loss["mIoU validation"])) > best_val:
                            
                            best_val = torch.mean(torch.tensor(val_loss["mIoU validation"]))
                            
                            if not os.path.exists(kwargs["model_add"]):
                                Path(kwargs["model_add"]).mkdir(parents=True)

                            torch.save({
                                'epoch': epoch,
                                'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'tr_loss': tr_loss,
                                'val_loss': val_loss,
                                'hyper_params': kwargs,
                                }, os.path.join(kwargs["model_add"], "best.pth"))
                            
                            
