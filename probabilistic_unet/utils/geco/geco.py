    
class MyGECO():

    def __init__(self,
                goal_fri = 0.08,
                goal_seg = 5.27,
                alpha = 0.9,
                beta = 0.9,
                step_size = 0.01,
                lambda_s = 1.0,
                lambda_f = 1.0,
                speedup = 10,
                beta_min = 1e-10,
                beta_max = 1e10
                ):
        
        
        self.goal_fri = torch.tensor(goal_fri)  
        self.goal_seg = torch.tensor(goal_seg)  
        self.alpha = torch.tensor(alpha)
        self.beta = torch.tensor(beta)
        self.step_size = torch.tensor(step_size)
        self.lambda_s = torch.tensor(lambda_s)
        self.lambda_f = torch.tensor(lambda_f)
        self.speedup = torch.tensor(speedup)
        self.beta_min = torch.tensor(beta_min)
        self.beta_max = torch.tensor(beta_max)
        
        # Initialize EMA values
        self.ema_seg = None
        self.ema_fri = None

    def loss(self, loss_fri, loss_seg, kl_loss):
        
        
        total_loss = self.lambda_s * loss_seg + self.lambda_f * loss_fri + kl_loss

        
        with torch.no_grad():
            
            if self.ema_seg is None:
                self.ema_seg = loss_seg.item()
                self.ema_fri = loss_fri.item()
            else:
                self.ema_seg = self.alpha * self.ema_seg + (1 - self.alpha) * loss_seg.item()
                self.ema_fri = self.beta * self.ema_fri + (1 - self.beta) * loss_fri.item()
        

            # Compute constraints
            C_seg = self.goal_seg - self.ema_seg
            C_fri = self.goal_fri - self.ema_fri
            
            
            if self.speedup is not None and C_seg > 0:
                self.lambda_s *= torch.exp(self.speedup * -C_seg * self.step_size)
            else:
                factor = torch.exp(-C_seg * self.step_size)
                self.lambda_s = (factor * self.lambda_s).clamp(self.beta_min, self.beta_max)


            if self.speedup is not None and C_fri > 0:
                self.lambda_f *= torch.exp(self.speedup * -C_fri * self.step_size)
            else:
                factor = torch.exp(-C_fri * self.step_size)
                self.lambda_f = (factor * self.lambda_f).clamp(self.beta_min, self.beta_max)
                
        # print(f"Loss: {total_loss.item():.4f}, Lambda_s: {self.lambda_s:.4f}, Lambda_f: {self.lambda_f:.4f}")

        return total_loss
    
class GECO():
    
# =========================== A2I Copyright Header ===========================
#
# Copyright (c) 2003-2021 University of Oxford. All rights reserved.
# Authors: Applied AI Lab, Oxford Robotics Institute, University of Oxford
#          https://ori.ox.ac.uk/labs/a2i/
#
# This file is the property of the University of Oxford.
# Redistribution and use in source and binary forms, with or without
# modification, is not permitted without an explicit licensing agreement
# (research or commercial). No warranty, explicit or implicit, provided.
#
# =========================== A2I Copyright Header ===========================
#https://github.com/applied-ai-lab/genesis/blob/master/utils/geco.py
    def __init__(self, goal, step_size, device, alpha=0.99, beta_init=1.,
                 beta_min=1e-10, speedup=None):
        self.err_ema = None
        self.goal = goal
        self.step_size = step_size
        self.alpha = alpha
        self.beta = torch.tensor(beta_init)
        self.beta_min = torch.tensor(beta_min)
        self.beta_max = torch.tensor(1e10)
        self.speedup = speedup
        self.device = device
        self.to_device()

    def to_device(self):
        self.beta = self.beta.to(device=self.device)
        self.beta_min = self.beta_min.to(device=self.device)
        self.beta_max = self.beta_max.to(device=self.device)
        if self.err_ema is not None:
            self.err_ema = self.err_ema.to(device=self.device)

    def loss(self, err, kld):
        # Compute loss with current beta
        loss = err + self.beta * kld
        # Update beta without computing / backpropping gradients
        with torch.no_grad():
            if self.err_ema is None:
                self.err_ema = err
            else:
                self.err_ema = (1.0-self.alpha)*err + self.alpha*self.err_ema
            constraint = (self.goal - self.err_ema)
            if self.speedup is not None and constraint.item() > 0:
                factor = torch.exp(self.speedup * self.step_size * constraint)
            else:
                factor = torch.exp(self.step_size * constraint)
            self.beta = (factor * self.beta).clamp(self.beta_min, self.beta_max)
        # Return loss
        return loss
##################################################################################################
