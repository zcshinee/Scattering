import torch
import torch.nn.functional as F

def PCC_single(output, label):
  output_mean = torch.mean(output)
  label_mean = torch.mean(label)
  numer = (output-output_mean)*(label-label_mean)
  deno1 = (output-output_mean)**2
  deno2 = (label-label_mean)**2
  return torch.sum(numer)/(torch.sum(deno1)**0.5)/(torch.sum(deno2)**0.5)

def PCC_mean(outputs, labels):
  pcc_sum = 0
  for i in range(outputs.shape[0]):
    output = outputs[i]
    label = labels[i]
    pcc_sum += PCC_single(output, label)
  return pcc_sum/outputs.shape[0]

def calc_loss(outputs, labels, metrics):
  npcc_weight = 0 #if npcc_weight = 1, use npcc loss; if npcc_weight = 0, use bce loss.
  pcc = PCC_mean(outputs, labels) #losses are averaged over batch
  bce = F.binary_cross_entropy(outputs, labels)
  loss = -pcc*npcc_weight+bce*(1-npcc_weight)

  metrics['bce'] += bce.data.cpu().numpy()*outputs.size(0) #final metrics are averaged over all samples
  metrics['pcc'] += pcc.data.cpu().numpy()*outputs.size(0)
  metrics['loss'] += loss.data.cpu().numpy()*outputs.size(0)
  return loss

def print_metrics(metrics, epoch_samples, phase):    
    outputs = []
    for k in metrics.keys():
        outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))
        
    print("{}: {}".format(phase, ", ".join(outputs)))