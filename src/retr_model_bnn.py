from src.retr_model import FusionRetrModelBase
from src.bnn.bayesian_linear import BayesianLinear
import torch.nn

class RetrModelBNN(FusionRetrModelBase):
    def create_linear_layer(self, in_features, out_features, prior):
        #import pdb; pdb.set_trace()
        if prior is None:
            weight_mu = 0
            weight_sigma = 0.1
            bias_mu = 0
            bias_sigma = 0.1
        else:
           weight_mu = prior['weight_mu']
           weight_sigma = prior['weight_sigma']
           bias_mu = prior['bias_mu']
           bias_sigma = prior['bias_sigma']
            
        return BayesianLinear(in_features, out_features, weight_mu, weight_sigma, bias_mu, bias_sigma)
    
    def log_prior(self):
        log_prior_sum = self.passage_fnt.log_prior \
                      + self.table_fnt.log_prior \
                      + self.feat_l1.log_prior \
                      + self.feat_l2.log_prior
        return log_prior_sum

    def log_variational_posterior(self):
        log_posterior = self.passage_fnt.log_variational_posterior \
                      + self.table_fnt.log_variational_posterior \
                      + self.feat_l1.log_variational_posterior \
                      + self.feat_l2.log_variational_posterior
        return log_posterior

    def sample_forward(self, batch_data, fusion_scores, fusion_states, passage_masks, 
                      sample=False, calculate_log_probs=False, opts=None, num_samples=1):
        outputs = []
        log_priors = torch.zeros(num_samples)
        log_variational_posteriors = torch.zeros(num_samples)
        for i in range(num_samples):
            output_item = self(batch_data, fusion_scores, fusion_states, passage_masks, 
                                 sample=True, opts=opts)
            output_item = torch.stack(output_item).unsqueeze(0)
            outputs.append(output_item)
            log_priors[i] = self.log_prior()
            log_variational_posteriors[i] = self.log_variational_posterior()
        
        outputs = torch.cat(outputs, dim=0).mean(dim=0)
        log_prior = log_priors.mean()
        log_variational_posterior = log_variational_posteriors.mean()
        
        opts['log_prior'] = log_prior
        opts['log_variational_posterior'] = log_variational_posterior
        return outputs 
        
