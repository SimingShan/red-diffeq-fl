import torch

class RED_DiffEq():
    """Diffusion-based regularization via RED-DiffEq."""
    def __init__(self, diffusion_model):
        self.diffusion_model = diffusion_model

    def generate_time_tensor(self, mu):
        """Generate a random diffusion time tensor per batch element."""
        time = torch.randint(0, self.diffusion_model.num_timesteps, (1,)).item()
        time_cond = torch.full((mu.shape[0],), time, device=mu.device, dtype=torch.long)
        return time_cond
    
    def generate_noisy_sample(self, mu, time_tensor):
        """Create a noisy sample `x_t` and return it with the noise and x0_pred."""
        x0_pred = mu 
        noise = torch.randn_like(mu)    
        x_t = self.diffusion_model.q_sample(x0_pred, t=time_tensor, noise=noise)
        return x_t, noise, x0_pred

    def get_reg_loss(self, mu):
        """Compute the diffusion regularization signal for `mu`."""
        time_tensor = self.generate_time_tensor(mu)
        x_t, noise, x0_pred = self.generate_noisy_sample(mu, time_tensor)
        # Generate noisy predictions
        with torch.no_grad():
            model_predictions = self.diffusion_model.model_predictions(
                x_t, t=time_tensor, x_self_cond=None, clip_x_start=True, rederive_pred_noise=True
            )
        pred_noise, _ = model_predictions.pred_noise, model_predictions.pred_x_start
        et = pred_noise.detach()
        reg_loss = torch.mul((et - noise).detach(), x0_pred).mean()
        return reg_loss
