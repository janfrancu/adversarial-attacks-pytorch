import torch
import torch.nn as nn

from ..attack import Attack


def project_simplex(u, bound):
    s = torch.sort(u, descending=True)[0]
    c = (torch.cumsum(s, dim=1) - bound) / torch.arange(1, u.size(1)+1, device=u.device)
    l = torch.max(c, dim=1).values
    l = l.view(-1, 1)
    return torch.clamp(u - l, min=0)


def project_l1ball(u, bound):
    assert bound >= 0
    
    abs_u = torch.abs(u)
    l1norm = abs_u.sum(dim=1)
    mask = l1norm > bound

    if not mask.any():
        return u

    out = u.clone()
    u_outside = abs_u[mask]
    v = project_simplex(u_outside, bound)
    out[mask] = torch.sign(u[mask]) * v

    return out


class PGDL1(Attack):
    r"""
    PGD-L1 Attack.

    This implements PGD under an L1 constraint. The update uses a sparse
    gradient direction by keeping only the largest components and
    L1-normalizing the sign vector before each step.

    Distance Measure : L1

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 1.0)
        alpha (float): step size. (Default: 0.2)
        steps (int): number of steps. (Default: 10)
        random_start (bool): using random initialization of delta. (Default: True)
        loss_function (str): loss function for adversarial generation. (Default: 'crossentropy')
            - 'crossentropy': For multi-class classification (standard)
            - 'binary_crossentropy': For binary classification models with single output

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> # Multi-class classification (default)
        >>> attack = torchattacks.PGDL1(model, eps=12, alpha=1, steps=10, random_start=True)
        >>> adv_images = attack(images, labels)
        
        >>> # Binary classification with single output
        >>> binary_model = MyBinaryModel()  # outputs shape [batch_size, 1]
        >>> attack = torchattacks.PGDL1(binary_model, eps=12, alpha=1, steps=10,
        ...                              random_start=True, loss_function='binary_crossentropy')
        >>> adv_images = attack(images, binary_labels)  # binary_labels are 0/1
    """

    def __init__(
        self,
        model,
        eps=1.0,
        alpha=0.2,
        steps=10,
        random_start=True,
        eps_for_division=1e-10,
        loss_function="crossentropy",
    ):
        super().__init__("PGDL1", model)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
        self.eps_for_division = eps_for_division
        self.supported_mode = ["default", "targeted"]

        if loss_function != "crossentropy":
            self.set_loss_function(loss_function)

    def forward(self, images, labels):
        r"""
        Overridden.
        """

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self.targeted:
            target_labels = self.get_target_label(images, labels)

        adv_images = images.clone().detach()
        batch_size = len(images)

        if self.random_start:
            # Sample Gaussian noise
            delta = torch.randn_like(images)
            # Flatten for per-image norm computation
            delta_flat = delta.view(batch_size, -1)
            # Compute L2 norm per sample
            delta_norm = torch.norm(delta_flat, p=2, dim=1, keepdim=True)
            # Sample random radius u in [0,1]
            u = torch.rand(batch_size, 1, device=images.device)
            # Scale to get delta = u * eps * delta' / ||delta'||_2
            delta = delta_flat * (u * self.eps / (delta_norm + self.eps_for_division))
            # Now project to L1 ball
            delta = project_l1ball(delta, self.eps)
            # Reshape back to image
            delta = delta.view_as(images)
            # Create adversarial starting point
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()


        for _ in range(self.steps):
            adv_images.requires_grad = True
            outputs = self.get_logits(adv_images)

            # Calculate loss using the configured loss function
            if self.targeted:
                cost = -self.get_loss(outputs, target_labels)
            else:
                cost = self.get_loss(outputs, labels)

            # Get gradient
            grad = torch.autograd.grad(
                cost, adv_images, retain_graph=False, create_graph=False
            )[0]

            # Sparsification (top 1% values)
            g = grad.view(batch_size, -1)
            D = g.size(1)

            k = max(1, int(0.01 * D))
            idx = D - k

            thresh = torch.kthvalue(g.abs(), idx, dim=1).values
            thresh = thresh.view(batch_size, 1)

            mask = (g.abs() >= thresh)
            g_sparse = g * mask
            g_sparse = g_sparse.view_as(grad)

            # L1-normalized sparse gradient direction
            direction = g_sparse.sign()
            l1_norm = direction.abs().view(batch_size, -1).sum(dim=1).view(batch_size,1,1,1)
            direction = direction / (l1_norm + self.eps_for_division)

            # Update
            adv_images = adv_images + self.alpha * direction

            # Project to L1 ball
            delta = adv_images - images
            delta = delta.view(batch_size, -1)
            delta = project_l1ball(delta, self.eps)
            delta = delta.view_as(images)

            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images
    
