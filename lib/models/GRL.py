import torch


class _GradientScalarLayer(torch.autograd.Function):
    """
    Torch.autograd.Function for reversing the gradient.
    """

    @staticmethod
    def forward(ctx, input, weight):
        """
        Forward pass is an identity function.
        Args:
            ctx: context object
            input: input tensor
            weight: gradient scaling factor
        """
        ctx.weight = weight
        return input.view_as(input)

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass reverses the gradient and scales it by the weight.
        Args:
            ctx: context object
            grad_output: gradient from the subsequent layer
        """
        grad_input = grad_output.clone()
        return ctx.weight * grad_input, None


gradient_scalar = _GradientScalarLayer.apply


class GradientScalarLayer(torch.nn.Module):
    """
    A layer that reverses the gradient during backward pass.
    This is used for the adversarial objective in Domain-Adversarial
    Training of Neural Networks (DANN).
    """

    def __init__(self, weight):
        """
        Args:
            weight (float): The gradient scaling factor.
        """
        super(GradientScalarLayer, self).__init__()
        self.weight = weight

    def forward(self, input):
        """
        Applies the gradient reversal.
        """
        return gradient_scalar(input, self.weight)

    def __repr__(self):
        """
        String representation of the layer.
        """
        tmpstr = self.__class__.__name__ + "("
        tmpstr += "weight=" + str(self.weight)
        tmpstr += ")"
        return tmpstr
