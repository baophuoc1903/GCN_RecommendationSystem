import torch


def bpr_loss(output, batch_size, loss_mode):
    pred, l2_loss = output
    loss = -torch.log(torch.sigmoid(pred[:, 0] - pred[:, 1]) + 1e-8)

    # with torch.no_grad():
    #     loss[:] = loss.clamp(0.0, None)

    if loss_mode == 'mean':
        loss = torch.mean(loss)
    elif loss_mode == 'sum':
        loss = torch.sum(loss)
    else:
        raise ValueError("loss_mode must be 'mean' or 'sum'!")

    loss += l2_loss / batch_size

    return loss
