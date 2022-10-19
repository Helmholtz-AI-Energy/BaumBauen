import torch


def encode_onehot(labels, device):
    classes = set(labels)
    classes_dict = {c: torch.eye(len(classes), device=device)[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = torch.tensor(
        list(map(classes_dict.get, labels)),
        dtype=torch.int32,
        device=device
    )
    return labels_onehot
