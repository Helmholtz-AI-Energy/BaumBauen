import torch

from ignite.metrics import Metric
from ignite.exceptions import NotComputableError
from ignite.metrics.metric import sync_all_reduce, reinit__is_reduced

from baumbauen.utils import lca2adjacency, InvalidLCAMatrix


class Efficiency(Metric, object):
    ''' Calculates the efficiency, calculated as (valid trees/total trees)

    - `update` must receive output of the form `(y_pred, y)` or `{'y_pred': y_pred, 'y': y}`.
    - `y_pred` must contain logits and has the following shape (batch_size, num_categories, ...)
    - `y` should have the following shape (batch_size, ...) and contains ground-truth class indices
        with or without the background class. During the computation, argmax of `y_pred` is taken to determine
        predicted classes.

    First the average percentage per batch is computed.
    Then the average of all the batches is returned.

    Args:
        ignored_class (int or list[int]): Class index(es) to ignore when calculating accuracy
        ignore_disconnected_leaves (bool): Whether to ignore disconnected leaves in the LCA when determining if valid
    '''

    def __init__(self, ignore_index=-1., ignore_disconnected_leaves=False, output_transform=lambda x: x, device='cpu'):

        self.ignore_index = ignore_index if isinstance(ignore_index, list) else [ignore_index]
        self.ignore_disconnected = ignore_disconnected_leaves
        self.device = device

        self._num_valid = None
        self._num_examples = None

        super(Efficiency, self).__init__(
            output_transform=output_transform, device=device
        )

    @reinit__is_reduced
    def reset(self):

        self._num_valid = 0
        self._num_examples = 0

        super(Efficiency, self).reset()

    # @reinit__is_reduced
    def _is_valid_lca(self, y_pred, y):
        ''' Remove padding from y_pred and build adjacency

        Args:
            y_pred (Tensor): (d1, d2) square matrix of predicted LCA (no batch dim)
            y (Tensor): (d1, d2) square matrix of true LCA (no batch dim)

        Returns:
            bool: True if adjacency can be built and is valid, else false
        '''
        # Check we weren't passed an empty LCA
        if y.nelement() == 0:
            return False

        # First remove padding rows/cols
        # True represents keep, False is ignore
        ignore_mask = torch.ones(y.size(), dtype=torch.bool, device=self.device)
        # Only interested in those that are not the ignore indices
        for ig_class in self.ignore_index:
            ignore_mask &= (y != ig_class)

        # Now that we have the ignore_index mask,
        # check that predicted LCA isn't trvial (i.e. all zeros)
        if not (y_pred * ignore_mask).any():
            return False

        if self.ignore_disconnected:
            # Create mask to ignore rows of disconnected leaves
            # This is done on predicted LCA since we're only concerned with ignoring
            # what it predicts are disconnected, and calculating if what's left is valid.
            # PerfectLCA will take care of which are actually correct and purity will tell
            # us the correct/valid ratio
            ignore_mask &= (y_pred != 0)

        # Ignore diagonal to be safe
        ignore_mask = ignore_mask.fill_diagonal_(False)

        get_rows = ignore_mask.any(dim=0)  # (d1,) boolean mask of non-empty rows

        bare_y_pred = y_pred[get_rows][:, get_rows]  # (d1, d2) padding rows/cols removed

        # Finally, set diagonal to zero to match expected leaf values in lca2adjacency
        bare_y_pred = bare_y_pred.fill_diagonal_(0)

        # If empty then we've probably predicted all zeros
        if bare_y_pred.numel() == 0:
            return False

        try:
            lca2adjacency(bare_y_pred)
            return True
        except InvalidLCAMatrix:
            return False
        except Exception as e:
            # Something has gone badly wrong
            raise(e)

    @reinit__is_reduced
    def update(self, output):
        ''' Computes the number of valid LCAs PER BATCH!. '''

        y_pred, y = output  # (N, C, d1, d2), (N, d1, d2), where d1 = L =leaves

        # n_leaves = int(y_pred.shape[-1]**0.5)

        # First extract most predicted LCA
        probs = torch.softmax(y_pred, dim=1)  # (N, C, d1, d2)
        winners = probs.argmax(dim=1)  # (N, d1, d2)

        # y = y.flatten(start_dim=1)  # (N, d1)
        assert winners.shape == y.shape, print(f' winners: {winners.shape}, y: {y.shape}')  #

        # mask = (y != 0).float()  # create a mask for the zeroth elements (padded entries and diagonal)

        # y_pred_mask = winners * mask  # zero the respective entries in the predictions

        # Need to loop through LCAs and check they can be built
        n_valid = sum(map(
            self._is_valid_lca,
            torch.unbind(winners),
            torch.unbind(y),
            # torch.unbind(y_pred_mask.view(y_pred_mask.shape[0], n_leaves, n_leaves)),
            # torch.unbind(y.view(y.shape[0], n_leaves, n_leaves)),
        ))

        self._num_valid += n_valid
        self._num_examples += y.shape[0]

    @sync_all_reduce("_efficiency")
    def compute(self):
        ''' Computes the average fraction of valid LCAs across all batches '''

        if self._num_examples == 0:
            raise NotComputableError(
                "CustomAccuracy must have at least one example before it can be computed."
            )
        return self._num_valid / self._num_examples


class Pad_Accuracy(Metric, object):

    """ Computes the average classification accuracy ignoring the given class (e.g. 0 for padding)

    This is almost identical to the CustomAccuracy example in https://pytorch.org/ignite/metrics.html
    except we don't ignore y_pred occurances of the ignored class.

    - `update` must receive output of the form `(y_pred, y)` or `{'y_pred': y_pred, 'y': y}`.
    - `y_pred` must contain logits and has the following shape (batch_size, num_categories, ...)
    - `y` should have the following shape (batch_size, ...) and contains ground-truth class indices
        with or without the background class. During the computation, argmax of `y_pred` is taken to determine
        predicted classes.

    First the average percentage per batch is computed.
    Then the average of all the batches is returned.

    Args:
        ignored_class(int or [int]): Class index(es) to ignore when calculating accuracy

    """
    def __init__(self, ignored_class, output_transform=lambda x: x, device='cpu'):
        self.ignored_class = ignored_class if isinstance(ignored_class, list) else [ignored_class]
        self.device = device
        self._num_correct = None
        self._num_examples = None
        super(Pad_Accuracy, self).__init__(output_transform=output_transform, device=device)

    @reinit__is_reduced
    def reset(self):
        self._num_correct = 0
        self._num_examples = 0
        super(Pad_Accuracy, self).reset()

    @reinit__is_reduced
    def update(self, output):
        y_pred, y = output  # (N, C, d1, d2), (N, d1, d2)

        indices = torch.argmax(y_pred, dim=1)

        mask = torch.ones(y.size(), dtype=torch.bool, device=self.device)  # (N, d1, d2)
        for ig_class in self.ignored_class:
            mask &= (y != ig_class)

        y = y[mask]
        indices = indices[mask]
        correct = torch.eq(indices, y).view(-1)

        self._num_correct += torch.sum(correct).item()
        self._num_examples += correct.shape[0]

    # @sync_all_reduce("_num_examples", "_num_correct")
    @sync_all_reduce("_pad_accuracy")
    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError('Pad_Accuracy must have at least one example before it can be computed.')
        return self._num_correct / self._num_examples


class PerfectLCA(Metric, object):

    """ Computes the percentage of the Perfectly predicted LCAs

    - `update` must receive output of the form `(y_pred, y)` or `{'y_pred': y_pred, 'y': y}`.
    - `y_pred` must contain logits and has the following shape (batch_size, num_categories, ...)
    - `y` should have the following shape (batch_size, ...) and contains ground-truth class indices
        with or without the background class. During the computation, argmax of `y_pred` is taken to determine
        predicted classes.

    First the average percentage per batch is computed.
    Then the average of all the batches is returned.
    """

    def __init__(self, ignore_index=-1., output_transform=lambda x: x, device='cpu'):

        self.ignore_index = ignore_index if isinstance(ignore_index, list) else [ignore_index]
        self.device = device
        self._per_corrects = None
        self._num_examples = None

        super(PerfectLCA, self).__init__(
            output_transform=output_transform, device=device
        )

    @reinit__is_reduced
    def reset(self):

        self._per_corrects = 0
        self._num_examples = 0

        super(PerfectLCA, self).reset()

    ''' Computes the percentage of Perfect LCAs PER BATCH!.
    the tensors y_pred and y contain multiple LCAs that belong in a batch.
    '''

    @reinit__is_reduced
    def update(self, output):

        y_pred, y = output  # (N, C, d1, d2), (N, d1, d2)

        probs = torch.softmax(y_pred, dim=1)  # (N, C, d1, d2)
        winners = probs.argmax(dim=1)  # (N, d1, d2)

        # print(y.shape)
        assert winners.shape == y.shape

        # Create a mask for the ignored elements (padded entries and diagonal)
        mask = torch.ones(y.size(), dtype=torch.bool, device=self.device)
        for ig_class in self.ignore_index:
            mask &= (y != ig_class)
        # mask = (y != self.ignore_index).float()

        # Zero the respective entries in the predictions
        y_pred_mask = winners * mask
        y_mask = y * mask

        # Do this to make comparison across d1 and d2
        y_mask = y_mask.flatten(start_dim=1)  # (N, d1)
        y_pred_mask = y_pred_mask.flatten(start_dim=1)  # (N, d1)

        # (N) compare the masked predictions with the target. The padded and the diagonal will be equal due to masking
        truth = y_pred_mask.eq(y_mask).all(dim=1)

        # Count the number of zero wrong predictions across the batch.
        batch_perfect = truth.sum().item()

        self._per_corrects += batch_perfect
        self._num_examples += y.shape[0]

    @sync_all_reduce("_perfect")
    def compute(self):

        if self._num_examples == 0:
            raise NotComputableError(
                "CustomAccuracy must have at least one example before it can be computed."
            )
        return self._per_corrects / self._num_examples
