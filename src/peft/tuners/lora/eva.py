import time
import torch
from tqdm import tqdm
from functools import reduce, partial
from collections import Counter, defaultdict

from typing import Optional, Union


def find_equal_values(dictionary):
    value_dict = defaultdict(list)
    for k,v in dictionary.items():
        value_dict[v].append(k)
    return {k: v for k, v in value_dict.items() if len(v) > 1}


def cycle(iterable):
    while True:
        for x in iterable:
            yield x


def match_module_name(module_name, name_to_match):
    return ".".join(module_name.split(".")[-name_to_match.count(".")-1:]) == name_to_match


class IncrementalPCA:
    """
    An implementation of Incremental Principal Components Analysis (IPCA) that leverages PyTorch for GPU acceleration.

    This class provides methods to fit the model on data incrementally in batches, and to transform new data
    based on the principal components learned during the fitting process.

    Attributes:
        n_components (int, optional): Number of components to keep. If `None`, it's set to the minimum of the
                                number of samples and features. Defaults to None.
        copy (bool): If False, input data will be overwritten. Defaults to True.
        batch_size (int, optional): The number of samples to use for each batch. Only needed if self.fit is called.
                                If `None`, it's inferred from the data and set to `5 * n_features`. Defaults to None.
        svd_driver (str, optional): name of the cuSOLVER method to be used for torch.linalg.svd. This keyword 
                                argument only works on CUDA inputs. Available options are: None, gesvd, gesvdj,
                                and gesvda. Defaults to None.
        lowrank (bool, optional): Whether to use torch.svd_lowrank instead of torch.linalg.svd which can be faster.
                                Defaults to False.
        lowrank_q (int, optional): For an adequate approximation of n_components, this parameter defaults to 
                                n_components * 2.
        lowrank_niter (int, optional): Number of subspace iterations to conduct for torch.svd_lowrank.
                                Defaults to 4. 
    """

    def __init__(
        self,
        n_components: Optional[int] = None,
        copy: Optional[bool] = True,
        batch_size: Optional[int] = None,
        svd_driver: Optional[str] = None,
        lowrank: bool = False,
        lowrank_q: Optional[int] = None,
        lowrank_niter: int = 4
    ):
        self.n_components_ = n_components
        self.copy = copy
        self.batch_size = batch_size
        self.lowrank = lowrank

        if lowrank:
            if lowrank_q is None:
                lowrank_q = n_components * 2
            else:
                assert lowrank_q >= n_components, "lowrank_q must be greater than or equal to n_components."
            
            def svd_fn(X):
                U, S, V = torch.svd_lowrank(X, q=lowrank_q, niter=lowrank_niter)
                return U, S, V.mH # V is returned as a conjugate transpose
            self.svd_fn = svd_fn

        else:
            self.svd_fn = partial(torch.linalg.svd, full_matrices=False, driver=svd_driver)
        

    def _validate_data(self, X, dtype=torch.float32):
        """
        Validates and converts the input data `X` to the appropriate tensor format.

        This method ensures that the input data is in the form of a PyTorch tensor and resides on the correct device (CPU or GPU).
        It also provides an option to create a copy of the tensor, which is useful when the input data should not be overwritten.

        Args:
            X (torch.Tensor): Input data.
            dtype (torch.dtype, optional): Desired data type for the tensor. Defaults to torch.float32.
            copy (bool, optional): Whether to clone the tensor. If True, a new tensor is returned; otherwise, the original tensor
                                   (or its device-transferred version) is returned. Defaults to True.

        Returns:
            torch.Tensor: Validated and possibly copied tensor residing on the specified device.
        """
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=dtype)
        elif self.copy:
            X = X.clone()

        if X.dtype != dtype:
            X = X.to(dtype)

        return X

    @staticmethod
    def _incremental_mean_and_var(X, last_mean, last_variance, last_sample_count):
        """
        Computes the incremental mean and variance for the data `X`.

        Args:
            X (torch.Tensor): The batch input data tensor with shape (n_samples, n_features).
            last_mean (torch.Tensor): The previous mean tensor with shape (n_features,).
            last_variance (torch.Tensor): The previous variance tensor with shape (n_features,).
            last_sample_count (torch.Tensor): The count tensor of samples processed before the current batch.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, int]: Updated mean, variance tensors, and total sample count.
        """
        if X.shape[0] == 0:
            return last_mean, last_variance, last_sample_count

        if last_sample_count > 0:
            assert (
                last_mean is not None
            ), "last_mean should not be None if last_sample_count > 0."
            assert (
                last_variance is not None
            ), "last_variance should not be None if last_sample_count > 0."

        new_sample_count = torch.tensor([X.shape[0]], device=X.device)
        updated_sample_count = last_sample_count + new_sample_count

        if last_mean is None:
            last_sum = torch.zeros(X.shape[1], dtype=torch.float64, device=X.device)
        else:
            last_sum = last_mean * last_sample_count

        new_sum = X.sum(dim=0, dtype=torch.float64)

        updated_mean = (last_sum + new_sum) / updated_sample_count

        T = new_sum / new_sample_count
        temp = X - T
        correction = temp.sum(dim=0, dtype=torch.float64).square()
        temp.square_()
        new_unnormalized_variance = temp.sum(dim=0, dtype=torch.float64)
        new_unnormalized_variance -= correction / new_sample_count
        if last_variance is None:
            updated_variance = new_unnormalized_variance / updated_sample_count
        else:
            last_unnormalized_variance = last_variance * last_sample_count
            last_over_new_count = last_sample_count.double() / new_sample_count
            updated_unnormalized_variance = (
                last_unnormalized_variance
                + new_unnormalized_variance
                + last_over_new_count
                / updated_sample_count
                * (last_sum / last_over_new_count - new_sum).square()
            )
            updated_variance = updated_unnormalized_variance / updated_sample_count

        return updated_mean, updated_variance, updated_sample_count

    @staticmethod
    def _svd_flip(u, v, u_based_decision=True):
        """
        Adjusts the signs of the singular vectors from the SVD decomposition for deterministic output.

        This method ensures that the output remains consistent across different runs.

        Args:
            u (torch.Tensor): Left singular vectors tensor.
            v (torch.Tensor): Right singular vectors tensor.
            u_based_decision (bool, optional): If True, uses the left singular vectors to determine the sign flipping. Defaults to True.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Adjusted left and right singular vectors tensors.
        """
        if u_based_decision:
            max_abs_cols = torch.argmax(torch.abs(u), dim=0)
            signs = torch.sign(u[max_abs_cols, range(u.shape[1])])
        else:
            max_abs_rows = torch.argmax(torch.abs(v), dim=1)
            signs = torch.sign(v[range(v.shape[0]), max_abs_rows])
        u *= signs[:u.shape[1]].view(1, -1)
        v *= signs.view(-1, 1)
        return u, v

    def fit(self, X, check_input=True):
        """
        Fits the model with data `X` using minibatches of size `batch_size`.

        Args:
            X (torch.Tensor): The input data tensor with shape (n_samples, n_features).

        Returns:
            IncrementalPCAGPU: The fitted IPCA model.
        """
        if check_input:
            X = self._validate_data(X)
        n_samples, n_features = X.shape
        if self.batch_size is None:
            self.batch_size_ = 5 * n_features
        else:
            self.batch_size_ = self.batch_size

        for start in range(0, n_samples, self.batch_size_):
            end = min(start + self.batch_size_, n_samples)
            X_batch = X[start:end]
            self.partial_fit(X_batch, check_input=False)

        return self

    def partial_fit(self, X, check_input=True):
        """
        Incrementally fits the model with batch data `X`.

        Args:
            X (torch.Tensor): The batch input data tensor with shape (n_samples, n_features).
            check_input (bool, optional): If True, validates the input. Defaults to True.

        Returns:
            IncrementalPCAGPU: The updated IPCA model after processing the batch.
        """
        first_pass = not hasattr(self, "components_")

        if check_input:
            X = self._validate_data(X)
        n_samples, n_features = X.shape

        # Initialize attributes to avoid errors during the first call to partial_fit
        if first_pass:
            self.mean_ = None  # Will be initialized properly in _incremental_mean_and_var based on data dimensions
            self.var_ = None  # Will be initialized properly in _incremental_mean_and_var based on data dimensions
            self.n_samples_seen_ = torch.tensor([0], device=X.device)
            if not self.n_components_:
                self.n_components_ = min(n_samples, n_features)

        col_mean, col_var, n_total_samples = self._incremental_mean_and_var(
            X, self.mean_, self.var_, self.n_samples_seen_
        )

        if first_pass:
            X -= col_mean
        else:
            col_batch_mean = torch.mean(X, dim=0)
            X -= col_batch_mean
            mean_correction_factor = torch.sqrt(
                (self.n_samples_seen_.double() / n_total_samples) * n_samples
            )
            mean_correction = mean_correction_factor * (self.mean_ - col_batch_mean)
            X = torch.vstack(
                (
                    self.singular_values_.view((-1, 1)) * self.components_,
                    X,
                    mean_correction,
                )
            )

        U, S, Vt = self.svd_fn(X)
        U, Vt = self._svd_flip(U, Vt, u_based_decision=False)
        explained_variance = S**2 / (n_total_samples - 1)
        explained_variance_ratio = S**2 / torch.sum(col_var * n_total_samples)

        self.n_samples_seen_ = n_total_samples
        self.components_ = Vt[:self.n_components_]
        self.singular_values_ = S[:self.n_components_]
        self.mean_ = col_mean
        self.var_ = col_var
        self.explained_variance_ = explained_variance[:self.n_components_]
        self.explained_variance_ratio_ = explained_variance_ratio[:self.n_components_]
        if self.n_components_ not in (n_samples, n_features):
            self.noise_variance_ = explained_variance[self.n_components_:].mean()
        else:
            self.noise_variance_ = torch.tensor(0.0, device=X.device)
        return self

    def transform(self, X):
        """
        Applies dimensionality reduction to `X`.

        The input data `X` is projected on the first principal components previously extracted from a training set.

        Args:
            X (torch.Tensor): New data tensor with shape (n_samples, n_features) to be transformed.

        Returns:
            torch.Tensor: Transformed data tensor with shape (n_samples, n_components).
        """
        X -= self.mean_
        return torch.mm(X, self.components_.T)
    

class PCAHook:
    def __init__(
        self,
        name: str,
        n_components: int,
        sim_thresh: Union[float, torch.Tensor]
    ):
        self.name = name
        self.n_components = n_components
        self.sim_thresh = sim_thresh

        if isinstance(sim_thresh, torch.Tensor) and len(sim_thresh.shape) > 0:
            check1 = sim_thresh.size(0) == n_components or sim_thresh.size(0) == 1
            check2 = len(sim_thresh.shape) == 1
            assert check1 and check2, "if sim_thresh is a tensor with more than 0 dimensions it must have shape (n_components,) or (1,)"

        self.pca = IncrementalPCA(n_components=n_components, copy=True, lowrank=True)

        self.indices = None
        self.converged = torch.zeros((n_components,), dtype=torch.bool)

    def __call__(self, model, input, output):
        previous_components = None
        if hasattr(self.pca, "components_"):
            previous_components = self.pca.components_.clone().detach()

        try:
            states = input.detach()
        except AttributeError:
            states = input[0].detach()
        states = states[self.indices[:, 0], self.indices[:, 1], :]

        if states.size(0) < self.n_components:
            return

        self.pca.partial_fit(states.to(torch.float32))

        if previous_components is not None:
            components = self.pca.components_
            if len(components.shape) == 1:
                components = components.reshape(1, -1)
                previous_components = previous_components.reshape(1, -1)
            # consider as converged if enough components have converged via cossim
            sim = torch.nn.functional.cosine_similarity(components, previous_components)
            self.converged = (sim >= self.sim_thresh)


class HashHook:

    def __init__(self, name: str):
        self.name = name
        self.hashed_inputs = []

    @staticmethod
    def hash_fn(tensor):
        return hash(tuple(tensor.view(-1).tolist()))

    def __call__(self, model, input, output):
        try:
            x = input.detach().cpu()
        except AttributeError:
            x = input[0].detach().cpu()
        self.hashed_inputs.append(self.hash_fn(x))


@torch.no_grad()
def compute_pca(
    model,
    r,
    dataloader,
    rho=2,
    tau=0.99,
    use_label_mask=True,
    label_mask_value=-100,
    whiten=False,
    target_modules=None,
    device="cuda"
):
        
    def _get_rank_distribution(hooks, hook_layer_map, equal_inputs_map, rank_budget, max_components):
        exp_vars = {k: h.pca.explained_variance_ratio_[:max_components] for k, h in hooks.items()}
        keys, values = zip(*[(k, c) for k, name in hook_layer_map.items() for c in exp_vars[name]])
        idx = torch.stack(values).argsort(descending=True)
        counts = Counter([keys[i] for i in idx[:rank_budget]])
        counts = {k: counts.get(k, 0) for k in hook_layer_map.keys()} # add layers with 0 rank
        for k, k_hook in equal_inputs_map.items():
            # ensure hook layers have the highest rank if they are equal to another layer
            rank, rank_hook = counts[k], counts[k_hook]
            if rank_hook >= rank:
                continue
            counts[k_hook], counts[k] = rank, rank_hook
        return counts
    
    assert rho >= 1.0, "early_stop_rho must be >= 1"
    
    # TODO multi gpu support, only run rank 0
    orig_device = next(model.parameters()).device
    training = model.training
    model = model.to(device)
    model.eval()

    hooks = {}
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            if target_modules:
                check = [match_module_name(name, t) for t in target_modules] # TODO probably there is a helper function in peft already to do this matching
                if not any(check):
                    continue
            hook = HashHook(name)
            module.register_forward_hook(hook)
            hooks[name] = hook
    rank_budget = r * len(hooks)
    max_components = round(r * rho)

    # forward for one batch to check which layer inputs are equal to avoid unneeded pca calculations
    inputs = {k: v.to(device) for k, v in next(iter(dataloader)).items() if k != "labels"}
    model(**inputs)
    hash_dict = {k: h.hashed_inputs[0] for k, h in hooks.items()}
    equal_inputs_map = {vv: v[0] for v in find_equal_values(hash_dict).values() for vv in v[1:]}
    hooks = {k: PCAHook(k, max_components, tau) for k in hooks.keys() if k not in equal_inputs_map}
    layer_hook_map = {**dict(zip(hooks.keys(), hooks.keys())), **equal_inputs_map}
    for name in layer_hook_map.keys():
        module = reduce(getattr, name.split("."), model)
        module._forward_hooks.clear()
    
    # start pca calculation
    pbar = tqdm(iter(cycle(dataloader)), position=0, leave=False)
    convergence_dict = {k: False for k in hooks.keys()}
    rank_dist = {k: max_components for k in layer_hook_map.keys()}
    for inputs in pbar:

        mask = torch.ones_like(inputs["input_ids"], dtype=torch.bool)
        if hasattr(inputs, "attention_mask"):
            mask = inputs["attention_mask"].bool()
        if use_label_mask and hasattr(inputs, "labels"):
            mask = torch.logical_and(mask, inputs["labels"] != label_mask_value)
        indices = torch.nonzero(mask)
        inputs = {k: v.to(device) for k, v in inputs.items() if k != "labels"}

        for name, hook in hooks.items():
            module = reduce(getattr, name.split("."), model)
            module._forward_hooks.clear()
            # check if all components that are needed for the rank distribution have converged
            if torch.all(hook.converged[:rank_dist[name]]):
                convergence_dict[name] = True
                continue
            convergence_dict[name] = False
            hook.indices = indices
            module.register_forward_hook(hook)

        if all(convergence_dict.values()):
            print("exiting - all PCA components have converged.")
            break

        model(**inputs)

        # in case some hooks have to skip the pca calculation because the number of tokens is less than the number of components
        if not all([hasattr(h.pca, "components_") for h in hooks.values()]):
            continue

        rank_dist = _get_rank_distribution(hooks, layer_hook_map, equal_inputs_map, rank_budget, max_components)

        layer_converged = list(convergence_dict.values()) + [convergence_dict[v] for v in equal_inputs_map.values()]
        pbar.set_description(f"{sum(layer_converged)}/{len(layer_converged)} layers have converged")

    pca_dict = {}
    for name, rank in rank_dist.items():
        if rank == 0:
            continue
        hook = hooks[layer_hook_map[name]]
        assert torch.all(hook.converged[:rank]) # this should never happen because we check for convergence
        u = hook.pca.components_[:rank]
        if whiten:
            u /= hook.pca.singular_values_[:rank].sqrt().reshape(-1, 1)
        pca_dict[name] = u

    # objects are torch tensors on the model device
    pca_dict = {k: v.to(orig_device) for k, v in pca_dict.items()}

    # restore model state
    model.train(training)
    model.to(orig_device) # TODO

    return pca_dict