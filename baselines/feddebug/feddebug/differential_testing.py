"""Fed_Debug Differential Testing."""

import itertools
import time


import torch
import torch.nn.functional as F


from feddebug.neuron_activation import get_neurons_activations
from feddebug.utils import seed_everything, create_transform
from logging import DEBUG
from flwr.common.logger import log

seed_everything(786)

class InferenceGuidedInputGenerator:
    """Generate random inputs based on the feedback from the clients."""

    def __init__(self, clients2models, input_shape, transform_func, k_gen_inputs=10,
                 min_nclients_same_pred=3, time_delta=60, faster_input_generation=False):
        self.clients2models = clients2models
        self.input_shape = input_shape
        self.transform = transform_func
        self.k_gen_inputs = k_gen_inputs
        self.min_nclients_same_pred = min_nclients_same_pred
        self.time_delta = time_delta
        self.faster_input_generation = faster_input_generation
        self.seed = 0

    def _get_random_input(self):
        torch.manual_seed(self.seed)
        self.seed += 1
        img = torch.rand(self.input_shape)
        if self.transform:
            return self.transform(img).unsqueeze(0)
        return img.unsqueeze(0)

    def _simple_random_inputs(self):
        start_time = time.time()
        random_inputs = [self._get_random_input() for _ in range(self.k_gen_inputs)]
        elapsed_time = time.time() - start_time
        return random_inputs, elapsed_time

    def _generate_feedback_random_inputs(self):
        print("Generating feedback-based random inputs")
        random_inputs = []
        same_prediction_set = set()
        start_time = time.time()
        timeout = 60

        while len(random_inputs) < self.k_gen_inputs:
            img = self._get_random_input()
            if self.min_nclients_same_pred > 1:
                self._append_or_not(img, random_inputs, same_prediction_set)
            else:
                random_inputs.append(img)

            if time.time() - start_time > timeout:
                timeout += 60
                self.min_nclients_same_pred -= 1
                print(f">> Timeout: Number of distinct inputs: {len(random_inputs)}, "
                      f"decreasing min_nclients_same_pred to {self.min_nclients_same_pred} "
                      f"and extending timeout to {timeout} seconds")

        elapsed_time = time.time() - start_time
        return random_inputs, elapsed_time

    def _append_or_not(self, input_tensor, random_inputs, same_prediction_set):
        preds = [self._predict_func(model, input_tensor)
                 for model in self.clients2models.values()]
        for ci1, pred1 in enumerate(preds):
            seq = {ci1}
            for ci2, pred2 in enumerate(preds):
                if ci1 != ci2 and pred1 == pred2:
                    seq.add(ci2)

            seq_str = ",".join(map(str, seq))
            if seq_str not in same_prediction_set and len(seq) >= self.min_nclients_same_pred:
                same_prediction_set.add(seq_str)
                random_inputs.append(input_tensor)

    def _predict_func(self, model, input_tensor):
        model.eval()
        logits = model(input_tensor)
        preds = torch.argmax(F.log_softmax(logits, dim=1), dim=1)
        pred = preds.item()
        return pred

    def get_inputs(self):
        """Return generated random inputs."""
        if self.faster_input_generation or len(self.clients2models) <= 10:
            return self._simple_random_inputs()
        return self._generate_feedback_random_inputs()


class FaultyClientDetector:
    """Faulty Client Localization using Neuron Activation."""

    def __init__(self, device):
        self.leave_1_out_combs = None
        self.device = device

    def _generate_leave_one_out_combinations(self, clients_ids):
        """Generate and update all subsets of clients with a specified subset size."""
        subset_size = len(clients_ids) - 1
        subsets = [set(sub) for sub in itertools.combinations(clients_ids, subset_size)]
        return subsets

    def _torch_intersection(self, client2tensors):
        intersect = torch.ones_like(
            next(iter(client2tensors.values())), dtype=torch.bool)
        for temp_t in client2tensors.values():
            intersect = torch.logical_and(intersect, temp_t)
        return intersect

    def _get_clients_ids_with_highest_common_neurons(self, clients2neurons2boolact):
        def _count_common_neurons(comb):
            """Return the number of common neurons.

            In PyTorch, boolean values are treated as integers (True as 1 and False as
            0), so summing a tensor of boolean values will give you the count of True
            values.
            """
            c2act = {cid: clients2neurons2boolact[cid] for cid in comb}
            intersect_tensor = self._torch_intersection(c2act)
            return intersect_tensor.sum().item()

        count_of_common_neurons = [
            _count_common_neurons(comb) for comb in self.leave_1_out_combs
        ]

        highest_number_of_common_neurons = max(count_of_common_neurons)
        val_index = count_of_common_neurons.index(highest_number_of_common_neurons)
        val_clients_ids = self.leave_1_out_combs[val_index]
        return val_clients_ids

    def get_client_neurons_activations(self, client2model, input_tensor):
        client2acts = {}
        for cid, model in client2model.items():
            model = model.to(self.device)
            neurons_act = get_neurons_activations(model, input_tensor.to(self.device))
            client2acts[cid] = neurons_act.cpu()
            model = model.cpu()
            input_tensor = input_tensor.cpu()
        return client2acts

    def get_malicious_clients(self, client2acts, na_t, num_bugs):
        potential_faulty_clients = None
        all_clients_ids = set(client2acts.keys())
        self.leave_1_out_combs = self._generate_leave_one_out_combinations(
            all_clients_ids)
        for _ in range(num_bugs):
            client2_na = {cid: activations > na_t for cid,
                          activations in client2acts.items()}
            normal_clients_ids = self._get_clients_ids_with_highest_common_neurons(
                client2_na)

            potential_faulty_clients = all_clients_ids - normal_clients_ids
            log(DEBUG, f"Malicious clients {potential_faulty_clients}")
            self.leave_1_out_combs = self._generate_leave_one_out_combinations(
                all_clients_ids - potential_faulty_clients)

        return potential_faulty_clients


def differential_testing_fl_clients(client2model, num_bugs, num_inputs, input_shape, na_threshold, faster_input_generation, device):
    """Differential Testing for FL Clients."""
    generate_inputs = InferenceGuidedInputGenerator(
        clients2models=client2model,
        input_shape=input_shape,
        transform_func=create_transform(),
        k_gen_inputs=num_inputs,
        min_nclients_same_pred=3,
        faster_input_generation=faster_input_generation
    )

    # Generate selected inputs for analysis
    selected_inputs, _ = generate_inputs.get_inputs()

    predicted_faulty_clients = []
    localize = FaultyClientDetector(device)

    # Iterate over each input tensor to detect malicious clients
    for input_tensor in selected_inputs:
        # Get neuron activations for each client model
        client2acts = localize.get_client_neurons_activations(
            client2model, input_tensor)

        # Identify potential malicious clients based on activations and thresholds
        potential_malicious_clients = localize.get_malicious_clients(
            client2acts, na_threshold, num_bugs
        )

        predicted_faulty_clients.append(potential_malicious_clients)
    return predicted_faulty_clients


# def eval_na_threshold(cfg):
#     """Evaluate the impact of Neuron Activation threshold on the debugging."""
#     debug_results_cache = Index(cfg.storage.dir + cfg.storage.results_cache_name)
#     na_cached_results_dict = debug_results_cache.get(cfg.threshold_variation_exp_key, {})

#     for exp_key in cfg.threshold_exps_keys:
#         cfg.exp_key = exp_key
#         na2acc = {}
#         for n_act_t in cfg.neuron_act_thresholds:
#             temp_cfg = copy.deepcopy(cfg)
#             temp_cfg.neuron_activation_threshold = n_act_t
#             r2results = run_fed_debug_differential_testing(temp_cfg, store_in_cache=False)["round2debug_result"]

#             all_accs = [r["accuracy"] for r in r2results]
#             avg_accs = sum(all_accs) / len(all_accs)

#             log(
#                 INFO,
#                 f"Neuron Activation threshold {n_act_t} "
#                 f"and average malicious client localization accuracy is {avg_accs}.",
#             )
#             na2acc[n_act_t] = avg_accs

#         na_cached_results_dict[cfg.exp_key] = {
#             "cfg": debug_results_cache[exp_key]["debug_cfg"],
#             "na2acc": na2acc,
#         }

#     debug_results_cache[cfg.threshold_variation_exp_key] = na_cached_results_dict
