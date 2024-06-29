import logging
import hydra
import time
from diskcache import Index
from multiprocessing import Pool
from torch.nn.init import  kaiming_uniform_
from typing import Dict, List, Optional, Tuple
import copy

from models import initialize_model
from differential_testing import FaultyClientLocalization, InferenceGuidedInputs
from dataset_preparation import train_test_transforms_factory


class FedDebug:
    def __init__(self, cfg, round_key) -> None:
        self.cfg = cfg
        self.round_key = round_key
        

        self.random_generator=kaiming_uniform_
        self._extractRoundID()
        self._loadTrainingConfig()
        self._initializeAndLoadModels()
         
    

    def _extractRoundID(self) -> None:
        self.round_id = self.round_key.split(":")[-1]

    def _loadTrainingConfig(self) -> None:
        self.training_cache = Index(self.cfg.storage.dir + self.cfg.storage.train_cache_name)
        exp_dict = self.training_cache[self.cfg.exp_key]
        self.train_cfg = exp_dict["train_cfg"]  # type: ignore
        self.num_bugs = len(self.train_cfg.faulty_clients_ids) # type: ignore
        self.input_shape = tuple(list(exp_dict["input_shape"]))  # type: ignore 
        self.transform_func = train_test_transforms_factory(self.train_cfg.data_dist)['test']
        



    def _initializeAndLoadModels(self) -> None:
        logging.info(
            f'\n\n             ----------Round key {self.round_key} -------------- \n')
        round2ws = self.training_cache[self.round_key]
        self.client2num_examples = round2ws["client2num_examples"] # type: ignore
          
        self.client2model = {}
        for cid, ws in round2ws["client2ws"].items():  # type: ignore
            cmodel = initialize_model(self.train_cfg.model.name, self.train_cfg.dataset)['model'] 
            cmodel.load_state_dict(ws)  # type: ignore
            cmodel = cmodel.cpu().eval()  # type: ignore
            self.client2model[cid] = cmodel
    
    def _evaluateFaultLocalization(self, predicted_faulty_clients_on_each_input):
        true_faulty_clients = set(self.train_cfg.faulty_clients_ids)
        detection_acc = 0
        for pred_faulty_clients in predicted_faulty_clients_on_each_input:
            print(f"+++ Faulty Clients {pred_faulty_clients}")
            correct_localize_faults = len(
                true_faulty_clients.intersection(pred_faulty_clients))
            acc = (correct_localize_faults/len(true_faulty_clients))*100
            detection_acc += acc
        fault_localization_acc = detection_acc / \
            len(predicted_faulty_clients_on_each_input)
        return fault_localization_acc

    def _help_run(self, k_gen_inputs=10, na_threshold=0.003, use_gpu=True):
        print(">  Running FaultyClientLocalization ..")
        generate_inputs = InferenceGuidedInputs(self.client2model, self.input_shape, randomGenerator= self.random_generator, transform_func=self.transform_func, min_nclients_same_pred=3, k_gen_inputs=k_gen_inputs)

        selected_inputs, input_gen_time = generate_inputs.getInputs()
        # print(selected_inputs)

        start = time.time()
        faultyclientlocalization = FaultyClientLocalization(self.client2model, selected_inputs, use_gpu=use_gpu)

        potential_faulty_clients_for_each_input = faultyclientlocalization.runFaultLocalization(na_threshold, num_bugs=self.num_bugs)
        fault_localization_time = time.time()-start
        return potential_faulty_clients_for_each_input, input_gen_time, fault_localization_time


    # def _computeEvalMetrics(self, input2debug: List[Dict]) -> Dict[str, float]:
    #     correct_tracing = 0
    #     return {"accuracy": correct_tracing / len(self.subset_test_data)}

    def run(self) -> Dict[str, any]:  # type: ignore
        predicted_faulty_clients, input_gen_time, fault_localization_time = self._help_run()

        fault_localization_acc = self._evaluateFaultLocalization(predicted_faulty_clients)

        eval_metrics = {'accuracy': fault_localization_acc}

        logging.info(f'Fault Localization Accuracy: {fault_localization_acc}')

        
        debug_result = {
            "clients": list(self.client2model.keys()),
            "eval_metrics": eval_metrics,
            "fault_localization_time": fault_localization_time,
            "input_gen_time": input_gen_time,
            "round_id": self.round_id,
        }

        return debug_result


def _getRoundKeysAndCentralTestData(fl_key, train_cache_path):
    training_cache = Index(train_cache_path)
    r_keys = []
    for k in training_cache.keys():
        if fl_key == k:
            continue
        elif fl_key in k and len(k) > len(fl_key):
            r_keys.append(k)
    return r_keys


def _checkAlredyDone(fl_config_key: str, results_cache):
    if fl_config_key in results_cache.keys():
        d = results_cache[fl_config_key]        
        return d["round2debug_result"]
    return []




def _roundLambdadebug(cfg, round_key):
    round_debug = FedDebug(cfg, round_key)
    debug_result_dict = round_debug.run() 
    return debug_result_dict


def run_fed_debug(cfg):
    train_cache_path = cfg.storage.dir + cfg.storage.train_cache_name
    debug_results_cache = Index(cfg.storage.dir + cfg.storage.debug_cache_name)
    
    # round2debug_result = _checkAlredyDone(cfg.exp_key, debug_results_cache)
    round2debug_result = []

    if len(round2debug_result) > 0:
        logging.info(f">> Debugging is already done.")
        return round2debug_result

    rounds_keys = _getRoundKeysAndCentralTestData(
        cfg.exp_key, train_cache_path)
    logging.debug(f"rounds_keys {rounds_keys}")

    

    start_time = time.time()

    if cfg.parallel_processes > 1:        
        with Pool(processes=cfg.parallel_processes) as p:
            logging.info(
                f"Running Debugging analysis for {len(rounds_keys)} rounds in parallel...")
            round2debug_result = p.starmap(_roundLambdadebug, [(
                cfg, round_key) for round_key in rounds_keys])
            p.close()
            p.join()
    else:
        round2debug_result = [_roundLambdadebug(cfg, round_key) for round_key in rounds_keys]

    end_time = time.time()

    avg_debug_time_per_round = (end_time - start_time) / len(rounds_keys)

    debug_results_cache[cfg.exp_key] = {
        "round2debug_result": round2debug_result, "debug_cfg": cfg, "training_cache_path": train_cache_path, "avg_debug_time_per_round": avg_debug_time_per_round}

    logging.info(
        f"Debugging results saved for {cfg.exp_key}, avg Debugging time per round: {avg_debug_time_per_round} seconds")
    
    return round2debug_result

@hydra.main(config_path="conf", config_name="debug", version_base=None)
def main(cfg):
    if len(cfg.all_exp_keys) > 0:
        for k in cfg.all_exp_keys:
            new_cfg = copy.deepcopy(cfg)
            new_cfg.exp_key = k
            run_fed_debug(new_cfg)
    else:
        run_fed_debug(cfg)

if __name__ == "__main__":
    main()
