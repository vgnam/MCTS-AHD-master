from source.ab_mcts_ahd import AB_MCTS_A_AHD
from source.getParas import Paras
from source import prob_rank, pop_greedy
from problem_adapter import Problem

from utils.utils import init_client

class AB_AHD:
    def __init__(self, cfg, root_dir, workdir) -> None:
        self.cfg = cfg
        self.root_dir = root_dir
        self.problem = Problem(cfg, root_dir)

        self.paras = Paras()
        self.paras.set_paras(method = "ab_mcts_ahd",
                             init_size = self.cfg.init_pop_size,
                             pop_size = self.cfg.pop_size,
                             ec_fe_max = self.cfg.max_fe,
                             exp_output_path = f"{workdir}/",
                             exp_debug_mode = False,
                             eva_timeout=cfg.timeout)

    def evolve(self):
        print("- Evolution Start -")

        method = AB_MCTS_A_AHD(self.paras, self.problem, prob_rank, pop_greedy)

        results = method.run()

        print("> End of Evolution! ")
        print("-----------------------------------------")
        print("---  AB-MCTS-AHD successfully finished!  ---")
        print("-----------------------------------------")

        return results


