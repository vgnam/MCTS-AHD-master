"""Adapters for the MoH and HiFo-Prompt baselines (same interface as AB_AHD)."""

from source.getParas import Paras
from source import prob_rank, pop_greedy
from problem_adapter import Problem


class _BaselineAdapter:
    method = None
    label = None

    def __init__(self, cfg, root_dir, workdir) -> None:
        self.cfg = cfg
        self.root_dir = root_dir
        self.problem = Problem(cfg, root_dir)
        self.paras = Paras()
        self.paras.set_paras(method=self.method,
                             init_size=self.cfg.init_pop_size,
                             pop_size=self.cfg.pop_size,
                             ec_fe_max=self.cfg.max_fe,
                             exp_output_path=f"{workdir}/",
                             llm_model=self.cfg.model,
                             llm_model_names=[self.cfg.model],
                             exp_debug_mode=False,
                             eva_timeout=cfg.timeout)

    def build(self):
        raise NotImplementedError

    def evolve(self):
        print("- Evolution Start -")
        results = self.build().run()
        print("> End of Evolution! ")
        print("-----------------------------------------")
        print(f"---  {self.label} successfully finished!  ---")
        print("-----------------------------------------")
        return results


class MoH_AHD_Adapter(_BaselineAdapter):
    method = "moh"
    label = "MoH"

    def build(self):
        from source.moh import MoH_AHD
        return MoH_AHD(self.paras, self.problem, prob_rank, pop_greedy, cfg=self.cfg)


class HiFo_AHD_Adapter(_BaselineAdapter):
    method = "hifo_prompt"
    label = "HiFo-Prompt"

    def build(self):
        from source.hifo_prompt import HiFo_AHD
        return HiFo_AHD(self.paras, self.problem, prob_rank, pop_greedy, cfg=self.cfg)
