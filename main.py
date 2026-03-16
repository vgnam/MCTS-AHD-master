import hydra
import logging
import os
import traceback
from pathlib import Path
import subprocess
# from utils.utils import init_client



# API keys

# [2026-01-01 20:57:13,912][root][INFO] - [*] Running ...
# [2026-01-01 20:57:13,912][root][INFO] - [*] Average for 20: 4.843840492309927
# [2026-01-01 20:57:13,912][root][INFO] - [*] Average for 50: 8.603443911600397
# [2026-01-01 20:57:13,912][root][INFO] - [*] Average for 100: 15.610836345009186,
#
# [2026-01-01 09:41:42,917][root][INFO] - [*] Running ...
# [2026-01-01 09:41:42,917][root][INFO] - [*] Average for 20: 4.87364156701144
# [2026-01-01 09:41:42,917][root][INFO] - [*] Average for 50: 9.21774709920679
# [2026-01-01 09:41:42,917][root][INFO] - [*] Average for 100: 15.77815229937681,
#
#
# [2026-01-02 11:08:59,784][root][INFO] - [*] Average for 20: 4.8095718357584545
# [2026-01-02 11:08:59,784][root][INFO] - [*] Average for 50: 9.191845707996968
# [2026-01-02 11:08:59,784][root][INFO] - [*] Average for 100: 16.090914473409164

# os.environ["GEMINI_API_KEY"] = ("AIzaSyB7_zB7UcJ18cmFKMzrGtreSm"
#                                 "1000zDBUeI-Rc")
# os.environ["MISTRAL_API_KEY"] = "gz7HXe7UZEJBmbez2brM07sUPGDN4WKm"

os.environ["MISTRAL_API_KEY"] = "v0ReHEBIecibPUAQFCw8m5oWkg7eJP3r"

#
os.environ['NVIDIA_NIM_API_KEY'] = "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"

ROOT_DIR = os.getcwd()
logging.basicConfig(level=logging.INFO)


@hydra.main(version_base=None, config_path="cfg", config_name="config")
def main(cfg):
    try:
        workspace_dir = Path.cwd()
        print(workspace_dir)

        # Logging info
        logging.info(f"Workspace: {workspace_dir}")
        logging.info(f"Project Root: {ROOT_DIR}")
        logging.info(f"Using LLM: {cfg.get('model', cfg.llm_client.model)}")
        logging.info(f"Using Algorithm: {cfg.algorithm}")


        # Select algorithm adapter
        if cfg.algorithm == "mcts-ahd":
            from ahd_adapter import AHD as LHH
        elif cfg.algorithm == "ab-mcts-ahd":
            from ab_mcts_adapter import AB_AHD as LHH
        else:
            raise NotImplementedError(f"Unknown algorithm: {cfg.algorithm}")

        # Main algorithm
        lhh = LHH(cfg, ROOT_DIR, workspace_dir)
        best_code_overall, best_code_path_overall = lhh.evolve()
        logging.info(f"Best Code Overall: {best_code_overall}")
        logging.info(f"Best Code Path Overall: {best_code_path_overall}")


        # Write best code to file
        problem_dir = Path(ROOT_DIR) / "problems" / cfg.problem.problem_name
        gpt_file = problem_dir / "gpt.py"
        with open(gpt_file, 'w') as file:
            file.writelines(best_code_overall + '\n')


        # Run validation and redirect stdout
        test_script = problem_dir / "eval.py"
        test_script_stdout = "best_code_overall_val_stdout.txt"

        logging.info(f"Running validation script: {test_script}")
        with open(test_script_stdout, 'w') as stdout:
            subprocess.run(
                ["python", str(test_script), "-1", ROOT_DIR, "val"],
                stdout=stdout,
                stderr=subprocess.STDOUT,  # capture errors too
                check=False
            )

        logging.info(f"Validation scri"
                     f"pt finished. Results saved in {test_script_stdout}.")

        # Print results
        with open(test_script_stdout, 'r') as file:
            for line in file:
                logging.info(line.strip())

    except Exception as e:
        logging.error("An error occurred during execution!")
        traceback.print_exc()   # In ra full trace


if __name__ == "__main__":
    main()
