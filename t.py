import logging
import os
from pathlib import Path
import subprocess
import hydra

# from ahd_adapter import AHD as LHH

os.environ["GEMINI_API_KEY"] = "AIzaSyB7_zB7UcJ18cmFKMzrGtreSmzDBUeI-Rc"
os.environ["MISTRAL_API_KEY"] = "59cjVL5qVzQqKN5CfRlhIhJvaDQT8jwt"

ROOT_DIR = os.getcwd()
logging.basicConfig(level=logging.INFO)


@hydra.main(version_base=None, config_path="cfg", config_name="config")
def main(cfg):
    workspace_dir = Path.cwd()
    print(workspace_dir)
    # Set logging level
    logging.info(f"Workspace: {workspace_dir}")
    logging.info(f"Project Root: {ROOT_DIR}")
    logging.info(f"Using LLM: {cfg.get('model', cfg.llm_client.model)}")
    logging.info(f"Using Algorithm: {cfg.algorithm}")

    if cfg.algorithm == "mcts-ahd":
        from ahd_adapter import AHD as LHH
    elif cfg.algorithm == "ab-mcts-ahd":
        from ab_mcts_adapter import AB_AHD as LHH
    else:
        raise NotImplementedError
    # client = init_client(cfg)

    # Main algorithm
    lhh = LHH(cfg, ROOT_DIR, workspace_dir)
    best_code_overall, best_code_path_overall = lhh.evolve()
    logging.info(f"Best Code Overall: {best_code_overall}")
    logging.info(f"Best Code Path Overall: {best_code_path_overall}")

    # Run validation and redirect stdout to a file "best_code_overall_stdout.txt"
    with open(f"{ROOT_DIR}/problems/{cfg.problem.problem_name}/gpt.py", 'w') as file:
        file.writelines(best_code_overall + '\n')
    test_script = f"{ROOT_DIR}/problems/{cfg.problem.problem_name}/eval.py"
    test_script_stdout = "best_code_overall_val_stdout.txt"
    logging.info(f"Running validation script...: {test_script}")
    with open(test_script_stdout, 'w') as stdout:
        subprocess.run(["python", test_script, "-1", ROOT_DIR, "val"], stdout=stdout)
    logging.info(f"Validation script finished. Results are saved in {test_script_stdout}.")

    # Print the results
    with open(test_script_stdout, 'r') as file:
        for line in file.readlines():
            logging.info(line.strip())


if __name__ == "__main__":
    main()