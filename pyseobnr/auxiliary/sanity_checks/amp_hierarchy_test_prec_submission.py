import argparse
import os
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

this_folder = os.path.abspath(os.path.dirname(__file__))
script_home = os.path.join(this_folder, "../../")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--n-cores",
        type=int,
        help="Number of cores to use. Should match number of tasks per node",
        default=64,
    )
    p.add_argument("--queue", type=str, help="The queue to use", default="syntrofos")
    p.add_argument(
        "--run-time",
        type=int,
        help="Number of hours to ask from the cluster",
        default=23,
    )
    p.add_argument(
        "--env", type=str, help="The path to the python env one wishes to use"
    )
    p.add_argument(
        "--test-dir",
        type=str,
        help="Name of directory where tests will be created",
        default="./",
    )
    p.add_argument("--points", type=int, help="Number of points", default=100000)
    p.add_argument(
        "--wrapper-path", type=str, help="The path to the wrapper, including name"
    )
    p.add_argument("--seed", type=int, help="Seed for random generation use", default=150914)
    args = p.parse_args()
    # Create submit files for the different tests
    file_loader = FileSystemLoader(f"{this_folder}/templates/")
    env = Environment(loader=file_loader)
    template = env.get_template("slurm.jinja")

    Path(args.test_dir).mkdir(parents=True, exist_ok=True)
    os.chdir(args.test_dir)

    # Amplitude hierarchy
    cmd = f"""python {script_home}auxiliary/sanity_checks/amplitude_hierarchy_test_precessing.py --points {args.points} --q-max 100 --a-max 0.998 --seed {args.seed} --wrapper-path {args.wrapper_path} """
    prog = template.render(
        job_name="amplitude_hierarchy_test_prec",
        label="sanity_check",
        ncores=args.n_cores,
        queue=args.queue,
        time=args.run_time,
        exec_script=cmd,
        env=args.env,
    )
    with open("amplitude_hierarchy.sh", "w") as fw:
        fw.write(prog)
