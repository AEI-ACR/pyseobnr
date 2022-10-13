import argparse
import os

from bilby.core.utils import check_directory_exists_and_if_not_mkdir
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
    p.add_argument(
        "--wrapper-path", type=str, help="The path to the wrapper, including name"
    )
    args = p.parse_args()
    # Create submit files for the different tests
    file_loader = FileSystemLoader(f"{this_folder}/templates/")
    env = Environment(loader=file_loader)
    template = env.get_template("slurm.jinja")

    check_directory_exists_and_if_not_mkdir(args.test_dir)
    os.chdir(args.test_dir)
    # Mismatch with NR
    cmd = f"""python {script_home}/auxiliary/sanity_checks/NR_mismatches.py --cases-file /work/lpompili/csv_files/spinning_calibs.csv"""
    prog = template.render(
        job_name="NR_mismatch",
        label="sanity_check",
        ncores=args.n_cores,
        queue=args.queue,
        time=args.run_time,
        exec_script=cmd,
        env=args.env,
    )
    with open("NR_mismatches.sh", "w") as fw:
        fw.write(prog)
    # Mismatch with NRSur3dq8
    cmd = f"""python {script_home}/auxiliary/sanity_checks/NRSur_matches.py --points 5000 --chi-max 0.9 --q-max 8.0 --name mm_3dq8_90.dat --model-name NRHybSur3dq8 --approximant SEOBNRv5HM"""
    prog = template.render(
        job_name="NRSur3dq8_mismatch",
        label="sanity_check",
        ncores=args.n_cores,
        queue=args.queue,
        time=args.run_time,
        exec_script=cmd,
        env=args.env,
    )
    with open("NRSur3dq8_mismatches.sh", "w") as fw:
        fw.write(prog)

    # Mismatch with NRSur2dq15
    cmd = f"""python {script_home}auxiliary/sanity_checks/NRSur_matches.py --points 5000 --chi-max 0.6 --q-max 15.0 --name mm_2dq15_60.dat --model-name NRHybSur2dq15 --approximant SEOBNRv5HM"""
    prog = template.render(
        job_name="NRSur2dq15_mismatch",
        label="sanity_check",
        ncores=args.n_cores,
        queue=args.queue,
        time=args.run_time,
        exec_script=cmd,
        env=args.env,
    )
    with open("NRSur2dq15_mismatches.sh", "w") as fw:
        fw.write(prog)

    # Mismatch with SEOBNRv4HM - q<=10
    cmd = f"""python {script_home}auxiliary/sanity_checks/EOB_matches.py --points 10000 --chi-max 0.998 --q-max 10 --plots --model-1-name SEOBNRv4HM --model-2-name SEOBNRv5HM --n-cpu 64 --name comp"""
    prog = template.render(
        job_name="SEOBNRv4HM_mismatch",
        label="sanity_check",
        ncores=args.n_cores,
        queue=args.queue,
        time=args.run_time,
        exec_script=cmd,
        env=args.env,
    )
    with open("SEOBNv4HM_mismatches.sh", "w") as fw:
        fw.write(prog)

    # Mismatch with SEOBNRv4HM - q>10
    cmd = f"""python {script_home}auxiliary/sanity_checks/EOB_matches.py --points 3000 --q-min 10.0 --chi-max 0.96 --q-max 100 --plots --model-1-name SEOBNRv4HM --model-2-name SEOBNRv5HM --n-cpu 64 --name ext"""
    prog = template.render(
        job_name="SEOBNRv4HM_mismatch_ext",
        label="sanity_check",
        ncores=args.n_cores,
        queue=args.queue,
        time=args.run_time,
        exec_script=cmd,
        env=args.env,
    )
    with open("SEOBNv4HM_mismatches_ext.sh", "w") as fw:
        fw.write(prog)



    # Amplitude monotonicity
    cmd = f"""python {script_home}auxiliary/sanity_checks/monotonicity_test.py --points 100000  --quantity amplitude"""
    prog = template.render(
        job_name="amplitude_monotonicity",
        label="sanity_check",
        ncores=args.n_cores,
        queue=args.queue,
        time=args.run_time,
        exec_script=cmd,
        env=args.env,
    )
    with open("amp_monotonicity.sh", "w") as fw:
        fw.write(prog)

    # Amplitude monotonicity (more cases for high q)
    cmd = f"""python {script_home}auxiliary/sanity_checks/monotonicity_test.py --q-min 70.0 --points 500000 --name monotonicity_test_high_q  --quantity amplitude"""
    prog = template.render(
        job_name="amplitude_monotonicity_high_q",
        label="sanity_check",
        ncores=args.n_cores,
        queue=args.queue,
        time=args.run_time,
        exec_script=cmd,
        env=args.env,
    )
    with open("amp_monotonicity_high_q.sh", "w") as fw:
        fw.write(prog)

    # Frequency mononicity
    cmd = f"""python {script_home}auxiliary/sanity_checks/monotonicity_test.py --points 100000  --quantity frequency"""
    prog = template.render(
        job_name="frequency_monotonicity",
        label="sanity_check",
        ncores=args.n_cores,
        queue=args.queue,
        time=args.run_time,
        exec_script=cmd,
        env=args.env,
    )
    with open("freq_monotonicity.sh", "w") as fw:
        fw.write(prog)

    # Amplitude hierarchy
    cmd = f"""python {script_home}auxiliary/sanity_checks/amplitude_hierarchy_test.py --points 100000 --q-max 100 --chi-max 0.998"""
    prog = template.render(
        job_name="amplitude_hierarchy",
        label="sanity_check",
        ncores=args.n_cores,
        queue=args.queue,
        time=args.run_time,
        exec_script=cmd,
        env=args.env,
    )
    with open("amplitude_hierarchy.sh", "w") as fw:
        fw.write(prog)

    # Smoothness varying q, chi
    cmd = f"""python {script_home}auxiliary/sanity_checks/smoothness_q_chi.py"""
    prog = template.render(
        job_name="smoothness_plots",
        label="sanity_check",
        ncores=args.n_cores,
        queue=args.queue,
        time=args.run_time,
        exec_script=cmd,
        env=args.env,
    )
    with open("smoothness_plots.sh", "w") as fw:
        fw.write(prog)
