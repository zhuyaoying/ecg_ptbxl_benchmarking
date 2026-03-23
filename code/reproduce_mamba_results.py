from experiments.scp_experiment import SCP_Experiment
from utils import utils
from configs.mamba_configs import MAMBA_MODEL_CONFIGS


def main():
    datafolder = '../data/ptbxl/'
    outputfolder = '../output/'

    experiments = [
        ('exp_mamba_all', 'all'),
        ('exp_mamba_diagnostic', 'diagnostic'),
        ('exp_mamba_subdiagnostic', 'subdiagnostic'),
        ('exp_mamba_superdiagnostic', 'superdiagnostic'),
        ('exp_mamba_form', 'form'),
        ('exp_mamba_rhythm', 'rhythm'),
    ]

    for name, task in experiments:
        e = SCP_Experiment(name, task, datafolder, outputfolder, MAMBA_MODEL_CONFIGS)
        e.prepare()
        e.perform()
        e.evaluate()

    utils.generate_ptbxl_summary_table()


if __name__ == '__main__':
    main()
