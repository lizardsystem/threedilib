# Windows 3Di lib, don't know if this works
import shlex
import subprocess
import os
import logging
from shutil import copyfile
from threedi import setup_and_run_3di


logger = logging.getLogger(__name__)

subgrid_root = "C:\\3di\\subgrid"
subgrid_exe = os.path.join(subgrid_root, "subgridf90.exe")


def setup_and_run_3di_win(mdu_full_path, skip_if_results_available=True):
    return setup_and_run_3di(
        mdu_full_path,
        skip_if_results_available=skip_if_results_available
        subgrid_root=subgrid_root,
        subgrid_exe=subgrid_exe)


def main():
    mdu_full_path = "Z:\\git\\sites\\flooding\\driedi\\Vecht\\vecht_autostartstop.mdu"
    setup_and_run_3di_win(mdu_full_path)


if __name__ == '__main__':
    main()

