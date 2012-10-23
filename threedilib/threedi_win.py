# Windows 3Di lib
import shlex
import subprocess
import os
import logging
from shutil import copyfile


logger = logging.getLogger(__name__)

subgrid_root = "C:\\3di\\subgrid"
subgrid_exe = os.path.join(subgrid_root, "subgridf90.exe")
copy_to_work_dir = [
    'subgrid.ini',
    'unstruc.hlp',
    'isocolour.hls',
    'land.rgb',
    'water.rgb',
    'interception.rgb',
    'land.zrgb',
    'interact.ini',
    'UNSA_SIM.inp',
    'ROOT_SIM.INP',
    'CROP_OW.PRN',
    'CROPFACT',
    'EVAPOR.GEM',
    ]


def setup_3di(full_path):
    """
    Copies default files to 3di work folder
    """
    logger.info('Setting up working directory in %s...' % full_path)

    # Copy default files
    for filename in copy_to_work_dir:
        dst_filename = os.path.join(full_path, filename)
        if not os.path.exists(dst_filename):
            src_filename = os.path.join(subgrid_root, filename)
            logger.info('Copying %s from defaults...' % filename)
            copyfile(src_filename, dst_filename)
        else:
            logger.info('%s exists, ok.' % filename)
    # TODO: Extra checks, update files, etc.
    # subgrid.ini


def setup_and_run_3di(mdu_full_path, skip_if_results_available=True):
    """This wil produce a file located on result_filename.

    You are responsible for moving the file away.
    """
    full_path = os.path.dirname(mdu_full_path)
    result_filename = os.path.join(full_path, 'subgrid_map.nc')

    # Go to working dir
    os.chdir(full_path)

    if os.path.exists(result_filename) and skip_if_results_available:
        print 'skipping calculation, already calculated.'
    else:
        setup_3di(full_path)
        # Run
        os.system('%s %s' % (subgrid_exe, mdu_full_path))

        # Results in full_path + subgrid_map.nc
        # *.tim is also produced.

    # # process results
    # process_3di_nc(result_filename)
    return result_filename


def main():
    mdu_full_path = "Z:\\git\\sites\\flooding\\driedi\\Vecht\\vecht_autostartstop.mdu"
    setup_and_run_3di(mdu_full_path)


if __name__ == '__main__':
    main()

