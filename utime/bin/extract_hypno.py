import logging
import numpy as np
import re
from argparse import ArgumentParser, Namespace
from glob import glob
from pathlib import Path
from typing import List, Optional, Union
from psg_utils.io.hypnogram import extract_ids_from_hyp_file
from psg_utils.hypnogram.utils import fill_hyp_gaps
from utime.utils.scriptutils import add_logging_file_handler

logger = logging.getLogger(__name__)


def get_argparser():
    parser = ArgumentParser(description='Extract hypnograms from various'
                                        ' file formats.')
    parser.add_argument("--file_pattern", type=str,
                        help='A glob pattern matching all files to extract '
                             'from (e.g., "data/**/*.xml")')
    parser.add_argument("--out_dir", type=str,
                        help="Directory in which extracted files will be "
                             "stored")
    parser.add_argument("--fill_blanks", type=str, default=None,
                        help="A stage string value to insert into the hypnogram when gaps "
                             "occour, e.g. 'UNKNOWN' or 'Not Scored', etc.")
    parser.add_argument("--extract_func", type=str, default=None,
                        help="Name of hyp extraction function. If not specified, the file extension defines the "
                             "function to use.")
    parser.add_argument("--remove_offset", action="store_true",
                        help="Remove potential offsets so that the first sleep stage always starts at init sec 0.")
    parser.add_argument("--correct_zero_durations", type=int, default=None, help="Optionally change any stage with duration "
                                                                                 "0 seconds to some other duration. E.g., --correct_zero_durations 30 will set those events to 30 seconds.")
    parser.add_argument("--use_dir_names", action="store_true",
                         help='Use the parent directory name as the hypnogram name')
    parser.add_argument("--id_regex", type=str, default=None,
                         help='Regular expression pattern with a capture group to extract ID from filename. '
                              'E.g. "(.*)-nsrr" would extract "mesa-sleep-1234" from "mesa-sleep-1234-nsrr.xml"')
    parser.add_argument("--ignore_extraction_errors", action="store_true",
                        help="Ignore extraction errors and continue with next file.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing files of identical name and log files")
    parser.add_argument("--log_file", type=str, default="hyp_extraction_log",
                        help="Relative path (from Defaults.LOG_DIR as specified by ut --log_dir flag) of "
                             "output log file for this script. "
                             "Set to an empty string to not save any logs to file for this run. "
                             "Default is 'hyp_extraction_log'")
    return parser


def to_ids(start: np.ndarray, durs: np.ndarray, stage: List[str], out: Union[str, Path]) -> None:
    with open(out, "w") as out_f:
        for i, d, s in zip(start, durs, stage):
            out_f.write("{},{},{}\n".format(int(i), int(d), s))


def extract_id_with_regex(filename: Union[str, Path], pattern: Optional[str]) -> str:
    """
    Extract ID from filename using a regex pattern with a capture group
    Example: with pattern "mesa-sleep-(.*)-nsrr" and filename "mesa-sleep-1234-nsrr.xml"
    this would extract "1234"
    
    Args:
        filename: The filename to extract ID from
        pattern: Regex pattern with exactly one capture group
        
    Returns:
        The extracted ID or the original filename if no match
    """
    if not pattern:
        return str(filename)
    
    # Convert to Path and get stem (filename without extension)
    path = Path(filename)
    name = path.stem
    
    try:
        match = re.search(pattern, name)
        if match and match.groups():
            return match.group(1)
    except re.error as e:
        logger.warning(f"Invalid regex pattern '{pattern}': {str(e)}")
    
    return name

def remove_offset(inits):
    offset = inits[0]
    for i in range(len(inits)):
        new_init = inits[i] - offset
        rounded_new_init = np.round(new_init)
        if new_init - rounded_new_init > 1e-6:
            raise ValueError(f"Unexpectedly large difference of {new_init - rounded_new_init} between new_init of "
                             f"{new_init} and round(new_init) of {rounded_new_init} when "
                             "removing offset. The implementation expects inits to land on whole-seconds, not "
                             "fractions.")
        inits[i] = new_init
    return inits


def run(args: Namespace) -> None:
    logger.info(f"Args dump: {vars(args)}")
    files = glob(args.file_pattern)
    out_dir = Path(args.out_dir).absolute()
    out_dir.mkdir(parents=True, exist_ok=True)
    n_files = len(files)
    logger.info(f"Found {n_files} files matching glob statement")
    if n_files == 0:
        return
    logger.info(f"Saving .ids files to '{out_dir}'")
    if n_files == 0:
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    for i, file_ in enumerate(files):
        file_path = Path(file_)
        
        name = file_path.parent.name if args.use_dir_names else file_path.stem

        if args.id_regex:
            name = extract_id_with_regex(file_path, args.id_regex)
        
        out_dir_subject = out_dir / name
        out = out_dir_subject / f"{name}.ids"
        logger.info(f"{i+1}/{n_files} Processing {name}\n"
                    f"-- In path    {file_path}\n"
                    f"-- Out path   {out}")
        out_dir_subject.mkdir(exist_ok=True)
        if out.exists():
            if not args.overwrite:
                continue
            out.unlink()
        try:
            inits, durs, stages = extract_ids_from_hyp_file(file_,
                                                            period_length=30,
                                                            extract_func=args.extract_func,
                                                            replace_zero_durations=args.correct_zero_durations)
        except Exception as e:
            if args.ignore_extraction_errors:
                logger.warning(f"Error extracting hypnogram from file {file_}: {str(e)}")
                continue
            raise
        if args.remove_offset:
            inits = remove_offset(inits)
        if args.fill_blanks:
            inits, durs, stages = fill_hyp_gaps(inits, durs, stages, args.fill_blanks)
        to_ids(inits, durs, stages, out)


def entry_func(args=None):
    # Get the script to execute, parse only first input
    parser = get_argparser()
    args = parser.parse_args(args)
    add_logging_file_handler(args.log_file, args.overwrite, mode="w")
    run(args)


if __name__ == "__main__":
    entry_func()
