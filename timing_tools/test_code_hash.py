#!/usr/bin/env python

from uuid import uuid4
from gevent import monkey; monkey.patch_all()

import argparse
import functools
import subprocess
import gevent
import os
import tempfile
import csv
import io

_IACA_HEADER = "7f454c4602010100000000000000000001003e000100000000000000000000000000000000000000100100000000000000000000400000000000400004000100bb6f000000646790"
_IACA_TAIL = "bbde000000646790"

_LLVM_BODY = '''        .text
        .att_syntax
        .globl          main
main:
        # LLVM-MCA-BEGIN test
{}
        # LLVM-MCA-END test
    '''

_IACA = os.path.join(os.environ['ITHEMAL_HOME'], 'timing_tools', 'iaca-bin')
_LLVM = os.path.join(os.environ['ITHEMAL_HOME'], 'timing_tools', 'llvm-build', 'bin', 'llvm-mca')
_DISASSMBLER = os.path.join(os.environ['ITHEMAL_HOME'], 'data_collection', 'disassembler', 'build', 'disassemble')

# print('Disassembler: {}'.format(_DISASSMBLER))
# print('IACA: {}'.format(_IACA))
# print('LLVM: {}'.format(_LLVM))

def time_llvm_base(arch, verbose, code):
    with tempfile.NamedTemporaryFile() as f:
        disassembler = subprocess.Popen([_DISASSMBLER, '-att'], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        (output, _) = disassembler.communicate(code.encode('utf-8'))
        f.write(_LLVM_BODY.format(output.decode('utf-8')).encode('utf-8'))
        f.flush()
        output = subprocess.check_output([_LLVM, '-march=x86', '-mcpu={}'.format(arch), f.name]).decode('utf-8')
        if verbose:
            print(output)
        return output

def time_llvm_cycles(arch, verbose, code):
    # print(code, arch, verbose)
    output = time_llvm_base(arch, verbose, code)
    total_cycles_line = output.split('\n')[5]
    cycles = total_cycles_line.split()[2]
    return float(cycles)

def time_llvm_rthroughput(arch, verbose, code):
    output = time_llvm_base(arch, verbose, code)
    total_cycles_line = output.split('\n')[11]
    cycles = total_cycles_line.split()[2]
    return float(cycles) * 100

def time_iaca(arch, verbose, code):
    with tempfile.NamedTemporaryFile() as f:
        f.write('{}{}{}'.format(_IACA_HEADER, code, _IACA_TAIL).decode('hex'))
        f.flush()
        output = subprocess.check_output([_IACA, '-arch', arch, '-reduceout', f.name])
        if verbose:
            print(output)
        txput_line = output.split('\n')[3]
        txput = txput_line.split()[2]
        return float(txput) * 100

def time_code_ids(code_ids, timer):
    # get code
    mysql = subprocess.Popen(['mysql', '-N'], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    (out, _) = mysql.communicate('SELECT code_id, code_raw FROM code WHERE code_id IN ({});\n'.format(','.join(map(str, code_ids))))
    jobs = {int(code_id): gevent.spawn(timer, code_raw) for (code_id, code_raw) in map(str.split, out.rstrip().split('\n'))}
    gevent.joinall(list(jobs.values()), timeout=240)
    return {code_id: jobs[code_id].value for code_id in jobs}

def time_code_hashes(raw_codes: list[str], timer):
    # get code
    # print('Timing {} codes'.format(len(raw_codes)))
    jobs = {code_raw: gevent.spawn(timer, code_raw) for code_raw in raw_codes}
    gevent.joinall(list(jobs.values()), timeout=240)
    return {code_id: jobs[code_id].value for code_id in jobs}

iaca_kind = (2, time_iaca, {'haswell': 'HSW', 'broadwell': 'BDW', 'skylake': 'SKL'})
llvm_kind_cycles = (3, time_llvm_cycles, {'haswell': 'haswell', 'broadwell': 'broadwell', 'skylake': 'skylake', 'nehalem': 'nehalem', 'ivybridge': 'ivybridge'})
llvm_kind_rthroughput = (5, time_llvm_rthroughput, {'haswell': 'haswell', 'broadwell': 'broadwell', 'skylake': 'skylake', 'nehalem': 'nehalem', 'ivybridge': 'ivybridge'})

_kind_map = {
    'iaca': iaca_kind,
    'llvm-cycles': llvm_kind_cycles,
    'llvm-rthroughput': llvm_kind_rthroughput,
}

_arch_map = {
    'haswell': 1,
    'skylake': 2,
    'broadwell': 3,
    'nehalem': 4,
    'ivybridge': 5,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('arch', type=str)
    parser.add_argument('kind')
    parser.add_argument('--insert', action='store_true', default=False)
    parser.add_argument('--verbose', action='store_true', default=False)
    parser.add_argument('code_hash', type=str, nargs='+')
    args = parser.parse_args()

    (kind_id, timer_func, arch_dict) = _kind_map[args.kind]
    arch_id = _arch_map[args.arch]
    # print('Arch: {} Timer: {}'.format(arch_dict[args.arch], timer_func.__name__))
    timer = functools.partial(timer_func, arch_dict[args.arch], args.verbose)
    times = time_code_hashes(args.code_hash, timer)

    # mysql = subprocess.Popen(['mysql'], stdin=subprocess.PIPE)
    speeds = [(code_hash, arch_id, kind_id, speed or 0.0) for (code_hash, speed) in list(times.items())]
    # get csv string text from speeds
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerows(speeds)
    values = output.getvalue().strip()
    # if args.insert:
    #     print(('Inserting {}'.format(len(times))))
    #     mysql.communicate('INSERT INTO time (code_hash, arch_id, kind_id, cycle_count) VALUES {};\n'.format(values))
    # else:
    print(values)


if __name__ == '__main__':
    main()
