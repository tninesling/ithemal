import os
import subprocess

import common_libs.utilities as ut
import pytorch.data.data_cost as dt

_TOKENIZER = os.path.join(
    os.environ["ITHEMAL_HOME"], "data_collection", "build", "bin", "tokenizer"
)
_fake_intel = "\n" * 500

def datum_of_code(data, block_hex):
    xml = subprocess.check_output([_TOKENIZER, block_hex, "--token"])
    print(f"XML: {xml}")
    data.raw_data = [(-1, -1, _fake_intel, xml)]
    data.data = []
    data.prepare_data(fixed=False, progress=False)
    return data.data[-1]

embedder = dt.DataInstructionEmbedding()

block_hex = '4183ff0119c083e00885c98945c4b8010000000f4fc139c2'
datum = datum_of_code(embedder, block_hex)

datum.block.print_block()

print(f"Raw instrs: {datum.x}")