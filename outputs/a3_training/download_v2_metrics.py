"""Download completed v2 metrics, predictions, plots and adapters; skip checkpoints."""
from pathlib import Path
import modal

RUNS = ['a3_qwen3_17b_20261109_122845_028798_50aa903b',
        'a3_vinallama_27b_20261109_123046_192478_2df1dfe0',
        'a3_mediphi_38b_20261109_122937_227148_1d0bf5fc']
volume = modal.Volume.from_name('dental-a3-training')
root = Path(__file__).resolve().parent
for run in RUNS[1:]:
    prefix = 'runs/' + run + '/'
    count = 0
    for entry in volume.iterdir('runs/' + run, recursive=True):
        relative = entry.path.removeprefix('/').removeprefix(prefix)
        if relative.startswith(('checkpoints/', 'adapter/')) or not Path(relative).suffix:
            continue
        dest = root / run / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open('wb') as handle:
            for block in volume.read_file(entry.path):
                handle.write(block)
        count += 1
    print(run, count, 'files downloaded', flush=True)

