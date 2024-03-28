import helm
import os, shutil

install_path = helm.__file__
source_path = 'helm/src/helm/benchmark/metrics/toxicity_metrics.py'
target_path = '/'.join(install_path.split('/')[:-1])
target_path = os.path.join(target_path, 'benchmark/metrics/toxicity_metrics.py')
shutil.copy(source_path, target_path)