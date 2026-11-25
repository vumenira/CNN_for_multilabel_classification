set -e
BRANCH="${1:-scoremax}"
mkdir -p /content/data /content/out /content/logs
if [ ! -d /content/repo ]; then
  git clone -q -b "$BRANCH" https://github.com/vumenira/CNN_for_multilabel_classification.git /content/repo
else
  git -C /content/repo fetch -q origin "$BRANCH" && git -C /content/repo reset -q --hard "origin/$BRANCH"
fi
python -c "import timm" 2>/dev/null || pip install -q timm
pip install -q -U timm kaggle >/dev/null 2>&1 || true
if [ ! -f /content/data/test.csv ]; then
  cd /content/data
  kaggle competitions download -c cats-and-dogs -p /content/data
  unzip -q -o cats-and-dogs.zip -d /content/data && rm cats-and-dogs.zip
  if [ -d /content/data/cats-and-dogs ] && [ ! -d /content/data/train ]; then mv /content/data/cats-and-dogs/* /content/data/; fi
fi
cd /content/repo
DATA_DIR=/content/data python -c "
from scoremax.common import build_cache
for s in ('train', 'test'):
    print(build_cache(s), flush=True)
"
python -c "import torch, timm; print('torch', torch.__version__, 'timm', timm.__version__, torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
nproc; free -g | head -2; df -h /content | tail -1
echo SETUP_DONE
