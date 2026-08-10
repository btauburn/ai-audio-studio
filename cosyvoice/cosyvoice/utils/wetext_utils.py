# Copyright 2026 TadashiJLU
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Offline WeText helpers for the vendored CosyVoice runtime."""

import os
from pathlib import Path


def load_project_wetext_normalizers(model_dir: str):
    """Load WeText TN graphs from explicit ordinary files.

    Passing both graph paths is important: wetext otherwise calls
    ModelScope's snapshot_download during frontend construction.
    """
    root = Path(model_dir).expanduser().resolve()
    paths = {
        'en_tagger': root / 'en' / 'tn' / 'tagger.fst',
        'en_verbalizer': root / 'en' / 'tn' / 'verbalizer.fst',
        'zh_tagger': root / 'zh' / 'tn' / 'tagger.fst',
        'zh_verbalizer': root / 'zh' / 'tn' / 'verbalizer.fst',
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError('missing project-local WeText FST: {}'.format(', '.join(missing)))

    from wetext import Normalizer
    zh_normalizer = Normalizer(
        tagger_path=str(paths['zh_tagger']),
        verbalizer_path=str(paths['zh_verbalizer']),
        lang='zh',
        operator='tn',
        remove_erhua=False,
    )
    en_normalizer = Normalizer(
        tagger_path=str(paths['en_tagger']),
        verbalizer_path=str(paths['en_verbalizer']),
        lang='en',
        operator='tn',
    )
    return zh_normalizer, en_normalizer


def load_wetext_from_environment():
    model_dir = os.environ.get('COSYVOICE_WETEXT_DIR')
    if not model_dir:
        raise RuntimeError('COSYVOICE_WETEXT_DIR is required for offline WeText')
    return load_project_wetext_normalizers(model_dir)
