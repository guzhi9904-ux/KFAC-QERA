#!/usr/bin/env python3
"""Write a host-local configuration after verified private asset migration."""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('assets', 'model', 'output-parent', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    args = p.parse_args()
    assets = args.assets.resolve(); model = args.model.resolve(); output_parent = args.output_parent.resolve()
    verified = json.loads((assets/'migration_verified.json').read_text())
    assert verified['passed']
    assert all((assets/name/'identity.json').is_file() for name in ('exp01', 'exp03'))
    assert (assets/'vendor/src/qera/quantize/quantizers/mxint.py').is_file()
    assert (model/'config.json').is_file()
    for protected in (assets, model, Path(__file__).resolve().parent):
        assert not output_parent.is_relative_to(protected)
    config = dict(schema=1, assets=str(assets), model=str(model), output_parent=str(output_parent),
                  migration_manifest_sha256=verified['manifest_sha256'], workers=1, gpus=2)
    with args.output.resolve().open('x', encoding='utf-8') as f:
        json.dump(config, f, indent=2); f.write('\n')
    print('Configuration saved:', args.output.resolve())
    print('No GPU work started. Next step is explicit pilot, not formal.')


if __name__ == '__main__': main()
