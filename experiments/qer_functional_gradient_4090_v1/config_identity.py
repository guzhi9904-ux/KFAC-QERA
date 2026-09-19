"""Compare model configurations in the representation used by saved identities."""
import json


def json_snapshot(value):
    # JSON stores dictionary keys as strings and tuples as arrays. Apply the
    # same conversion to live configurations before comparing or persisting.
    return json.loads(json.dumps(value, allow_nan=False))


def configuration_differences(expected, actual):
    expected = json_snapshot(expected)
    actual = json_snapshot(actual)
    ignored = {'_name_or_path', 'transformers_version', 'torch_dtype', 'use_cache',
               '_attn_implementation_autoset'}
    return {k: (v, actual.get(k)) for k, v in expected.items()
            if k not in ignored and not k.startswith('_attn') and actual.get(k) != v}
