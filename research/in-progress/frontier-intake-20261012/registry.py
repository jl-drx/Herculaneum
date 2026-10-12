"""Conservative physical-family admission for the new research programme."""
from __future__ import annotations
import re

LEGACY = {'Scroll1': 'PHercParis4', 'Scroll3': 'PHerc0332', 'PHerc332': 'PHerc0332'}


def family(sample):
    sample = LEGACY.get(sample, sample)
    match = re.fullmatch(r'PHerc0*(\d+)(?:[A-Za-z].*)?', sample, flags=re.IGNORECASE)
    if match:
        return 'PHerc' + str(int(match.group(1))).zfill(4)
    if re.fullmatch(r'PHerc(?:Paris[1-4]|MAN[A-Za-z0-9]*)', sample):
        return sample
    raise ValueError(f'Unresolved physical family: {sample}')


def require_admitted(sample, purpose, registry):
    """Check family scope only; this is not full scientific cohort admission."""
    if purpose not in {'metadata', 'development', 'training', 'confirmation'}:
        raise ValueError('Unknown purpose')
    group = family(sample)
    if purpose == 'metadata':
        return group
    if group in registry['protected_running_families']:
        raise ValueError('New payload use of the running-study family is locked')
    if group in registry['reserved_families']:
        raise ValueError('Reserved family: no payload admission before a separate confirmation freeze')
    if purpose == 'confirmation':
        raise ValueError('Confirmation support is not qualified by this intake')
    if group not in registry['development_families']:
        raise ValueError('Unassigned physical family requires explicit admission')
    return group
