"""Prepare checked Windows CPython 3.12 wheels as an optional runtime payload.

This builds an importable library runtime, without pip's console-script wrappers.
The caller supplies the portable builder's archive/path checks to avoid importing
the builder here. ``packaging`` is a build-time dependency, not a runtime one.
"""
from __future__ import annotations

from email.parser import BytesParser
import hashlib
from pathlib import Path, PurePosixPath
import shutil
import zipfile

from packaging.markers import default_environment
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.tags import compatible_tags, cpython_tags, parse_tag
from packaging.utils import canonicalize_name, parse_wheel_filename
from packaging.version import Version


PAYLOAD_NAME = '10-pytorch_cpu.zip'
TORCH_VERSION = '2.10.0+cpu'
PYTHON_VERSION = '3.12.10'
SITE_PACKAGES = 'runtime/Lib/site-packages'


def _sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _reject_symlinks(path):
    for part in (path, *path.parents):
        if part.is_symlink():
            raise ValueError(f'Symlink wheel input/output is not accepted: {part}')


def _destination(name, dist_info, site_packages):
    parts = PurePosixPath(name).parts
    if parts[0].endswith('.data'):
        if parts[0] != dist_info[:-len('.dist-info')] + '.data' or len(parts) < 3:
            raise ValueError(f'Invalid wheel data path: {name}')
        schemes = {'purelib': site_packages, 'platlib': site_packages,
                   'scripts': 'runtime/Scripts', 'data': 'runtime',
                   'headers': 'runtime/Include/' + dist_info.split('-')[0]}
        if parts[1] not in schemes:
            raise ValueError(f'Unsupported wheel data scheme: {name}')
        return schemes[parts[1]] + '/' + '/'.join(parts[2:])
    return site_packages + '/' + name


def prepare_payload(wheel_dir, destination, existing_files, *, bundled_versions,
                    safe_name, inspect_zip, site_packages=SITE_PACKAGES):
    """Write a CPU payload and return wheel provenance; reject incomplete inputs.

    ``existing_files`` contains case-folded archive paths from other payloads and
    is updated only on success. ``bundled_versions`` maps already bundled wheel
    distribution names (e.g. filelock/numpy) to their actual versions. Matching
    wheels in the input directory are checked but not duplicated in the payload.
    """
    wheel_dir, destination = Path(wheel_dir).absolute(), Path(destination).absolute()
    for path in (wheel_dir, destination):
        _reject_symlinks(path)
    if not wheel_dir.is_dir():
        raise ValueError(f'Wheel directory does not exist: {wheel_dir}')
    if destination.exists():
        raise ValueError(f'PyTorch payload already exists: {destination}')
    site_packages = safe_name(site_packages)
    if site_packages != SITE_PACKAGES:
        raise ValueError(f'Unexpected portable site-packages path: {site_packages}')
    bundled = {canonicalize_name(name): Version(str(version))
               for name, version in bundled_versions.items()}
    if 'torch' in bundled:
        raise ValueError('Base runtime must not already bundle PyTorch')
    target_tags = set(cpython_tags((3, 12), abis=['cp312'], platforms=['win_amd64']))
    target_tags.update(compatible_tags((3, 12), interpreter='cp312', platforms=['win_amd64']))
    environment = default_environment()
    environment.update(implementation_name='cpython', implementation_version=PYTHON_VERSION,
                       os_name='nt', platform_machine='AMD64', platform_python_implementation='CPython',
                       platform_release='', platform_system='Windows', platform_version='',
                       python_full_version=PYTHON_VERSION, python_version='3.12', sys_platform='win32', extra='')
    wheels = {}
    for path in sorted(wheel_dir.glob('*.whl')):
        _reject_symlinks(path)
        name, version, _, tags = parse_wheel_filename(path.name)
        if not tags & target_tags:
            raise ValueError(f'Wheel is incompatible with Windows CPython 3.12 x64: {path.name}')
        if name in wheels:
            raise ValueError(f'Duplicate wheel distribution: {name}')
        inspect_zip(path)
        with zipfile.ZipFile(path) as archive:
            names = [entry.filename for entry in archive.infolist() if not entry.is_dir()]
            metadata_names = [item for item in names if item.count('/') == 1 and item.endswith('.dist-info/METADATA')]
            if len(metadata_names) != 1:
                raise ValueError(f'Expected one wheel METADATA: {path.name}')
            metadata_name = metadata_names[0]
            dist_info = metadata_name.split('/')[0]
            metadata = BytesParser().parsebytes(archive.read(metadata_name))
            if canonicalize_name(metadata.get('Name', '')) != name or Version(metadata.get('Version', '0')) != version:
                raise ValueError(f'Wheel filename/METADATA mismatch: {path.name}')
            expected_dist_info = name.replace('-', '_') + '-' + str(version) + '.dist-info'
            if dist_info.casefold() != expected_dist_info.casefold():
                raise ValueError(f'Wheel dist-info directory mismatch: {path.name}')
            wheel_metadata = BytesParser().parsebytes(archive.read(dist_info + '/WHEEL'))
            declared_tags = set()
            for value in wheel_metadata.get_all('Tag', []):
                declared_tags.update(parse_tag(value))
            if declared_tags != tags:
                raise ValueError(f'Wheel filename/WHEEL tags mismatch: {path.name}')
            python_requirement = metadata.get('Requires-Python')
            if python_requirement and not SpecifierSet(python_requirement).contains(PYTHON_VERSION):
                raise ValueError(f'Wheel requires an incompatible Python: {path.name}')
            requirements = [Requirement(value) for value in metadata.get_all('Requires-Dist', [])]
            licenses = [item for item in names
                        if PurePosixPath(item).name.lower().startswith(('license', 'copying', 'notice'))]
            if not licenses:
                raise ValueError(f'Wheel has no bundled license: {path.name}')
            wheels[name] = {'path': path, 'version': version, 'dist_info': dist_info,
                            'requirements': requirements, 'files': names, 'licenses': licenses,
                            'sha256': _sha256(path)}
    if 'torch' not in wheels or wheels['torch']['version'] != Version(TORCH_VERSION):
        raise ValueError(f'Exactly torch=={TORCH_VERSION} is required')
    # Traverse only requirements active on the target, not the build host. This
    # catches Windows-only and Python 3.12 dependencies such as setuptools.
    selected, pending = set(), ['torch']
    reused = set()
    while pending:
        name = pending.pop()
        if name in selected:
            continue
        selected.add(name)
        for requirement in wheels[name]['requirements']:
            if requirement.marker and not requirement.marker.evaluate(environment):
                continue
            dependency = canonicalize_name(requirement.name)
            if requirement.url or requirement.extras:
                raise ValueError(f'Unsupported wheel requirement: {requirement}')
            if dependency in bundled:
                available = bundled[dependency]
                reused.add(dependency)
            elif dependency in wheels:
                available = wheels[dependency]['version']
                pending.append(dependency)
            else:
                raise ValueError(f'Missing PyTorch wheel dependency: {requirement}')
            if not requirement.specifier.contains(available, prereleases=True):
                raise ValueError(f'Incompatible dependency {requirement}: found {available}')
    extra = set(wheels) - selected - set(bundled)
    if extra:
        raise ValueError('Unrelated wheels in PyTorch input: ' + ', '.join(sorted(extra)))
    occupied = {safe_name(name).casefold() for name in existing_files}
    directories = {parent.as_posix() for name in occupied
                   for parent in PurePosixPath(name).parents if parent.as_posix() != '.'}
    plan, provenance = [], []
    for name in sorted(selected):
        wheel = wheels[name]
        mapped = {}
        for member in wheel['files']:
            target = safe_name(_destination(safe_name(member), wheel['dist_info'], site_packages))
            key = target.casefold()
            ancestors = {parent.as_posix() for parent in PurePosixPath(key).parents
                         if parent.as_posix() != '.'}
            if key in occupied or key in directories or ancestors & occupied:
                raise ValueError(f'File collision in PyTorch payload: {target}')
            occupied.add(key)
            directories.update(ancestors)
            mapped[member] = target
        plan.append((wheel, mapped))
        provenance.append({'file': wheel['path'].name, 'name': name, 'version': str(wheel['version']),
                           'sha256': wheel['sha256'], 'bytes': wheel['path'].stat().st_size,
                           'licenses': [mapped[item] for item in wheel['licenses']]})
    destination.parent.mkdir(parents=True, exist_ok=True)
    created = False
    try:
        with destination.open('xb') as stream:
            created = True
            with zipfile.ZipFile(stream, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as output:
                for wheel, mapped in plan:
                    if _sha256(wheel['path']) != wheel['sha256']:
                        raise ValueError(f'Wheel changed during packaging: {wheel["path"].name}')
                    with zipfile.ZipFile(wheel['path']) as source:
                        for member, target in mapped.items():
                            # Stream the large native DLLs; never extract wheel paths
                            # onto the builder's filesystem or load their Python code.
                            with source.open(member) as source_file, output.open(target, 'w', force_zip64=True) as target_file:
                                shutil.copyfileobj(source_file, target_file, length=1024 * 1024)
        inspect_zip(destination)
    except Exception:
        if created:
            destination.unlink(missing_ok=True)
        raise
    existing_files.update(occupied)
    return {'payload': PAYLOAD_NAME, 'torch_version': TORCH_VERSION,
            'python_tag': 'cp312', 'python_version': PYTHON_VERSION,
            'platform': 'win_amd64', 'site_packages': site_packages,
            'wheels': provenance, 'reused_dependencies': {name: str(bundled[name]) for name in sorted(reused)},
            'console_script_wrappers_generated': False}
