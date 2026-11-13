"""Device contracts; fake runtimes verify routing, not physical GPU execution."""
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
import sys

import pytest

from badminton import inference


class FakeTensor:
    def __init__(self, device, value=36., read_error=None):
        self.device = SimpleNamespace(type=device.split(':')[0], index=int(device.split(':')[1]) if ':' in device else None)
        self.value, self.read_error = value, read_error

    def sum(self):
        return self

    def cpu(self):
        if self.read_error:
            raise self.read_error
        return self

    def __float__(self):
        return float(self.value)


def fake_torch(monkeypatch, *, cuda_count=2, mps=False, failure=None):
    """Small observable driver boundary, never a substitute for hardware tests."""
    calls = []

    def ones(shape, *, dtype, device):
        calls.append(('allocate', device, shape))
        if failure:
            raise failure
        return FakeTensor(device)

    def convolution(sample, kernel):
        calls.append(('convolution', sample.device.type, sample.device.index))
        return sample

    @contextmanager
    def device_context(index):
        calls.append(('device_enter', index))
        yield
        calls.append(('device_exit', index))

    runtime = SimpleNamespace(
        float32='float32', inference_mode=nullcontext, ones=ones,
        nn=SimpleNamespace(functional=SimpleNamespace(conv2d=convolution)),
        version=SimpleNamespace(hip=None),
        cuda=SimpleNamespace(
            is_available=lambda: cuda_count > 0, device_count=lambda: cuda_count,
            device=device_context,
            get_device_properties=lambda index: SimpleNamespace(name=f'Runtime GPU {index}', total_memory=8 * 1024**3),
            empty_cache=lambda: calls.append(('empty_cuda',)),
            set_per_process_memory_fraction=lambda fraction, device: calls.append(('fraction', fraction, device))),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: mps)),
        mps=SimpleNamespace(empty_cache=lambda: calls.append(('empty_mps',))))
    monkeypatch.setattr(inference, '_torch', lambda: runtime)
    monkeypatch.setitem(sys.modules, 'torch', runtime)
    return runtime, calls


def test_cpu_does_not_import_optional_torch(monkeypatch):
    monkeypatch.delitem(sys.modules, 'torch', raising=False)
    monkeypatch.delitem(sys.modules, 'badminton.contact_model', raising=False)
    monkeypatch.setattr(inference, '_torch', lambda: pytest.fail('CPU must work without optional PyTorch'))
    assert inference.resolve_device('cpu', verify=True) == 'cpu'
    assert inference.configure_device('cpu') == 'cpu'


@pytest.mark.parametrize('value', [None, True, 0, '', 'cuda', 'cuda:-1', 'cuda:00',
                                  'cuda:1 ', 'cuda:1;echo x', 'mps:0', 'CUDA:0', {}, []])
def test_invalid_device_ids_are_rejected_without_loading_torch(monkeypatch, value):
    monkeypatch.setattr(inference, '_torch', lambda: pytest.fail('Malformed IDs should not load GPU runtime'))
    with pytest.raises(inference.InferenceDeviceError, match='cpu, cuda:'):
        inference.resolve_device(value)


def test_missing_pytorch_is_actionable_and_inventory_remains_available(monkeypatch):
    def missing(_):
        raise ImportError('torch absent')
    monkeypatch.setattr(inference.importlib, 'import_module', missing)
    with pytest.raises(inference.InferenceDeviceError, match='GPU 환경 설치'):
        inference.resolve_device('cuda:0')
    assert inference.available_torch_devices() == []


@pytest.mark.parametrize('count,selected', [(0, 'cuda:0'), (1, 'cuda:1'), (2, 'cuda:99')])
def test_unavailable_or_out_of_range_cuda_is_rejected(monkeypatch, count, selected):
    _, calls = fake_torch(monkeypatch, cuda_count=count)
    with pytest.raises(inference.InferenceDeviceError, match=selected):
        inference.resolve_device(selected, verify=True)
    assert not calls


def test_runtime_visible_index_is_not_reinterpreted_as_physical_index(monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '7,3')
    _, calls = fake_torch(monkeypatch)
    assert inference.resolve_device('cuda:1', verify=True) == 'cuda:1'
    allocations = [call[1] for call in calls if call[0] == 'allocate']
    assert allocations == ['cuda:1', 'cuda:1']
    assert ('convolution', 'cuda', 1) in calls
    assert [row['id'] for row in inference.available_torch_devices()] == ['cuda:0', 'cuda:1']


def test_smoke_failure_never_allocates_a_cpu_replacement(monkeypatch):
    _, calls = fake_torch(monkeypatch, failure=RuntimeError('CUDA driver lost'))
    with pytest.raises(inference.InferenceDeviceError, match='CPU로 자동 전환하지 않았습니다'):
        inference.resolve_device('cuda:1', verify=True)
    assert [call[1] for call in calls if call[0] == 'allocate'] == ['cuda:1']


def test_result_copy_failure_is_treated_as_gpu_failure(monkeypatch):
    runtime, _ = fake_torch(monkeypatch)
    runtime.nn.functional.conv2d = lambda *args: FakeTensor('cuda:0', read_error=RuntimeError('asynchronous CUDA error'))
    with pytest.raises(inference.InferenceDeviceError, match='asynchronous CUDA error'):
        inference.resolve_device('cuda:0', verify=True)


@pytest.mark.parametrize('result', [FakeTensor('cpu'), FakeTensor('cuda:0', value=0)])
def test_smoke_requires_gpu_result_and_correct_observed_value(monkeypatch, result):
    runtime, _ = fake_torch(monkeypatch)
    runtime.nn.functional.conv2d = lambda *args: result
    with pytest.raises(inference.InferenceDeviceError, match='확인 결과'):
        inference.resolve_device('cuda:0', verify=True)


def test_inventory_keeps_detected_but_failed_device_unselectable(monkeypatch):
    fake_torch(monkeypatch, cuda_count=1, failure=RuntimeError('kernel image unavailable'))
    rows = inference.available_torch_devices()
    assert len(rows) == 1
    assert rows[0]['id'] == 'cuda:0'
    assert rows[0]['name'] == 'Runtime GPU 0'
    assert rows[0]['memory_mb'] == 8192
    assert rows[0]['selectable'] is False
    assert 'kernel image unavailable' in rows[0]['reason']


def test_inventory_reports_property_query_failure_as_disabled_device(monkeypatch):
    runtime, _ = fake_torch(monkeypatch, cuda_count=1)
    def broken(_):
        raise RuntimeError('device disconnected')
    runtime.cuda.get_device_properties = broken
    rows = inference.available_torch_devices()
    assert rows[0]['id'] == 'cuda:0'
    assert rows[0]['memory_mb'] is None
    assert rows[0]['selectable'] is False
    assert 'device disconnected' in rows[0]['reason']


def test_memory_fraction_targets_requested_device_and_releases_cached_models(monkeypatch):
    _, calls = fake_torch(monkeypatch)
    monkeypatch.setitem(sys.modules, 'badminton.contact_model', SimpleNamespace(
        clear_contact_model_cache=lambda: calls.append(('clear_models',))))
    assert inference.configure_device('cuda:1', gpu_memory_percent=30) == 'cuda:1'
    assert ('fraction', .3, 1) in calls
    assert calls.index(('clear_models',)) < calls.index(('empty_cuda',)) < calls.index(('fraction', .3, 1))
    assert calls.index(('device_enter', 1)) < calls.index(('empty_cuda',)) < calls.index(('device_exit', 1))


@pytest.mark.parametrize('value', [9, 91, 50.5, True, '50', None])
def test_invalid_allocator_budget_does_not_change_existing_limit(monkeypatch, value):
    _, calls = fake_torch(monkeypatch)
    with pytest.raises(inference.InferenceDeviceError, match='10~90'):
        inference.configure_device('cuda:0', gpu_memory_percent=value)
    assert not any(call[0] in ('fraction', 'empty_cuda') for call in calls)


def test_allocator_failure_is_actionable_and_not_a_successful_cpu_fallback(monkeypatch):
    runtime, _ = fake_torch(monkeypatch)
    def failure(*args):
        raise RuntimeError('CUDA out of memory')
    runtime.cuda.set_per_process_memory_fraction = failure
    with pytest.raises(inference.InferenceDeviceError, match='GPU 메모리 예산'):
        inference.configure_device('cuda:1', gpu_memory_percent=30)


def test_mps_auto_cpu_fallback_must_be_disabled(monkeypatch):
    _, calls = fake_torch(monkeypatch, cuda_count=0, mps=True)
    monkeypatch.setenv('PYTORCH_ENABLE_MPS_FALLBACK', '1')
    with pytest.raises(inference.InferenceDeviceError, match='PYTORCH_ENABLE_MPS_FALLBACK=0'):
        inference.resolve_device('mps', verify=True)
    assert not calls
    rows = inference.available_torch_devices()
    assert rows[0]['id'] == 'mps'
    assert rows[0]['selectable'] is False


def test_mps_uses_shared_memory_without_cuda_fraction(monkeypatch):
    _, calls = fake_torch(monkeypatch, cuda_count=0, mps=True)
    monkeypatch.setenv('PYTORCH_ENABLE_MPS_FALLBACK', '0')
    assert inference.configure_device('mps') == 'mps'
    assert ('convolution', 'mps', None) in calls
    assert ('empty_mps',) in calls
    assert not any(call[0] == 'fraction' for call in calls)


def test_cpu_selection_survives_unavailable_gpu_cleanup(monkeypatch):
    runtime, calls = fake_torch(monkeypatch)
    def failure():
        raise RuntimeError('GPU was disconnected')
    runtime.cuda.empty_cache = failure
    assert inference.configure_device('cpu') == 'cpu'
    assert not any(call[0] == 'fraction' for call in calls)


def test_physical_gpu_smoke_when_runtime_and_hardware_are_present():
    """This is the only physical-GPU test here; CPU hosts explicitly skip it."""
    torch = pytest.importorskip('torch')
    if torch.cuda.is_available() and torch.cuda.device_count():
        selected = 'cuda:0'
    elif getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available():
        selected = 'mps'
    else:
        pytest.skip('No physical CUDA/MPS GPU in this execution environment')
    assert inference.resolve_device(selected, verify=True) == selected
