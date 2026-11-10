import json

from badminton.pipeline import analyze_dataset, model_fingerprints


def test_retrained_model_at_same_path_invalidates_dataset_resume(tmp_path, monkeypatch):
    videos = tmp_path / 'video'
    videos.mkdir()
    (videos / 'one.mp4').write_bytes(b'input-bytes')
    model = tmp_path / 'model.onnx'
    model.write_bytes(b'first-checkpoint')
    called = []

    def analyze(path, output, config):
        from badminton.media import file_hash
        called.append(model.read_bytes())
        return {'video_id': config['video_id'], 'file': 'one.mp4', 'source_sha256': file_hash(path),
                'analysis': {'frame_count': 1}, 'decision': {'decision': 'REVIEW'},
                'calibration': {'status': 'review'}, 'tracks': [], 'landings': [],
                'model_fingerprints': model_fingerprints(config)}

    monkeypatch.setattr('badminton.pipeline.analyze_video', analyze)
    config = {'detector': {'model_path': str(model)}}
    run = tmp_path / 'run'
    analyze_dataset(videos, run, config)
    analyze_dataset(videos, run, config)
    assert called == [b'first-checkpoint']
    model.write_bytes(b'retrained-checkpoint')
    analyze_dataset(videos, run, config)
    assert called == [b'first-checkpoint', b'retrained-checkpoint']
    index = json.loads((run / 'index.json').read_text())
    assert index['status'] == 'completed'


def test_assisted_request_cannot_be_counted_as_automatic_accuracy(tmp_path):
    from badminton.evaluation import evaluate_run
    folder = tmp_path / 'videos/v1'
    folder.mkdir(parents=True)
    result = {'video_id': 'v1', 'source_sha256': 'a', 'decision': {'decision': 'IN'},
              'config': {'request_context': {'camera_id': 'A'}}, 'tracks': [], 'landings': [],
              'metadata': {'duration': 1}, 'analysis': {'width': 160, 'height': 160}}
    (folder / 'result.json').write_text(json.dumps(result))
    (tmp_path / 'index.json').write_text(json.dumps({'videos': [{'id': 'v1', 'status': 'completed',
                                      'result_path': 'videos/v1/result.json'}]}))
    annotations = tmp_path / 'annotations'
    annotations.mkdir()
    (annotations / 'v1.json').write_text(json.dumps({'video_id': 'v1', 'source': 'human_review',
        'status': 'verified', 'reviewer': 'test reviewer', 'label': 'IN', 'source_sha256': 'a',
        'contact_observed': True, 'time': .5, 'contact_px': [50, 50]}))
    evaluation = evaluate_run(tmp_path)
    assert evaluation['decision']['automatic_accuracy'] is None
    assert evaluation['operational']['automatic_coverage'] == 0
    assert evaluation['operational']['assisted_decision_counts'] == {'IN': 1}
    assert evaluation['excluded_ground_truth_counts']['human_assisted_prediction'] == 1
