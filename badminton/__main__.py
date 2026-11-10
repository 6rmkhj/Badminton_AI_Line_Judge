import argparse
import json
from pathlib import Path
from .pipeline import analyze_video, analyze_dataset, write_json


def main():
    parser = argparse.ArgumentParser(description='Badminton offline analysis and human review')
    sub = parser.add_subparsers(dest='command', required=True)
    batch = sub.add_parser('analyze')
    batch.add_argument('input', type=Path)
    batch.add_argument('--output', type=Path, default=Path('runs/latest'))
    batch.add_argument('--width', type=int, default=1280)
    batch.add_argument('--config', type=Path)
    batch.add_argument('--limit', type=int)
    batch.add_argument('--no-resume', action='store_true')
    server = sub.add_parser('serve')
    server.add_argument('--run', type=Path, default=Path('runs/latest'))
    server.add_argument('--port', type=int, default=8765)
    server.add_argument('--host', default='127.0.0.1')
    evaluation = sub.add_parser('evaluate')
    evaluation.add_argument('--run', type=Path, default=Path('runs/latest'))
    evaluation.add_argument('--annotations', type=Path)
    training = sub.add_parser('train')
    training.add_argument('--manifest', type=Path, required=True)
    training.add_argument('--annotations', type=Path, required=True)
    training.add_argument('--video-dir', type=Path, default=Path('video'))
    training.add_argument('--output', type=Path, default=Path('models/candidate-scorer.pt'))
    training.add_argument('--epochs', type=int, default=20)
    training.add_argument('--readiness-only', action='store_true')
    contact = sub.add_parser('train-contact')
    contact.add_argument('--manifest', type=Path, required=True)
    contact.add_argument('--annotations', type=Path, required=True)
    contact.add_argument('--video-dir', type=Path, default=Path('video'))
    contact.add_argument('--output', type=Path, default=Path('models/contact-model.pt'))
    contact.add_argument('--epochs', type=int, default=20)
    contact.add_argument('--readiness-only', action='store_true')
    args = parser.parse_args()
    if args.command == 'analyze':
        config = json.loads(args.config.read_text()) if args.config else {}
        config.setdefault('analysis_width', args.width)
        if args.input.is_dir():
            result = analyze_dataset(args.input, args.output, config, args.limit, not args.no_resume)
            return int(result['status'] != 'completed')
        result = analyze_video(args.input, args.output, config)
        print(json.dumps(result['decision'], ensure_ascii=False))
    elif args.command == 'serve':
        from .web import serve
        serve(args.run, args.host, args.port)
    elif args.command == 'evaluate':
        from .evaluation import evaluate_run
        result = evaluate_run(args.run, args.annotations)
        write_json(args.run / 'evaluation.json', result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.command == 'train':
        from .learning import training_readiness, train_candidate_scorer
        manifest = json.loads(args.manifest.read_text())
        if args.readiness_only:
            result = training_readiness(manifest, args.annotations)
        else:
            result = train_candidate_scorer(manifest, args.annotations, args.video_dir, args.output, epochs=args.epochs)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.command == 'train-contact':
        from .contact_model import contact_training_readiness, train_contact_model
        manifest = json.loads(args.manifest.read_text())
        if args.readiness_only:
            result = contact_training_readiness(manifest, args.annotations)
        else:
            result = train_contact_model(manifest, args.annotations, args.video_dir, args.output, epochs=args.epochs)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
