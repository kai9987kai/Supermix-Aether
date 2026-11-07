"""Command-line entry points; no external services are required."""

import argparse
from dataclasses import asdict
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(prog='aether', description='SUPERMiX AETHER NeuroWave-X CPU research lab')
    commands = parser.add_subparsers(dest='command', required=True)
    run = commands.add_parser('run', help='train and evaluate a recorded paired study')
    run.add_argument('--config', type=Path, default=Path('configs/pilot.json'))
    run.add_argument('--output', type=Path, required=True, help='new output directory; existing paths are never overwritten')
    run.add_argument('--updates', type=int)
    run.add_argument('--seeds', help='comma-separated independent training seeds')
    replay = commands.add_parser('replay', help='replay recorded held-out actions from a checkpoint')
    replay.add_argument('checkpoint', type=Path)
    verify = commands.add_parser('verify', help='verify artifact hashes in a receipt')
    verify.add_argument('receipt', type=Path)
    serve = commands.add_parser('serve', help='serve a generated observatory on loopback')
    serve.add_argument('directory', type=Path)
    serve.add_argument('--port', type=int, default=8765)
    live = commands.add_parser('live', help='run the live neural execution and adaptation workbench')
    live.add_argument('--study', type=Path, default=Path('reports/pilot-2026-11-07'),
                      help='recorded study containing validated checkpoints')
    live.add_argument('--output-root', type=Path, default=Path('runs/live'),
                      help='separate directory for exploratory adaptation records')
    live.add_argument('--port', type=int, default=8789)
    profile = commands.add_parser('profile', help='profile computational FLOPs, memory, and CPU latency across protocols')
    profile.add_argument('--warmup', type=int, default=5, help='warmup iterations')
    profile.add_argument('--benchmark-steps', type=int, default=25, help='timed benchmark iterations')
    args = parser.parse_args(argv)
    try:
        if args.command == 'run':
            from aether.experiments.study import StudyConfig, run_study
            values = json.loads(args.config.read_text(encoding='utf-8'))
            if args.updates is not None:
                values['updates'] = args.updates
            if args.seeds is not None:
                values['seeds'] = [int(s.strip()) for s in args.seeds.split(',')]
            result = run_study(StudyConfig(**values), args.output, progress=lambda s: print(s, flush=True))
            print(json.dumps({'status': result['status'], 'promotion': result['promotion'],
                              'observatory': str((args.output / 'index.html').resolve())}, indent=2))
        elif args.command == 'replay':
            from aether.experiments.study import replay_checkpoint
            result = replay_checkpoint(args.checkpoint)
            print(json.dumps({'valid': result['valid'], 'state_digest': result['state_digest']}, indent=2))
            return 0 if result['valid'] else 1
        elif args.command == 'verify':
            from aether.evidence.receipts import verify_receipt
            result = verify_receipt(args.receipt)
            print(json.dumps(result, indent=2))
            return 0 if result['valid'] else 1
        elif args.command == 'live':
            if not 1 <= args.port <= 65535:
                raise ValueError('port must be in [1,65535]')
            from aether.live.server import serve_live
            serve_live(args.study, output_root=args.output_root, port=args.port)
        elif args.command == 'profile':
            from aether.config import ModelConfig
            from aether.kernel.runtime import PROTOCOLS, TissueModel
            from aether.kernel.profiler import profile_model, verify_resource_comparability
            config = ModelConfig()
            profiles = {}
            for protocol in PROTOCOLS:
                model = TissueModel(config, protocol=protocol, seed=11)
                profiles[protocol] = profile_model(model, warmup_steps=args.warmup, benchmark_steps=args.benchmark_steps)
            comparability = verify_resource_comparability(profiles['wave'], profiles)
            out = {
                'profiles': {k: v.to_dict() for k, v in profiles.items()},
                'comparability': comparability,
            }
            print(json.dumps(out, indent=2))
            return 0
        else:
            directory = args.directory.resolve(strict=True)
            if not directory.is_dir() or not (directory / 'index.html').is_file():
                raise ValueError('directory must contain a generated index.html')
            if not 1 <= args.port <= 65535:
                raise ValueError('port must be in [1,65535]')
            handler = partial(SimpleHTTPRequestHandler, directory=str(directory))
            server = ThreadingHTTPServer(('127.0.0.1', args.port), handler)
            print(f'Observatory: http://127.0.0.1:{args.port}', flush=True)
            try:
                server.serve_forever()
            except KeyboardInterrupt:
                pass
            finally:
                server.server_close()
    except (ValueError, TypeError, OSError, RuntimeError) as error:
        parser.exit(2, f'aether: {error}\n')
    return 0
