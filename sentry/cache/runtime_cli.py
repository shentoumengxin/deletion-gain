"""Run the offline fixture or an explicitly configured local-model/API runtime."""
from dataclasses import asdict
import argparse
import json
import os
from pathlib import Path
import tempfile
import urllib.request

import numpy as np

from .runtime import RuntimeConfig, create_runtime
from .defense.fence import ExcessFence

class DemoEmbedder:
    """Deliberately constant geometry for lifecycle demonstrations, not evaluation."""
    model_name = 'sentry-lifecycle-fixture'
    dimension = 3
    signature = {'model_name':model_name, 'revision':'fixture-v1', 'pooling':'synthetic',
                 'text_prefix':'', 'normalization':'l2', 'dimension':3}
    def encode(self, texts): return np.tile([1.,0.,0.], (len(texts),1))


def run_demo():
    with tempfile.TemporaryDirectory(prefix='sentry-demo-') as directory:
        root=Path(directory); embedder=DemoEmbedder()
        config=RuntimeConfig(root/'fixture-fence.json', embedder.signature, root/'cache')
        # These are explicit fixture constants, never real calibration thresholds.
        fence=ExcessFence(np.array([.01,0.,0.]), .05, embedder=embedder.model_name,
            policy=config.policy.fingerprint(), answer_rule='either', eta_a=.01,
            metadata={'joint_calibrated':True, 'encoder_signature':embedder.signature,
                      'fixture_only':True, 'provenance':'synthetic lifecycle constants; no detection claims'})
        fence.save(config.fence_path)
        prompt='what color is the clear sky during daylight'
        calls=[]
        def backend(query): calls.append(query); return 'Blue.'
        print('Synthetic lifecycle fixture; thresholds and embeddings are not an empirical calibration.')
        with create_runtime(config,embedder,backend) as runtime:
            print(json.dumps(asdict(runtime.ask(prompt))))
            print(json.dumps(asdict(runtime.ask(prompt))))
        with create_runtime(config,embedder,backend) as runtime:
            print(json.dumps(asdict(runtime.ask(prompt))))
        print(json.dumps({'backend_calls':len(calls), 'expected':1}))


def _api_backend(config):
    key=os.environ.get(config.get('api_key_env','OPENAI_API_KEY'))
    if not key: raise ValueError('API key environment variable is not set')
    url=config['url']; model=config['model']
    def call(query):
        payload=json.dumps({'model':model,'messages':[{'role':'user','content':query}]}).encode()
        request=urllib.request.Request(url, data=payload,
            headers={'Content-Type':'application/json','Authorization':'Bearer '+key})
        with urllib.request.urlopen(request,timeout=config.get('timeout_seconds',60)) as response:
            value=json.load(response)['choices'][0]['message']['content']
        if not isinstance(value,str): raise TypeError('API returned a non-text answer')
        return value
    return call


def _load_embedder(args):
    from .quickstart import DEFAULT_MODEL, DEFAULT_REVISION
    from sentry.embeddings import TransformerCLSEmbedder
    revision = args.revision or (DEFAULT_REVISION if args.model == DEFAULT_MODEL else None)
    return TransformerCLSEmbedder(args.model, pooling='cls', revision=revision,
                                  source=args.model_path)


def _add_encoder_args(parser):
    from .quickstart import DEFAULT_MODEL
    parser.add_argument('--model', default=DEFAULT_MODEL, help='Hugging Face model id')
    parser.add_argument('--revision', default=None, help='model revision (pinned for the default model)')
    parser.add_argument('--model-path', default=None, help='load the model from a local copy')


def run_calibrate(args):
    from .defense.calibrate import calibrate_from_hits
    hits=[json.loads(line) for line in args.hits.read_text(encoding='utf-8').splitlines() if line.strip()]
    fence, report = calibrate_from_hits(hits, _load_embedder(args), budget=args.budget,
                                        min_cosine=args.min_cosine)
    fence.save(args.output)
    print(json.dumps(report, indent=2))
    print(f'fence written to {args.output}')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    subs=parser.add_subparsers(dest='command',required=True)
    demo=subs.add_parser('demo',help='offline GPTCache lifecycle fixture; --real runs the defense on e5')
    demo.add_argument('--real',action='store_true',help='block a real poisoned entry with e5-small-v2')
    demo.add_argument('--model-path',default=None,help='local copy of e5-small-v2')
    calibrate=subs.add_parser('calibrate',help='fit a fence from your benign cache hits (JSONL: query, key, answer)')
    calibrate.add_argument('--hits',type=Path,required=True)
    calibrate.add_argument('--output',type=Path,required=True)
    calibrate.add_argument('--budget',type=float,default=0.05,help='share of benign hits the fence may reject')
    calibrate.add_argument('--min-cosine',type=float,default=0.90,help='the cache retrieval threshold')
    _add_encoder_args(calibrate)
    ask=subs.add_parser('ask',help='local encoder with an explicit API backend')
    ask.add_argument('--config',type=Path,required=True)
    ask.add_argument('query')
    export=subs.add_parser('export-fence',help='export held-out-fit joint either calibration')
    export.add_argument('--report',type=Path,required=True)
    export.add_argument('--signature',type=Path,required=True)
    export.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    if args.command=='demo':
        if args.real:
            from .demo import run_real_demo
            return run_real_demo(args.model_path)
        return run_demo()
    if args.command=='calibrate': return run_calibrate(args)
    if args.command=='export-fence':
        from .defense.calibrate import export_runtime_fence
        export_runtime_fence(json.loads(args.report.read_text()),args.output,json.loads(args.signature.read_text()))
        return
    from sentry.embeddings import TransformerCLSEmbedder
    value=json.loads(args.config.read_text())
    embedder=TransformerCLSEmbedder(**value['embedder'])
    signature=value['encoder_signature']
    config=RuntimeConfig(fence_path=value['fence_path'],encoder_signature=signature,data_dir=value['data_dir'])
    with create_runtime(config,embedder,_api_backend(value['backend'])) as runtime:
        print(json.dumps(asdict(runtime.ask(args.query)),ensure_ascii=False))

if __name__=='__main__': main()
