"""Loopback-only demo and one-at-a-time experiment launcher."""
import argparse
import json
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from .connectome import ROOT

LOCK=threading.Lock()
JOB=None
LOG=None


def status():
    with LOCK:
        if JOB is None:
            return dict(running=False,exit_code=None,log='')
        return dict(running=JOB.poll() is None,exit_code=JOB.poll(),
            log=LOG.read_text(errors='replace')[-6000:] if LOG.exists() else '')


class Handler(BaseHTTPRequestHandler):
    def send(self, code, content, mime='application/json'):
        data=content if isinstance(content,bytes) else json.dumps(content).encode()
        self.send_response(code)
        self.send_header('Content-Type',mime)
        self.send_header('Content-Length',str(len(data)))
        self.send_header('Cache-Control','no-store')
        self.send_header('X-Content-Type-Options','nosniff')
        self.end_headers();self.wfile.write(data)

    def do_GET(self):
        if self.path=='/api/status':
            return self.send(200,status())
        files={'/':'learning.html','/learning.html':'learning.html','/learning-data.json':'learning-data.json'}
        if self.path not in files:
            return self.send(404,{'error':'Not found'})
        path=ROOT/files[self.path]
        if not path.exists():
            return self.send(404,{'error':'Run an experiment first'})
        return self.send(200,path.read_bytes(),'text/html; charset=utf-8' if path.suffix=='.html' else 'application/json')

    def do_POST(self):
        global JOB, LOG
        # Reject cross-origin browser requests and unexpected host headers.
        allowed=f'127.0.0.1:{self.server.server_port}'
        if self.headers.get('Host')!=allowed or self.headers.get('Origin')!=f'http://{allowed}':
            return self.send(403,{'error':'Local same-origin request required'})
        if self.path!='/api/run':
            return self.send(404,{'error':'Not found'})
        try:
            size=int(self.headers.get('Content-Length','0'))
            if size<1 or size>1024:
                raise ValueError('Invalid request size')
            request=json.loads(self.rfile.read(size))
            seed=request['seed']
            if type(seed) is not int or not 0<=seed<=1000000:
                raise ValueError('Seed must be an integer from 0 to 1000000')
        except (ValueError,KeyError,TypeError) as error:
            return self.send(400,{'error':str(error)})
        with LOCK:
            if JOB is not None and JOB.poll() is None:
                return self.send(409,{'error':'An experiment is already running'})
            folder=ROOT/'outputs/brainlab/jobs';folder.mkdir(parents=True,exist_ok=True)
            LOG=folder/f'{time.time_ns()}.log'
            with LOG.open('w') as output:
                JOB=subprocess.Popen([sys.executable,'-m','brainlab.learning','--seed',str(seed)],
                    cwd=ROOT,stdout=output,stderr=subprocess.STDOUT)
        return self.send(202,{'started':True,'seed':seed})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port',type=int,default=8767)
    args=parser.parse_args()
    server=ThreadingHTTPServer(('127.0.0.1',args.port),Handler)
    print(f'http://127.0.0.1:{args.port}',flush=True)
    server.serve_forever()

if __name__=='__main__':
    main()
