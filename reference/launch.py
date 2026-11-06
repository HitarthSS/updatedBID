"""Finite smoke -> final manager; no retries, parameter changes or paid inference."""
from pathlib import Path
import subprocess,json,time,os,sys,fcntl,traceback
ROOT=Path(__file__).resolve().parent

def write(name,value):
    p=ROOT/name;t=p.with_suffix(p.suffix+'.tmp')
    t.write_text(json.dumps(value,indent=2)+'\n');t.replace(p)

def main():
    lock=(ROOT/'supervisor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (ROOT/'SUPERVISOR_START.json').exists(),'Refuse a duplicate launch'
    write('SUPERVISOR_START.json',dict(pid=os.getpid(),time=time.time(),gpu=os.environ['CUDA_VISIBLE_DEVICES'],planned_pairs=120,planned_episodes=240))
    for stage,args in [('smoke',[]),('evaluation',['--evaluate'])]:
        assert not (ROOT/'LAST_ERROR.json').exists()
        if stage=='evaluation':assert json.loads((ROOT/'READINESS.json').read_text())['status']=='PASS'
        with (ROOT/(stage+'.log')).open('a') as log:
            p=subprocess.Popen([sys.executable,'-u',str(ROOT/'run.py'),*args],cwd=ROOT,
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        write('CHILD.json',dict(stage=stage,pid=p.pid,supervisor_pid=os.getpid(),time=time.time()))
        code=p.wait()
        if code:raise RuntimeError(stage+' process failed, exit '+str(code)+'; no replacement')
    write('SUPERVISOR_COMPLETE.json',dict(time=time.time(),pid=os.getpid(),completed_pairs=120,physical_episodes=240))

if __name__=='__main__':
    try:main()
    except BaseException:
        write('SUPERVISOR_ERROR.json',dict(time=time.time(),error=traceback.format_exc()));raise
