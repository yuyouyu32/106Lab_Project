# coding=utf-8
import multiprocessing
import os
import shutil
import signal
import subprocess
import tarfile
import time
import json

from flask import Flask, request, send_from_directory
from flask_cors import CORS

from util import XMLNode, XMLUtil, RedisPool, MD5Util, CMDUtil

PLATFORM = 'windows'
MODNAME = 'RemoteDataCollection'
WORKDIR = 'D:\\RemoteDataCollection\\workdir'
EXECUTE = 'RDC_API.py'
COMMAND = 'python RDC.py {task_id}'
PARALLELISM = 4
PORT = 8891

app = Flask(__name__)
CORS(app)

pending = multiprocessing.Queue()
running = multiprocessing.Queue()


@app.route('/hello')
def hello():
    return f'hello, this is {MODNAME}'


@app.route('/submit', methods=['POST'])
def submit():
    task_id = request.values.get('taskId')
    if task_id is None:
        return '{"code": -1, "msg": "taskId is None"}'

    # make workdir
    if os.path.exists(os.path.join(WORKDIR, task_id)):
        return '{"code": -1, "msg": "duplicated taskId"}'
    os.mkdir(os.path.join(WORKDIR, task_id))

    # create result.xml
    with open(os.path.join(WORKDIR, 'result.xml'), 'r') as fp:
        result_xml = fp.read()
    with open(os.path.join(WORKDIR, task_id, 'result.xml'), 'w') as fp:
        for key in request.values:
            result_xml = result_xml.replace(f'{{{key}}}', request.values.get(key))
        for key in request.files:
            result_xml = result_xml.replace(f'{{{key}}}', request.files.get(key).filename)
            # save files
            request.files.get(key).save(os.path.join(WORKDIR, task_id, request.files.get(key).filename))
        fp.write(result_xml)

    # create parameters.txt
    # with open(os.path.join(WORKDIR, task_id, 'parameters.txt'), 'w') as fp:
    #     for node in XMLUtil.parse(os.path.join(WORKDIR, task_id, 'result.xml')).children()[0].children():
    #         if node.type() == XMLNode.TEXT_NODE:
    #             fp.write(f'{node.name()}:{node.data()}\n')
    #         else:
    #             if node.children()[0].name() == 'file':
    #                 fp.write(f'{node.name()}:{node.children()[0].children()[0].data()}\n')
    #             else:
    #                 fp.write(f'{node.name()}:{node.children()[0].data()}\n')

    # create parameters.json (modify by yuyouyu 2021.10.7)
    with open(os.path.join(WORKDIR, task_id, 'parameters.json'), 'w') as fp:
        parameters = {}
        for node in XMLUtil.parse(os.path.join(WORKDIR, task_id, 'result.xml')).children()[0].children():
            if node.type() == XMLNode.TEXT_NODE:
                parameters[node.name()] = node.data()
            else:
                if node.children()[0].name() == 'file':
                    parameters[node.name()] = node.children()[0].children()[0].data()
                else:
                    parameters[node.name()] = node.children()[0].data()
        json.dump(parameters, fp)
        del parameters

    # copy executable file
    shutil.copy(os.path.join(WORKDIR, EXECUTE), os.path.join(WORKDIR, task_id, EXECUTE))

    # put into pending queue
    pending.put(task_id)
    with open(os.path.join(WORKDIR, task_id, 'log.txt'), 'w') as fp:
        fp.write('waiting\n')

    return '{"code": 0}'


@app.route('/state', methods=['GET'])
def state():
    task_id = request.values.get('taskId')
    if task_id is None:
        return '{"code": -1, "msg": "taskId is None"}'
    if not os.path.exists(os.path.join(WORKDIR, task_id)):
        return '{"code": -1, "msg": "workdir not found"}'

    return f'{{"code": 0, "state": "{state_helper(task_id)}"}}'


@app.route('/result', methods=['GET'])
def result():
    task_id = request.values.get('taskId')
    if task_id is None:
        return '{"code": -1, "msg": "taskId is None"}'

    # for second and more requests
    if os.path.exists(os.path.join(WORKDIR, f'{task_id}.tar.gz')):
        return send_from_directory(WORKDIR, f'{task_id}.tar.gz', as_attachment=True)
    # for the first request
    else:
        # check task status
        if not os.path.exists(os.path.join(WORKDIR, task_id)):
            return '{"code": -1, "msg": "workdir not found"}'
        if state_helper(task_id) not in {'finish', 'error'}:
            return '{"code": -1, "msg": "task has not finished"}'

        # compress workdir
        with open(os.path.join(WORKDIR, task_id, 'result.xml'), 'r') as fp:
            result_xml = fp.read()
        with tarfile.open(os.path.join(WORKDIR, f'{task_id}.tar.gz'), 'w:gz') as tar:
            for filename in os.listdir(os.path.join(WORKDIR, task_id)):
                if filename in ['result.xml', 'parameters.txt', EXECUTE, 'log.txt']:
                    continue
                # change filename with its md5 value
                md5_value = MD5Util.get_md5(os.path.join(WORKDIR, task_id, filename))
                tar.add(os.path.join(WORKDIR, task_id, filename), arcname=md5_value)
                # modify the urls in result.xml
                result_xml = result_xml.replace(f'<url>{filename}</url>', f'<url>{md5_value}</url>')
            with open(os.path.join(WORKDIR, task_id, 'result.xml'), 'w') as fp:
                fp.write(result_xml)
            tar.add(os.path.join(WORKDIR, task_id, 'result.xml'), arcname='result.xml')

        return send_from_directory(WORKDIR, f'{task_id}.tar.gz', as_attachment=True)


@app.route('/kill', methods=['POST', 'GET'])
def kill():
    task_id = request.values.get('taskId')
    if task_id is None:
        return '{"code": -1, "msg": "taskId is None"}'

    # check task status
    if not os.path.exists(os.path.join(WORKDIR, task_id)):
        return '{"code": -1, "msg": "workdir not found"}'
    if state_helper(task_id) != 'running':
        return '{"code": -1, "msg": "task is not running"}'

    pid = RedisPool().get(task_id)
    if pid is None:
        return '{"code": -1, "msg": "task not found"}'
    else:
        if kill_helper(task_id):
            return '{"code": 0}'
        else:
            return '{"code": -1, "msg": "system error"}'


def run_helper(task_id):
    # run and save the pid
    os.chdir(os.path.join(WORKDIR, task_id))
    sp = subprocess.Popen(CMDUtil.parse_param(COMMAND, {'task_id': task_id,
                                                        'task_dir': os.path.join(WORKDIR, task_id)
                                                        }), shell=True, cwd=os.path.join(WORKDIR, task_id))
    RedisPool().set(task_id, sp.pid)
    os.chdir(WORKDIR)


def kill_helper(task_id):
    pid = RedisPool().get(task_id)
    try:
        if PLATFORM == 'windows':
            os.popen(f'taskkill /f /pid {pid}')
        else:
            os.kill(int(pid), signal.SIGKILL)
        with open(os.path.join(WORKDIR, task_id, 'log.txt'), 'a') as fp:
            fp.write('killed\n')
        return True
    except Exception:
        return False


def state_helper(task_id):
    # waiting, running, finish, killed
    with open(os.path.join(WORKDIR, task_id, 'log.txt'), 'r') as fp:
        state_str = fp.readlines()[-1][:-1]  # ignore /n
    return state_str


def daemon(r_queue, p_queue):
    while True:
        for i in range(PARALLELISM):
            # search for finished and killed task
            if not r_queue.empty():
                task_id = r_queue.get()
                if state_helper(task_id) not in ['finish', 'killed', 'error']:
                    r_queue.put(task_id)
            # put pending task to running
            if not p_queue.empty() and r_queue.qsize() < PARALLELISM:
                task_id = p_queue.get()
                r_queue.put(task_id)
                run_helper(task_id)
        print(f'{r_queue.qsize()} task(s) running and {p_queue.qsize()} task(s) pending.')
        time.sleep(60)


if __name__ == '__main__':
    # ignore status info of subprocess to avoid zombies
    if PLATFORM == 'linux':
        signal.signal(signal.SIGCHLD, signal.SIG_IGN)
    daemon_process = multiprocessing.Process(target=daemon, args=(running, pending,))
    daemon_process.start()
    app.run('0.0.0.0', PORT, threaded=True)
