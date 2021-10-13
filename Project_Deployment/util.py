# coding=utf-8
import hashlib
import re
from xml.dom import minidom

import redis


class CMDUtil:
    @staticmethod
    def parse_param(cmd: str, task_dir: dict) -> str:
        for key in task_dir:
            cmd = cmd.replace(f'{{{key}}}', task_dir[key])
        return cmd


class MD5Util:
    @staticmethod
    def get_md5(file: str) -> str:
        md5_tool = hashlib.md5()
        with open(file, 'rb') as fp:
            while True:
                buff = fp.read(8192)
                if not buff:
                    break
                md5_tool.update(buff)
        return md5_tool.hexdigest()


class RedisPool:
    REDIS_HOST = 'redis'
    REDIS_PORT = 6379

    def __init__(self):
        self.redis_pool = redis.ConnectionPool(host=RedisPool.REDIS_HOST, port=RedisPool.REDIS_PORT, db=6,
                                               decode_responses=True)

    def get(self, name: str) -> str:
        return redis.StrictRedis(connection_pool=self.redis_pool).get(name)

    def set(self, name: str, value: str):
        redis.StrictRedis(connection_pool=self.redis_pool).set(name, value)


class XMLNode:
    ELEM_NODE = 1
    TEXT_NODE = 3

    def __init__(self, node):
        self.node = node

    def name(self) -> str:
        return self.node.nodeName

    def type(self) -> int:
        for node in self.node.childNodes:
            if node.nodeType == XMLNode.ELEM_NODE:
                return XMLNode.ELEM_NODE
        return XMLNode.TEXT_NODE

    def data(self) -> str or None:
        if self.type() == XMLNode.TEXT_NODE:
            for node in self.node.childNodes:
                if re.fullmatch(r'^[\s]*$', node.data) is None:
                    return node.data
            return ''
        return None

    def children(self) -> list:
        ret = []
        for node in self.node.childNodes:
            if node.nodeType == XMLNode.ELEM_NODE:
                ret.append(XMLNode(node))
        return ret


class XMLUtil:
    @staticmethod
    def parse(file: str) -> XMLNode:
        return XMLNode(minidom.parse(file).documentElement)
