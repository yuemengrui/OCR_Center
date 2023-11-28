# *_*coding:utf-8 *_*
import re


def idcard_parse(result):
    res = {}
    res['name'] = ''
    res['gender'] = ''
    res['birth'] = ''
    res['id'] = ''
    res['address'] = ''
    res['nation'] = ''

    text = ''.join(result)

    ret = re.search(r'姓名(.+)性别', text)
    if ret is not None:
        res['name'] = ret.group(1)
        text = text.replace('姓名', '').replace(res['name'], '')

    ret = re.search(r'性别(.+)民族', text)
    if ret is not None:
        res['gender'] = ret.group(1)
        text = text.replace('性别', '').replace(res['gender'], '')

    ret = re.search(r'民族(.+)出生', text)
    if ret is not None:
        res['nation'] = ret.group(1)
        text = text.replace('民族', '').replace(res['nation'], '')

    ret = re.search(r'出生(.+)住址', text)
    if ret is not None:
        res['birth'] = ret.group(1)
        text = text.replace('出生', '').replace(res['birth'], '')

    ret = re.search(r'住址(.+)公民身份', text)
    if ret is not None:
        res['address'] = ret.group(1)
        text = text.replace('住址', '').replace(res['address'], '')

    ret = re.search(r'号码(.+)', text)
    if ret is not None:
        res['id'] = ret.group(1)

    return res
